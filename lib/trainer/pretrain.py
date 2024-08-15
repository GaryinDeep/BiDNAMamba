import numpy as np 
import os
import json
import sklearn
from sklearn.metrics import accuracy_score # 添加此句以解决sklearn隐藏错误

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam, AdamW, lr_scheduler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from torch.utils.tensorboard import SummaryWriter  

from lib.dataset.vocab import gene_vocab
from lib.model.DNAmamba.DNAmamba import MambaLMHeadModel

import tqdm

class MambaTrainer:

    def __init__(self, mamba: MambaLMHeadModel, vocab:gene_vocab,
                train_dataloader: DataLoader, test_dataloader: DataLoader = None,
                lr: float = 1e-3, betas=(0.9, 0.999), weight_decay: float = 1e-2,
                T_0: int = 1000, T_mult: int = 1, eta_min: float = 1e-5, 
                distributed_mode = "SingleKernal", with_cuda: bool = True, local_rank=None,
                verbose_steps: int = 10, save_steps: int = None, 
                save_directory: str = None, logging_filename = "train_log.txt", 
                train_writer: SummaryWriter =None, val_writer: SummaryWriter =None,
                save_best_only = False, save_best_eval="acc", early_stop = None,
                logits_mode = "whole", label_mode= "muiticlass",input_dic_format= "mamba", 
                ):
        """
        :param mamba: mamba model which you want to train
        :param vocab: total word vocab 
        :param train_dataloader: train dataset data loader
        :param test_dataloader: test dataset data loader [can be None]
        :param lr: learning rate of optimizer
        :param betas: Adam optimizer betas
        :param weight_decay: Adam optimizer weight decay param
        :param T_0: Number of iterations for the first restart with CosineAnnealingWarmRestarts 
        :param T_mult: A factor increases Ti after a restart with CosineAnnealingWarmRestarts 
        :param eta_min: Minimum learning rate with CosineAnnealingWarmRestarts
        :param with_cuda: traning with cuda
        :param verbose_steps: verbose frequency of the batch iteration
        :param logging_filename: log saving path 
        :param trained_model_path: storing trained model path 
        :param save_best_eval: "acc", "loss"
        """

        """
        train_flag  save_best_only 
        True        True            train save best model 
        True        False           train save model each epoch
        False       True            validation
        False       False           test
        

        """

        distributed_mode_ParamList = ["DDP", "DataParallel", "SingleKernal"]
        save_best_eval_ParamList = ["acc", "loss"]
        logits_mode_paramList = ["whole", "last"]
        label_mode_ParamList = ["muiticlass", "muititask"]
        input_dic_format_ParamList = ["mamba", "bert"]

        assert distributed_mode in distributed_mode_ParamList, 'error value of param distributed_mode:{}'.format(distributed_mode)
        assert save_best_eval in save_best_eval_ParamList, 'error value of param save_best_eval:{}'.format(save_best_eval)
        assert logits_mode in logits_mode_paramList, 'error value of param logits_mode:{}'.format(logits_mode)
        assert label_mode in label_mode_ParamList, 'error value of param label_mode:{}'.format(label_mode)
        assert input_dic_format in input_dic_format_ParamList, 'error value of param input_dic_format:{}'.format(input_dic_format)

        self.vocab = vocab
        # This mamba model will be saved every epoch
        self.model = mamba
    
        # Setting the train and test data loader
        self.train_data = train_dataloader
        self.test_data = test_dataloader

        self.verbose_steps = verbose_steps
        self.logging_filename = logging_filename
        self.save_steps = save_steps
        self.save_directory = save_directory
        self.train_writer = train_writer
        self.val_writer = val_writer
        self.global_step = 0

        # Optimizer and Scheduler
        self.optim = AdamW(params=self.model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        self.optim_schedule = lr_scheduler.CosineAnnealingWarmRestarts(self.optim, T_0=T_0, T_mult=T_mult, eta_min=eta_min)

        # Using cross entropy Loss function
        self.logits_mode = logits_mode
        self.label_mode = label_mode
        self.input_dic_format = input_dic_format
        if self.label_mode == "muiticlass":
            self.criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=self.vocab.pad_id)
        elif self.label_mode == "muititask":
            self.criterion = nn.BCEWithLogitsLoss(reduction='mean', ignore_index=self.vocab.pad_id)

        # Setup cuda device for mamba training : Distributed GPU training 
        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))
        self.distributed_mode = distributed_mode
        self.local_rank = local_rank
        self.with_cuda = with_cuda
        if self.distributed_mode == "DDP": # DDP 
            self.device = torch.device(f'cuda:{local_rank}')
            self.model = self.model.to(self.device) # 存储到GPU
            self.model = DDP(self.model, device_ids=[local_rank]) 
            # self.model = DDP(self.model, device_ids=[local_rank],output_device=[local_rank],find_unused_parameters=True) # 存在冗余参数时
        elif self.distributed_mode == "DataParallel": # DataParallel
            self.device = torch.device("cuda:0") # 此处设置主卡
            self.model = self.model.to(self.device) # 存储到GPU
            num_of_gpus = torch.cuda.device_count()
            if self.with_cuda and num_of_gpus > 1:
                print("Using %d GPUS for mamba" % num_of_gpus)
                self.model = nn.DataParallel(self.model, device_ids=list(range(num_of_gpus))) # 指定哪些卡用于训练 必须包含主卡(self.device)
        elif self.distributed_mode == "SingleKernal":
            self.device = torch.device("cuda:0" if (torch.cuda.is_available() and self.with_cuda) else "cpu")
            self.model = self.model.to(self.device) # 存储到GPU
            
        self.save_model_flag = True 
        if self.distributed_mode == "DDP" and self.local_rank != 0:    # DDP存储模型时通过local_rank限制单进程写入,只需保存设备0中的模型:
            self.save_model_flag = False
        
        self.save_best_only = save_best_only
        if save_best_only:
            self.save_best_eval = save_best_eval
            self.max_acc = float('-inf')
            self.min_loss = float('inf')

        # self.early_stop= early_stop
        # self.n_stag = 0
        # self.early_stoping_flag = False
        # if save_best_only and early_stop is not None:
        #     print("Using early stop for training")

        print("trainer initialization completed")

    def train(self, epoch):
        self.model.train()
        self.train_flag=True  # 注意train=True
        self.iteration(epoch, self.train_data)  

    def test(self, epoch):
        self.model.eval()
        self.train_flag=False
        self.iteration(epoch, self.test_data)

    def iteration(self, epoch, data_loader):
        """
        loop over the data_loader for training or testing
        if on train status, backward operation is activated
        and also auto save the model every peoch

        :param epoch: current epoch index
        :param data_loader: torch.utils.data.DataLoader for iteration
        :return: None
        """
        str_code = "train" if self.train_flag else "test"

        # Setting the tqdm progress bar
        data_iter = tqdm.tqdm(enumerate(data_loader),
                              desc="EP_%s:%d" % (str_code, epoch),
                              total=len(data_loader),
                              bar_format="{l_bar}{r_bar}")
        avg_loss = 0.0
        total_correct = 0
        total_element = 0
        all_outputs = []    
        all_labels= []
        all_sample_number = []
        for i, data in data_iter:
            # 0. batch_data will be sent into the device(GPU or cpu)
            data = {key: value.to(self.device) for key, value in data.items()}  # 放入CUDA
            
            if self.input_dic_format == "mamba":
                lm_input = data["mamba_input"]  # (batch_size, seq_len)
                lm_label = data["mamba_label"]  # (batch_size, seq_len) / (batch_size, n_label)
                sample_number = data.get("sample_number", None)
                # 1. forward the prediction
                if self.logits_mode == "whole":
                    lm_logits = self.model.forward(lm_input).logits # (batch_size, seq_len, n_output)
                elif self.logits_mode == "last":
                    lm_logits = self.model.forward(lm_input,num_last_tokens=1).logits # (batch_size, 1, n_output)

            elif self.input_dic_format == "bert":
                lm_input = data["bert_input"]  # (batch_size, seq_len)
                segment_label = data["segment_label"]  # (batch_size, seq_len)
                lm_label = data["bert_label"]   # (batch_size, seq_len)
                is_next = data["is_next"]   # (batch_size,1)      
                sample_number = data.get("sample_number", None)    
                # 1. forward the prediction
                if self.logits_mode == "whole":
                    lm_logits = self.model.forward(lm_input, segment_label) # (batch_size, seq_len, n_output)  
                elif self.logits_mode == "last":
                    lm_logits = self.model.forward(lm_input, segment_label, num_last_tokens=1) # (batch_size, 1, n_output)  不需要next sentence prediction

            # 2. Adding loss
            if self.label_mode == "muiticlass":
                loss = self.criterion(lm_logits.contiguous().view(-1, lm_logits.size(-1)), lm_label.contiguous().view(-1)) 
            elif self.label_mode == "muititask":
                loss = self.criterion(lm_logits.contiguous().view(-1), lm_label.view(-1))

            # 3. backward and optimization only in train
            if self.train_flag:
                self.optim.zero_grad()     # 梯度归零  
                loss.backward()            # 后传
                self.optim.step()          # 更新权重
                self.optim_schedule.step() # 按schedule更新lr
            
            # prediction loss
            avg_loss += loss.item()
            # prediction accuracy  计算token的预测准确率         
            if self.label_mode == "muiticlass":
                lm_output =  torch.softmax(lm_logits.contiguous().view(-1, lm_logits.size(-1)), dim=-1) # (batch_size*seq_len, vocab_size) / (batch_size, 2)
                lm_label = lm_label.view(-1)                                            # (batch_size*seq_len) / (batch_size)
                mask_index = torch.where(torch.ne(lm_label, self.vocab.pad_id))[0]   # 获得掩膜的位置
                lm_output = lm_output[mask_index] # (n, vocab_size) / (n, 2)
                lm_label = lm_label[mask_index]   # (n)

                correct = lm_output.argmax(dim=-1).eq(lm_label).sum().item()

            elif self.label_mode == "muititask":
                lm_output = torch.sigmoid(lm_logits.contiguous().view(-1)) # (batch_size*n_label)
                lm_label = lm_label.view(-1)               # (batch_size*n_label)
                mask_index = torch.where(torch.ne(lm_label, self.vocab.pad_id))[0]   # 获得掩膜的位置
                lm_output = lm_output[mask_index]  # (n)
                lm_label = lm_label[mask_index]    # (n)   
                
                threshold = 0.5
                correct = torch.gt(lm_output, threshold).eq(lm_label).sum().item()
     
            element = lm_label.nelement()  # tensor (张量)中 元素的个数
            total_correct += correct
            total_element += element    
            # 打印输出
            if i % self.verbose_steps == 0: # 每几个batch打印一次
                post_fix = {"epoch": epoch,"iter": i,
                            "avg_loss": "{:.4f}".format(avg_loss / (i + 1)),
                            "avg_acc": "{:.4f}%".format(total_correct / total_element * 100),
                            "loss": "{:.4f}".format(loss.item()),
                            "acc": "{:.4f}%".format(correct/element*100), }                
                data_iter.write(str(post_fix))   # 打印

                ## tensorboard
                # if self.save_model_flag:
                #     if self.train_flag
                #         self.train_writer.add_scalar(tag='loss', scalar_value=avg_loss / (i + 1), global_step=self.global_step)
                #     else:
                #         self.val_writer.add_scalar(tag='loss', scalar_value=avg_loss / (i + 1), global_step=self.global_step) 

                # # 保存训练loss
                # if self.train_flag and self.save_model_flag:
                #     with open("./data/bi_pretrain_loss", "a") as f: # 追加模式
                #         f.write(str(avg_loss / (i + 1))+"\n") # 自动关闭 

            # 保存临时模型     
            if (self.train_flag) and (self.save_model_flag) and (self.save_steps != None) and (i!=0) and (i % self.save_steps == 0):
                self.save_model(os.path.join(self.save_directory, "steps_model","epoch_{}_step_{}".format(epoch, i)))                 
                
            # 收集测试结果
            if not self.train_flag and not self.save_best_only:  # test
                all_outputs += lm_output.detach().cpu().tolist() # (n, vocab_size) / (n, 2) / (n)
                all_labels +=  lm_label.detach().cpu().tolist() # (n)
                all_sample_number += sample_number.detach().cpu().tolist() if sample_number is not None else None

            # 清理中间变量, otherwise out of memory while test
            del lm_input, lm_label
            del lm_logits, loss, lm_output, mask_index
            torch.cuda.empty_cache() if self.with_cuda else None # 清理显存
            
            self.global_step +=1

        # Calculate perplexity for the epoch
        try:
            perplexity = np.exp( avg_loss / len(data_iter) )
        except OverflowError:
            perplexity = float("-inf")   

        print("EP{}_{} | avg_loss={:.4f} | avg_acc={:.4f} | train_perplexity={:.4f}".format(
            epoch, str_code,avg_loss / len(data_iter),total_correct * 100.0 / total_element, perplexity) )

        # 提前停止模块
        # if not self.train_flag and self.save_best_only and self.early_stop is not None: # valid
        #     if self.save_best_eval=="acc":
        #         if (total_correct * 100.0 / total_element) > self.max_acc:
        #             self.n_stag = 0 
        #         else: 
        #             self.n_stag += 1  # 停滞的迭代数+1
        #     elif self.save_best_eval=="loss":
        #         if (avg_loss / len(data_iter)) < self. min_loss:
        #             self.n_stag = 0 
        #         else:
        #             self.n_stag += 1
        #     if self.n_stag>=self.early_stop:  # 提前停止
        #         print("early stoping")
        #         self.early_stoping_flag = True

        # train存储模型以及记录日志
        if self.save_model_flag:  
            # Ensure save_directory exists
            if not os.path.exists(os.path.join(self.save_directory, "epoch_model")): 
                    os.makedirs(os.path.join(self.save_directory, "epoch_model"))
            # 保存记录日志
            if self.train_flag:   # train
                log_txt_formatter = "Train [Epoch] {epoch_str:03d} [Loss] {loss_str} [Acc] {Acc_str}%\n"
            elif self.save_best_only: # valid
                log_txt_formatter = "Valid [Epoch] {epoch_str:03d} [Loss] {loss_str} [Acc] {Acc_str}%\n"
            else: # test
                log_txt_formatter = None
            if log_txt_formatter is not None :
                with open(os.path.join(self.save_directory, "epoch_model", self.logging_filename), "a") as f: # 追加模式
                    to_write = log_txt_formatter.format(epoch_str = epoch,
                                                        loss_str=avg_loss / len(data_iter),
                                                        Acc_str = total_correct * 100.0 / total_element )
                    f.write(to_write) # 自动关闭     
            # 保存模型
            if self.train_flag: # train
                if not self.save_best_only:
                    if epoch==0:
                        self.save_model(os.path.join(self.save_directory, "epoch_model","epoch_{}".format(epoch)), save_config=True)
                    else:
                        self.save_model(os.path.join(self.save_directory, "epoch_model","epoch_{}".format(epoch)))
            elif self.save_best_only: # valid
                if self.save_best_eval=="acc":
                    if (total_correct * 100.0 / total_element) > self.max_acc:
                        self.max_acc = (total_correct * 100.0 / total_element)
                        self.save_model(os.path.join(self.save_directory, "epoch_model","best"), save_config=True)
                elif self.save_best_eval=="loss":
                    self.save_model(os.path.join(self.save_directory, "epoch_model","best"), save_config=True)
                   # if (avg_loss / len(data_iter)) < self. min_loss:
                   #     self.min_loss = (avg_loss / len(data_iter))
                   #     self.save_model(os.path.join(self.save_directory, "epoch_model","best"), save_config=True)
                else:
                    print("eval metric flag error")

        # test计算结果的评价指标
        if not self.train_flag and not self.save_best_only:  # test
            self.result = self.calculate_scores(all_outputs, all_labels, all_sample_number)        
            del all_outputs, all_labels, all_sample_number

    def save_model(self, save_directory, save_config=False):  

        # Ensure save_directory exists
        os.makedirs(save_directory) if not os.path.exists(save_directory) else None

        # Save the model's state_dict
        model_path = os.path.join(save_directory, 'pytorch_model.bin')
        print("Model Saved on:", model_path)
        if self.distributed_mode == "DDP" or self.distributed_mode == "DataParallel":
            torch.save(self.model.module.state_dict(), model_path)  # ddp后需要保存model.module
        elif self.distributed_mode == "SingleKernal":
            torch.save(self.model.state_dict(), model_path)
        
        # Save the configuration of the model
        if save_config:
            config_path = os.path.join(save_directory, 'config.json')
            print("Config Saved on:", config_path)
            with open(config_path, 'w') as f:
                if self.distributed_mode == "DDP" or self.distributed_mode == "DataParallel":
                    json.dump(self.model.module.config.__dict__, f)
                elif self.distributed_mode == "SingleKernal":   
                    json.dump(self.model.config.__dict__, f)     


    def calculate_scores(self, all_outputs, all_labels, all_sample_number, pos_label=1):
        """
        :param label_mode: 选择标签的模式 
        :param pos_label: 选择画出哪个类
        """
        if self.label_mode == "muiticlass":
            all_labels = np.array(all_labels) 
            all_outputs = np.array(all_outputs) 
            all_pred_int = np.argmax(all_outputs, axis=-1)

            if all_outputs.shape[1] <= 2:  #二分类
                accuracy = sklearn.metrics.accuracy_score(all_labels, all_pred_int)    # 准确率 
                P = sklearn.metrics.precision_score(all_labels, all_pred_int)
                R = sklearn.metrics.recall_score(all_labels, all_pred_int)
                F1score = sklearn.metrics.f1_score(all_labels, all_pred_int)
                AUROC = sklearn.metrics.roc_auc_score(all_labels, all_outputs[:,pos_label])
                AUPRC = (sklearn.metrics.average_precision_score(all_labels, all_outputs[:,pos_label])) 
                False_Positive_Rates, True_Positive_Rates, thresholds = sklearn.metrics.roc_curve(all_labels, all_outputs[:,pos_label], pos_label=pos_label) 
                Precision_Rate, Recall_Rate, thresholds = sklearn.metrics.precision_recall_curve(all_labels, all_outputs[:,pos_label], pos_label=pos_label)

            else:  # 多分类
                accuracy = sklearn.metrics.accuracy_score(all_labels, all_pred_int)    # 准确率 
                P = sklearn.metrics.precision_score(all_labels, all_pred_int,  average='macro')
                R = sklearn.metrics.recall_score(all_labels, all_pred_int,  average='macro')
                F1score = sklearn.metrics.f1_score(all_labels, all_pred_int,  average='macro')
                AUROC = sklearn.metrics.roc_auc_score(all_labels, all_outputs, multi_class  ='ovr')
                AUPRC = sklearn.metrics.average_precision_score(all_labels, all_outputs, multi_class  ='ovr')  
                False_Positive_Rates, True_Positive_Rates, thresholds = sklearn.metrics.roc_curve(all_labels, all_outputs[:,pos_label], pos_label=pos_label) 
                Precision_Rate, Recall_Rate, thresholds = sklearn.metrics.precision_recall_curve(all_labels, all_outputs[:,pos_label], pos_label=pos_label)

            return {"accuracy":accuracy, "P":P, "R":R, "F1score":F1score, "AUROC":AUROC,"AUPRC":AUPRC,}
            
        elif self.label_mode == "muititask":
            all_pred_int = np.where(all_outputs>0,1,0)



