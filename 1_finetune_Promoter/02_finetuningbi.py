import os
import argparse
import pandas as pd

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from torch.utils.tensorboard import SummaryWriter  

import sys
sys.path.append('../')
from lib.dataset.vocab import gene_vocab
from lib.dataset.dataset import MambaDataset
from lib.model.DNAmamba.biDNAmamba import MambaConfig, MambaLMHeadModel_sft
from lib.trainer.pretrain import MambaTrainer


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default= "pretrain", help="pretrain/finetune")
    parser.add_argument("--distributed-mode", type=str, default="DDP", help="DDP/DataParallel/SingleKernal")
    parser.add_argument("--local-rank", type=int, default= -1, help="number of cpu threads to use during batch generation")
    parser.add_argument("--ngram", type=int, default=3, help="n_gram")
    parser.add_argument("--seq-size", type=int, default=1000, help="sequence's length")
    parser.add_argument("--corpus-lines", type=int, default=None, help="number of sequence in corpus")
    parser.add_argument("--num-workers", type=int, default=0, help="dataloader worker size")
    parser.add_argument("--d-model", type=int, default=768, help="hidden state's size")
    parser.add_argument("--n-layer", type=int, default=24, help="number of layer")
    parser.add_argument("--lr", type=float, default=1e-4 , help="learning rate")
    parser.add_argument("--betas", type=tuple, default=(0.9, 0.999), help="parameter of adamw")
    parser.add_argument("--weight-decay", type=float, default=1e-2, help="parameter of adamw 正则化")
    parser.add_argument("--with-cuda", type=bool, default=True, help="switch of using GPU")
    parser.add_argument("--batch-size", type=int, default=256, help="batch size")    
    parser.add_argument("--epochs", type=int, default=20, help="epoch")
    parser.add_argument("--verbose-steps", type=int, default=100 , help="每个epoch隔几个step打印一次")  
    parser.add_argument("--save-steps-interval", type=int, default=1, help="一个epoch存储几次临时模型 1=不存临时模型")
    parser.add_argument("--pretrain-seq-size", type=int, default=1000, help="pretrain sequence's length")
    parser.add_argument("--kfold", type=int, default=10, help="k for kfold")    
    args=parser.parse_args()


    task = args.task
    distributed_mode = args.distributed_mode # 该参数控制使用DDP(DistributedDataParallel)或者DataParallel
    local_rank=args.local_rank # DDP所需参数,由DDP自动传入
    if distributed_mode == "DDP": # DDP
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group('nccl', init_method='env://')
        os.environ['RANK'] = str(0) # 设置环境 必须放在这三条语句最后，否则死锁


    ngram=args.ngram
    pretrain_seq_size=args.pretrain_seq_size  # DNA seq 的长度
    seq_size=args.seq_size  # DNA seq 的长度
    seq_len =  seq_size // ngram # token seq 的长度
    corpus_lines=args.corpus_lines  # None表示自动读取语料库的大小
    num_workers=args.num_workers # dataloader worker size 注意采用即时读取预料库的方法由于指针关系无法并行读取

    d_model=args.d_model
    n_layer=args.n_layer
    lr=args.lr   # 微调为e-5级别(比预训练要小)
    betas=args.betas
    weight_decay=args.weight_decay
    with_cuda=args.with_cuda
    batch_size=args.batch_size
    epochs=args.epochs
    verbose_steps=args.verbose_steps             # 每个epoch隔几个step打印一次
    save_steps_interval=args.save_steps_interval  # 一个epoch存储几次临时模型 1=不存临时模型 

    kfold = args.kfold

    data_path = "./data/corpus"
    task_name = 'epdnew_BOTH'
    dataset_flags = ["train", "valid", "test"]
    

    data_file_name = "{}_{}_gram_{}_fold_{}.txt".format(task_name, ngram, 0, dataset_flags[0])
    data_label_file_name = "{}_{}_gram_label_{}_fold_{}.txt".format(task_name,ngram, 0, dataset_flags[0])
    train_corpus_path = os.path.join(data_path,"{}_gram".format(ngram), task_name, data_file_name)
    train_label_path = os.path.join(data_path,"{}_gram".format(ngram), task_name, data_label_file_name)
    
    data_file_name = "{}_{}_gram_{}_fold_{}.txt".format(task_name, ngram, 0, dataset_flags[1])
    data_label_file_name = "{}_{}_gram_label_{}_fold_{}.txt".format(task_name,ngram, 0, dataset_flags[1])
    valid_corpus_path = os.path.join(data_path,"{}_gram".format(ngram), task_name, data_file_name)
    valid_label_path = os.path.join(data_path,"{}_gram".format(ngram), task_name, data_label_file_name)

    pretrain_directory = "./data/model/pretrain/bi_{}_gram_{}_seq_size_{}_layer_{}_dim".format(ngram, pretrain_seq_size, n_layer, d_model)
    
    save_directory = "./data/model/finetune/bi_{}_gram_{}_seq_size_{}_layer_{}_dim_{}_fold_{}".format(ngram, seq_size, n_layer, d_model, task_name, 0)
    logging_filename = "train_log.txt"
    result_path = "./data/result/bimamba_result_{}_gram.xlsx".format(ngram)

    # 获取tokenizer
    print("\nLoading Vocab")
    vocab = gene_vocab(n_gram = ngram)
    word_dict_alphabet = vocab.word_dict_alphabet
    print("Vocab Size: ", vocab.vocab_size)

    # 获取数据集
    print("\nLoading Train Dataset\n", train_corpus_path, "\n", train_label_path)
    train_dataset = MambaDataset(train_corpus_path, vocab, seq_len, corpus_lines = corpus_lines, task = task, labels_path = train_label_path, labels_lines=corpus_lines)
    print("Loading Validation Dataset\n", valid_corpus_path, "\n", valid_label_path)
    valid_dataset = MambaDataset(valid_corpus_path, vocab, seq_len, corpus_lines = corpus_lines, task = task, labels_path = valid_label_path, labels_lines=corpus_lines)

    print("\nCreating Dataloader")
    print("Batch size: ", batch_size)
    if distributed_mode == "DDP": # DDP
        train_sampler = DistributedSampler(train_dataset,shuffle=True) # DataLoader中的shuffle应该设置为False，因为打乱的任务交给了sampler 
        train_data_loader = DataLoader(dataset=train_dataset,batch_size= batch_size, num_workers=num_workers, shuffle=False, drop_last=True,sampler=train_sampler)
        valid_data_loader = DataLoader(dataset=valid_dataset,batch_size= batch_size, num_workers=num_workers, shuffle=False, drop_last=False) # 不需要并行计算（保存最佳模型）      
    else:  # DataParallel / SingeleKernal
        train_data_loader = DataLoader(dataset=train_dataset,batch_size= batch_size, num_workers=num_workers, shuffle=True, drop_last=True)
        valid_data_loader = DataLoader(dataset=valid_dataset,batch_size= batch_size, num_workers=num_workers, shuffle=False, drop_last=False)

    # print(train_dataset.__len__())
    # output = train_dataset.__getitem__(3)
    # print(output)           
        
    print("\nBuilding mamba model")
    # config = MambaConfig(d_model=d_model, n_layer=n_layer, vocab_size=vocab.vocab_size, ssm_cfg={},)
    # mamba = MambaLMHeadModel_sft(config, dtype=torch.bfloat16, device="cuda")
    # mamba.replace_head(n_output=2,**{"device": "cuda", "dtype": torch.bfloat16})

    mamba = MambaLMHeadModel_sft.from_pretrained(pretrain_directory, dtype=torch.bfloat16, device="cuda", n_output=2) # 加载模型 n=2=二分类 
    mamba.backbone.bimamba = True  # 开启双向

    print("\nCreating mamba Trainer")  # 在此加载预训练模型
    trainer = MambaTrainer(mamba = mamba, vocab = vocab,
                    train_dataloader=train_data_loader, test_dataloader =valid_data_loader,
                    lr = lr, betas=betas, weight_decay = weight_decay, 
                    distributed_mode = distributed_mode, with_cuda = with_cuda, local_rank = local_rank,
                    verbose_steps = verbose_steps, save_steps = train_dataset.corpus_lines//batch_size//save_steps_interval,
                    save_directory = save_directory, logging_filename = logging_filename, 
                    save_best_only = True, save_best_eval="acc", early_stop=5,
                    logits_mode = "last", label_mode= "muiticlass",
                    )

    print("Training Start")  # 删除多余的model文件夹
    for epoch in range(epochs):
        trainer.train(epoch)
        trainer.test(epoch)

    # 测试
    if distributed_mode == "DDP": # DDP保证同步，防止出现模型读取失败
        print("synchronizing local_rank=", local_rank)
        torch.distributed.barrier(device_ids=[local_rank])

    data_file_name = "{}_{}_gram_{}_fold_{}.txt".format(task_name, ngram, 0, dataset_flags[2])
    data_label_file_name = "{}_{}_gram_label_{}_fold_{}.txt".format(task_name,ngram, 0, dataset_flags[2])
    test_corpus_path = os.path.join(data_path,"{}_gram".format(ngram), task_name, data_file_name)
    test_label_path = os.path.join(data_path,"{}_gram".format(ngram), task_name, data_label_file_name)
    
    print("\nLoading Test Dataset\n", test_corpus_path, "\n",test_label_path)
    test_dataset = MambaDataset(test_corpus_path, vocab, seq_len, corpus_lines = corpus_lines, task = task, labels_path = test_label_path, labels_lines=corpus_lines)       
    
    print("\nCreating Dataloader")
    print("Batch size: ", batch_size)
    test_data_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, drop_last=False) 

    print("\nBuilding mamba model")
    mamba = MambaLMHeadModel_sft.from_funetuned(os.path.join(save_directory, "epoch_model","best"), dtype=torch.bfloat16, device="cuda", n_output=2)
    
    print("\nCreating mamba Trainer")  # 在此加载预训练模型
    trainer = MambaTrainer(mamba = mamba, vocab = vocab,
                    train_dataloader=train_data_loader, test_dataloader =test_data_loader,
                    lr = lr, betas=betas, weight_decay = weight_decay, 
                    distributed_mode = distributed_mode, with_cuda = with_cuda, local_rank = local_rank,
                    verbose_steps = verbose_steps, save_steps = train_dataset.corpus_lines//batch_size//save_steps_interval,
                    save_directory = save_directory, logging_filename = logging_filename, 
                    save_best_only = False, save_best_eval="acc", 
                    logits_mode = "last", label_mode= "muiticlass",
                    ) # trainer.test() + save_best_only = False = 不保存模型

    print("\nTesting Start")  # 删除多余的model文件夹
    trainer.test(epoch = 0)

    result = trainer.result
    accuracy, P, R, F1score, AUROC, AUPRC = result["accuracy"], result["P"], result["R"], result["F1score"], result["AUROC"], result["AUPRC"]
    print("\n accuracy:{} | P:{} | R:{} | F1score:{} | AUROC:{} | AUPRC:{} |".format(accuracy, P, R, F1score, AUROC, AUPRC))

    if distributed_mode == "DDP": # DDP保证同步
        print("synchronizing local_rank=", local_rank)
        torch.distributed.barrier(device_ids=[local_rank])



