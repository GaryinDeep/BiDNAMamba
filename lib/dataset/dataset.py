from torch.utils.data import Dataset
import tqdm
import torch
import random

"""
注意：要求语料库每行长度一致
"""

class MambaDataset(Dataset):
    def __init__(self, corpus_path, vocab, seq_len, encoding="utf-8", corpus_lines=None,
                task = "pretrain", labels_path = None, labels_lines=None):
        self.vocab = vocab
        self.encoding = encoding
        self.task = task    # 任务模式
        self.seq_len = seq_len

        # lm
        self.corpus_path = corpus_path
        self.corpus_lines = corpus_lines
        self.line_offset= None   # 语料库中每行的偏移量
        
        # 获取行偏移量 注意:该用法需要每行长度一致
        with open(self.corpus_path, "r", encoding=encoding) as f:
            line = f.readline()
            self.line_offset = len(line)  # 行偏移量
        # 获取数据集的数据总量
        with open(self.corpus_path, "r", encoding=encoding) as f:
            if self.corpus_lines is None:  
                self.corpus_lines = 0
                for line in tqdm.tqdm(f, desc="Loading Dataset", total=corpus_lines):
                    self.corpus_lines += 1 
        print("offset: ", self.line_offset, end=" ")
        print("corpus_lines: ", self.corpus_lines)
        # 读取预料库
        self.corpus = open(self.corpus_path, "r", encoding=self.encoding)
        
        # sft 
        if self.task != "pretrain" and labels_path is not None:
            self.labels_path = labels_path
            self.labels_lines = labels_lines
            self.line_offset_label= None

            with open(self.labels_path, "r", encoding=encoding) as f:
                line = f.readline()
                self.line_offset_label = len(line)  # 行偏移量
            # 获取数据集的数据总量
            with open(self.labels_path, "r", encoding=encoding) as f:
                if self.labels_lines is None:  
                    self.labels_lines = 0
                    for line in tqdm.tqdm(f, desc="Loading Label Dataset", total=labels_lines):
                        self.labels_lines += 1 
            print("offset_label: ", self.line_offset_label, end=" ")
            print("labels_lines: ", self.labels_lines)
            if self.labels_lines != self.corpus_lines:
                print("errpr: labels number do not match corpus number")
            # 读取预料库
            self.labels = open(self.labels_path, "r", encoding=self.encoding)

    def __len__(self):  
        return self.corpus_lines # Dataset每个epoch加载出来的数据个数

    def get_corpus_line(self, item): # 随机读取
        self.corpus.seek(item* self.line_offset, 0)  # 0表示从开头开始偏移
        line = self.corpus.readline()
        if len(line) != self.line_offset:
            print("warming:  line'len not match ") 
        return line.replace(self.vocab.corpus_pad, "")[:-1]    # 去除padding以及\n         

    def get_label_line(self, item): # 随机读取
        self.labels.seek(item* self.line_offset_label, 0)  # 0表示从开头开始偏移
        line = self.labels.readline()
        if len(line) != self.line_offset_label:
            print("warming:  line'len not match ") 
        return line[:-1]    # 去除\n         

    def __getitem__(self, item):
       
        if self.task != "pretrain" and self.labels_path is not None: # finetuning with sft
            # 取sentence
            seq  = [int(i) for i in self.get_corpus_line(item).split()] 
            # 加上 [SEP]
            tokens = seq + [self.vocab.sep_token_id]
            seq_len = self.seq_len +1  # 加上 [SEP]
            # 截断
            mamba_input = tokens[:seq_len]
            # 补齐
            padding = [self.vocab.pad_id for _ in range(seq_len - len(mamba_input))]
            mamba_input.extend(padding)   
            
            # 取label
            label = self.get_label_line(item).split()
            sample_number = int(label[0].split("_")[-1])
            mamba_label = [int(i) for i in label[1:]] 

            output = {"mamba_input": mamba_input, # (len+1)
                    "mamba_label": mamba_label,   # (n_task)
                    "sample_number": sample_number}  # (1)

        else:  # pretraining or finetuning with lm
            # 取sentence
            seq  = [int(i) for i in self.get_corpus_line(item).split()] 
            # lm 
            tokens = seq[:-1] 
            labels = seq[1:]
            seq_len = self.seq_len - 1 # lm
            # 截断
            mamba_input = tokens[:seq_len]
            mamba_label = labels[:seq_len]
            # 补齐
            padding = [self.vocab.pad_id for _ in range(seq_len - len(mamba_input))]
            mamba_input.extend(padding), mamba_label.extend(padding)

            output = {"mamba_input": mamba_input,         # (len-1)
                    "mamba_label": mamba_label,}          # (len-1)
        
        return {key: torch.tensor(value) for key, value in output.items()}




class BERTDataset(MambaDataset):

    def random_word(self, sentence):
        tokens = sentence  
        output_label = []
        for i, token in enumerate(tokens):
            prob = random.random()
            if prob < 0.15:
                prob /= 0.15
                # 80% randomly change token to mask token
                if prob < 0.8:
                    tokens[i] = self.vocab.mask_token_id
                # 10% randomly change token to random token
                elif prob < 0.9:
                    tokens[i] = random.randint(self.vocab.first_valid_token_id, self.vocab.last_valid_token_id) 
                # 10% randomly change token to current token
                else:
                    tokens[i] = token
                output_label.append(token)
            else:
                tokens[i] = token
                output_label.append(self.vocab.pad_id)   # 未被选中(未被掩膜)

        return tokens, output_label  

    def __getitem__(self, item):

        if self.task != "pretrain" and self.labels_path is not None: # finetuning with sft
            # 取sentence
            seq, is_next_label  = [int(i) for i in self.get_corpus_line(item).split()] , 0  #分隔符为默认值时，认为空格、\n、\t等都是分隔符            
            # 加上[CLS]  与 [SEP]
            tokens = [self.vocab.cls_token_id] + seq + [self.vocab.sep_token_id]
            seq_len = self.seq_len +2 # 加上[CLS]  与 [SEP]
            # 截断
            bert_input = tokens[:seq_len]
            segment_label = [0 for _ in range(len(tokens))][:seq_len]
            # 补齐
            padding = [self.vocab.pad_id for _ in range(seq_len - len(bert_input))]
            bert_input.extend(padding), segment_label.extend(padding)   
            
            # 取label
            label = self.get_label_line(item).split()
            sample_number = int(label[0].split("_")[-1])
            bert_label = [int(i) for i in label[1:]] 

            output = {"bert_input": bert_input,    # (len+2)
                    "bert_label": bert_label,      # (n_task)
                    "segment_label": segment_label, # (len+2)
                    "is_next": is_next_label,        # (1)
                    "sample_number": sample_number} # (1)
        
        else:
            # 取sentence
            seq, is_next_label  = [int(i) for i in self.get_corpus_line(item).split()], 0  
            # 随机掩膜
            tokens_random, label = self.random_word(seq) # # 改
            # 加上[CLS]  与 [SEP]
            tokens = [self.vocab.cls_token_id] + tokens_random + [self.vocab.sep_token_id]
            label =  [self.vocab.pad_id] + label + [self.vocab.pad_id]
            seq_len = self.seq_len +2 # 加上[CLS]  与 [SEP]
            # 截断
            bert_input = tokens[:seq_len]
            bert_label = label[:seq_len]
            segment_label = [0 for _ in range(len(tokens))][:seq_len]
            # 补齐
            padding = [self.vocab.pad_id for _ in range(seq_len - len(bert_input))]
            bert_input.extend(padding), bert_label.extend(padding), segment_label.extend(padding)
            
            output = {"bert_input": bert_input,         # (len+2)
                    "bert_label": bert_label,           # (len+2)
                    "segment_label": segment_label,     # (len+2)
                    "is_next": is_next_label}           # (1)
        
        return {key: torch.tensor(value) for key, value in output.items()}            


