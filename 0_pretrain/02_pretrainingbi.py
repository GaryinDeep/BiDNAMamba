import os
import argparse

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from torch.utils.tensorboard import SummaryWriter  

import sys
sys.path.append('../')
from lib.dataset.vocab import gene_vocab
from lib.dataset.dataset import MambaDataset
from lib.model.DNAmamba.biDNAmamba import MambaConfig, MambaLMHeadModel
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
    args=parser.parse_args()


    task = args.task
    distributed_mode = args.distributed_mode # 该参数控制使用DDP(DistributedDataParallel)或者DataParallel
    local_rank=args.local_rank # DDP所需参数,由DDP自动传入
    if distributed_mode == "DDP": # DDP
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group('nccl', init_method='env://')
        os.environ['RANK'] = str(0) # 设置环境 必须放在这三条语句最后，否则死锁

    ngram=args.ngram
    seq_size=args.seq_size
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

    train_corpus_path = "./data/corpus/hg19_train_{0}_gram_seqsize_{1}.txt".format(ngram, seq_size)
    test_corpus_path = None
    save_directory = "./data/model/bi_{}_gram_{}_seq_size_{}_layer_{}_dim".format(ngram, seq_size, n_layer, d_model)  # replace
    logging_filename = "train_log.txt"

    # 获取tokenizer
    print("\nLoading Vocab")
    vocab = gene_vocab(n_gram = ngram)
    word_dict_alphabet = vocab.word_dict_alphabet
    print("Vocab Size: ", vocab.vocab_size)

    # 获取数据集
    print("\nLoading Train Dataset", train_corpus_path)
    train_dataset = MambaDataset(train_corpus_path, vocab, seq_len, corpus_lines = corpus_lines, task = task)
    print("Loading Test Dataset", test_corpus_path)
    test_dataset = MambaDataset(test_corpus_path, vocab, seq_len, corpus_lines = corpus_lines, task = task) \
        if test_corpus_path is not None else None

    print("\nCreating Dataloader")
    print("Batch size: ", batch_size)
    if distributed_mode == "DDP": # DDP
        train_sampler = DistributedSampler(train_dataset,shuffle=True) # DataLoader中的shuffle应该设置为False，因为打乱的任务交给了sampler
        train_data_loader = DataLoader(dataset=train_dataset,batch_size= batch_size, num_workers=num_workers, shuffle=False, drop_last=True,sampler=train_sampler)
        test_data_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, drop_last=False) \
            if test_dataset is not None else None
    else:  # DataParallel
        train_data_loader = DataLoader(dataset=train_dataset,batch_size= batch_size, num_workers=num_workers, shuffle=True, drop_last=True)
        test_data_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, drop_last=False) \
            if test_dataset is not None else None

    print("\nBuilding mamba model")
    config = MambaConfig(d_model=d_model, n_layer=n_layer, vocab_size=vocab.vocab_size, ssm_cfg={},) # replace
    mamba = MambaLMHeadModel(config, dtype=torch.bfloat16, device="cuda") # replace

    print("\nCreating mamba Trainer")  # 在此加载预训练模型
    train_writer = SummaryWriter(os.path.join(save_directory, "steps_model", "logs/train")) # 写入、可视化操作
    val_writer = SummaryWriter(os.path.join(save_directory, "steps_model", "logs/val")) if test_data_loader is not None else None
    trainer = MambaTrainer(mamba = mamba, vocab = vocab,
                    train_dataloader=train_data_loader, test_dataloader =test_data_loader,
                    lr = lr, betas=betas, weight_decay = weight_decay, 
                    distributed_mode = distributed_mode, with_cuda = with_cuda, local_rank = local_rank,
                    verbose_steps = verbose_steps, save_steps = train_dataset.corpus_lines//batch_size//save_steps_interval,
                    save_directory = save_directory, logging_filename = logging_filename, 
                    train_writer=train_writer, val_writer =val_writer,
                    save_best_only = False,
                    ) # replace

    print("\nTraining Start")  # 删除多余的model文件夹
    for epoch in range(epochs):
        trainer.train(epoch)
        if test_data_loader is not None:
            trainer.test(epoch)
        
