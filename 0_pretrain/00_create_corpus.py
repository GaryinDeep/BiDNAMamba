import argparse
import os
import sys
from Bio.Seq import Seq

sys.path.append("../")
from lib.dataset.vocab import gene_vocab



def process_fasta_raw_text(
    fname,
    vocab = None,
    ngram: int = 3,
    seq_size: int = 1000,
    chunk_size: int = 10000,
    seq_stride: int = 500,
    ngram_stride: int = 1,
    filter_txt: str = '>NC_',
    skip_n: bool = True,
    flip_strand: bool = False,
    output_file: str = './',
):

    # vocabulary
    word_dict = vocab.word_dict_alphabet
    str_len = len(str(vocab.last_valid_token_id))

    # Sequence length
    seq_size = max(seq_size, seq_size // ngram * ngram + (ngram - 1))
    print("seq_size: ", seq_size)
    seq_token_size  = seq_size // ngram
    print("seq_token_size: ", seq_token_size)

    index = 0
    chunks = ''
    set_atcg = set(list('ATCG'))
    with open(fname, mode='r', encoding='utf-8') as f:
        for line in f:
            line = line[:-1] # Delete '\n'
            line = line.upper()

            # Check if it’s filter content
            if filter_txt is not None and line.startswith(filter_txt):
                chunks = '' # 新的染色体
                continue

            # Check if it’s unkonw content
            if skip_n is True:
                if line.find('N') > -1:
                    continue

            # Check if it’s not ‘ATCG’
            set_seq = set(list(line))
            is_atcg = True
            for atcg in set_seq:
                if atcg not in set_atcg:
                    is_atcg = False
            if is_atcg is False:
                continue

            # 分块处理与存储
            if len(chunks) < chunk_size: 
                chunks += line              
            else:
                seq_token_all = []
                for ii in range(0, chunk_size, seq_stride): # seq_stride = 100
                    if ii + seq_size <= chunk_size:
                        seq = chunks[ii:int(ii + seq_size)]
                        # Check if it’s not ‘ATCG’
                        set_seq = set(list(seq))
                        is_atcg = True
                        for atcg in set_seq:
                            if atcg not in set_atcg:
                                is_atcg = False
                        if is_atcg is False:
                            continue
                        # convert to token
                        for kk in range(0, ngram, ngram_stride): # n_set ngram_stride = 1
                            seq_token_ngram = []
                            for jj in range(kk, len(seq), ngram): # ngram
                                if jj + ngram <= len(seq):
                                    if word_dict is not None:
                                        seq_token_ngram.append(str(word_dict.get(seq[jj:jj + ngram], 0))) # converting
                            # check seq_len:
                            if len(seq_token_ngram) == seq_token_size:
                                seq_token_all.append(seq_token_ngram)                   

                        # Complementary strand 
                        if flip_strand: 
                            my_dna = Seq(seq)
                            seq_flip = str(my_dna.reverse_complement())
                            # convert to token
                            for kk in range(0, ngram, ngram_stride): # n_set ngram_stride = 1
                                seq_flip_token_ngram = []
                                for jj in range(kk, len(seq_flip), ngram): # ngram
                                    if jj + ngram <= len(seq_flip):
                                        if word_dict is not None:
                                            seq_flip_token_ngram.append(str(word_dict.get(seq_flip[jj:jj + ngram], 0)))
                                # check seq_len:
                                if len(seq_flip_token_ngram) == seq_token_size:
                                    seq_token_all.append(seq_flip_token_ngram)

                # output
                with open(output_file, mode='a', encoding='utf-8') as f: # 追加模式
                    for seq_token in seq_token_all:
                        output = " ".join(seq_token)
                        pad = vocab.corpus_pad * ((str_len + 1)*seq_token_size - len(output) -1 ) # padding  +1=" " -1="\n"
                        string =  output + pad + "\n"
                        f.write(string)
                
                # Update to the remaining sequence
                chunks = chunks[ii:]
                chunks += line 

            # count
            index += 1
            if index % 100000 == 0:
                print("index ", index, "line_len ", len(line))




if __name__ == '__main__':

    _argparser = argparse.ArgumentParser(
        description='A data preprocessing of the Mamba language model in Genomics',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    _argparser.add_argument(
        '--data', type=str, required=True, metavar='PATH',
        help='A path of hg19/38 file.')
    _argparser.add_argument(
        '--ngram', type=int, default=3, metavar='INTEGER',
        help='NGram')
    _argparser.add_argument(
        '--seq_size', type=int, default=1000, metavar='INTEGER',
        help='Sequence size')
    _argparser.add_argument(
        '--chunk_size', type=int, default=10000, metavar='INTEGER',
        help='chunk size')
    _argparser.add_argument(
        '--seq_stride', type=int, default=500, metavar='INTEGER',
        help='Sequence stride size')
    _argparser.add_argument(
        '--ngram_stride', type=int, default=1, metavar='INTEGER',
        help='Ngram Stride size')
    _argparser.add_argument(
        '--hg_name', type=str, default='hg19', metavar='NAME',
        help='HG Name')
    _argparser.add_argument(
        '--output_path', type=str, default='./data/corpus', metavar='PATH',
        help='output path')

    _args = _argparser.parse_args()

    data_path = _args.data
    ngram = _args.ngram
    seq_size = _args.seq_size
    chunk_size = _args.chunk_size
    seq_stride = _args.seq_stride
    ngram_stride = _args.ngram_stride
    hg_name = _args.hg_name
    output_path = _args.output_path

    # 获取tokenizer
    vocab = gene_vocab(n_gram = ngram)
    word_dict_alphabet = vocab.word_dict_alphabet

    # 确认路径是否存在
    if os.path.exists(output_path) is False:
        os.makedirs(output_path)  
    # 重置记事本
    output_file_name = "{}_train_{}_gram_seqsize_{}.txt".format(hg_name, ngram, seq_size)
    f = open(os.path.join(output_path, output_file_name), "w")
    f.close()

    # 开始处理 
    process_fasta_raw_text(
        fname = data_path,
        vocab = vocab,
        ngram = ngram,
        seq_size = seq_size,
        chunk_size = chunk_size,
        seq_stride = seq_stride,
        ngram_stride = ngram_stride,
        filter_txt = '>NC_',
        skip_n = True,
        flip_strand = False,    
        output_file = os.path.join(output_path, output_file_name)
    )

    """
    只转换正链，不进行链翻转
    样本序列重叠为50%
    """


               
            