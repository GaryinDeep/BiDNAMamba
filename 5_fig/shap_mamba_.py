import os
import re
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import shap

import torch
from torch.utils.data import DataLoader

import sys
sys.path.append('../')
from lib.dataset.vocab import gene_vocab
from lib.dataset.dataset import MambaDataset
from lib.model.DNAmamba.biDNAmamba import MambaConfig, MambaLMHeadModel_sft

def softmax(outputs):
    maxes = np.max(outputs, axis=-1, keepdims=True)
    shifted_exp = np.exp(outputs - maxes)
    return shifted_exp / shifted_exp.sum(axis=-1, keepdims=True)


def f(x):  
    input_ids = torch.tensor(x).to("cuda")
    logits = mamba.forward(input_ids,num_last_tokens=1).logits[:,0].detach().to(torch.float).cpu().numpy() # 注意维度
    scores = softmax(logits)[:,1]
    # val = sp.special.logit(scores[:,1])
    return scores


def custom_tokenizer(s, return_offsets_mapping=True):
    """
    Custom tokenizers conform to a subset of the transformers API.
    this will create a basic whitespace tokenizer 匹配非字母数字及下划线
    """
    pos = 0
    offset_ranges = []
    input_ids = []
    for m in re.finditer(r"\W", s):
        start, end = m.span(0)
        offset_ranges.append((pos, start))
        input_ids.append(int(s[pos:start]))
        pos = end
    if pos != len(s):
        offset_ranges.append((pos, len(s)))
        input_ids.append(int(s[pos:]))
    out = {}
    out["input_ids"] = input_ids
    if return_offsets_mapping:
        out["offset_mapping"] = offset_ranges
    return out


if __name__ == '__main__':

    k = 0
    ngram = 3
    seq_size = 600
    seq_len =  seq_size // ngram
    n_layer = 24
    d_model = 768
    task_name = 'epdnew_BOTH'
    dataset_flag = "test"

    data_path = "./data/corpus"
    data_file_name = "{}_{}_gram_{}_fold_{}.txt".format(task_name, ngram, k, dataset_flag)
    data_label_file_name = "{}_{}_gram_label_{}_fold_{}.txt".format(task_name,ngram, k, dataset_flag)
    test_corpus_path = os.path.join(data_path,"{}_gram".format(ngram), task_name, data_file_name)
    test_label_path = os.path.join(data_path,"{}_gram".format(ngram), task_name, data_label_file_name)

    print("\nLoading Vocab")
    vocab = gene_vocab(n_gram = ngram)
    word_dict_alphabet = vocab.word_dict_alphabet

    print("\nLoading Test Dataset\n", test_corpus_path, "\n",test_label_path)
    test_dataset = MambaDataset(test_corpus_path, vocab, seq_len, corpus_lines = None, task = "finetune", labels_path = test_label_path, labels_lines=None)       

    print("\nBuilding mamba model")

    # not train
    # config = MambaConfig(d_model=d_model, n_layer=n_layer, vocab_size=vocab.vocab_size, ssm_cfg={},)
    # mamba = MambaLMHeadModel_sft(config, dtype=torch.bfloat16, device="cuda")
    # mamba.replace_head(n_output=2,**{"device": "cuda", "dtype": torch.bfloat16})
    
    # pretrain not finetune
    # save_directory = "./data/model/shap/bi_{}_gram_1000_seq_size_{}_layer_{}_dim".format(ngram, n_layer, d_model) 
    # mamba = MambaLMHeadModel_sft.from_pretrained(save_directory, dtype=torch.bfloat16, device="cuda", n_output=2) # 加载模型 n=2=二分类 

    # finetune
    save_directory = "./data/model/shap/bi_{}_gram_{}_seq_size_{}_layer_{}_dim_{}_fold_{}".format(ngram, seq_size, n_layer, d_model, task_name, k) 
    mamba = MambaLMHeadModel_sft.from_funetuned(os.path.join(save_directory, "epoch_model","best"), dtype=torch.bfloat16, device="cuda", n_output=2)

    # finetune only_forward
    # from lib.model.DNAmamba.DNAmamba import MambaLMHeadModel_sft
    # save_directory = "./data/model/shap/{}_gram_{}_seq_size_{}_layer_{}_dim_{}_fold_{}".format(ngram, seq_size, n_layer, d_model, task_name, k) 
    # mamba = MambaLMHeadModel_sft.from_funetuned(os.path.join(save_directory, "epoch_model","best"), dtype=torch.bfloat16, device="cuda", n_output=2)
 
    # data = ['26 130 27 21 26 34',]
    dic = test_dataset.__getitem__(0)
    data = [" ".join(map(str, dic["mamba_input"].tolist())),]
    label = dic["mamba_label"].tolist()

    masker = shap.maskers.Text(custom_tokenizer, mask_token = 0, output_type='token_ids') # 使用特定id进行掩码
    explainer = shap.Explainer(f, masker, algorithm='partition')

    shap_values = explainer(data)

    word_dict_alphabet_reverse = {v: k for k, v in word_dict_alphabet.items()}
    word_dict_alphabet_reverse[3] = "[Sep]"
    shap_values.data = ([word_dict_alphabet_reverse[i]+" " for i in dic["mamba_input"].tolist()],)
    
    result = shap.plots.text(shap_values[0], display=False)
    # shap.plots.waterfall(shap_values[0],max_display=101)
    # shap.plots.beeswarm(shap_values[:,:,1].mean(0))
    # shap.plots.bar(shap_values[:,:,1].mean(0))
    # shap.plots.heatmap(shap_values[0])


    # file = open('./data/bimmaba_nottrain_shap.html','w')
    # file = open('./data/bimmaba_pretrain_shap.html','w')
    file = open('./data/bimmaba_shap.html','w')
    # file = open('./data/mmaba_shap.html','w')
    file.write(result)
    file.close()
