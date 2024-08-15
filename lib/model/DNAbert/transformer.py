import json
import os

import torch
import torch.nn as nn

from lib.model.DNAbert.config_bert import BertConfig
from lib.model.DNAbert.modeule.bert_embedding import BERTEmbedding
from lib.model.DNAbert.modeule.multi_head import MultiHeadedAttention
from lib.model.DNAbert.modeule.feed_forward import PositionwiseFeedForward, SublayerConnection
from lib.model.DNAbert.modeule.lm_head import NextSentencePrediction, MaskedLanguageModel
from lib.model.DNAbert.utils.hf import load_config_hf, load_state_dict_hf

class TransformerBlock(nn.Module):
    """
    Bidirectional Encoder = Transformer (self-attention)
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    """

    def __init__(self, hidden, attn_heads, feed_forward_hidden, dropout):
        """
        :param hidden: hidden size of transformer
        :param attn_heads: head sizes of multi-head attention
        :param feed_forward_hidden: feed_forward_hidden, usually 4*hidden_size
        :param dropout: dropout rate
        """

        super().__init__()
        self.attention = MultiHeadedAttention(h=attn_heads, d_model=hidden)
        self.feed_forward = PositionwiseFeedForward(d_model=hidden, d_ff=feed_forward_hidden, dropout=dropout)
        self.input_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.output_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask):
        x = self.input_sublayer(x, lambda _x: self.attention.forward(_x, _x, _x, mask=mask))
        x = self.output_sublayer(x, self.feed_forward)
        return self.dropout(x)



class BERT(nn.Module):
    """
    BERT model : Bidirectional Encoder Representations from Transformers.
    """

    def __init__(self, vocab_size, hidden=768, n_layers=12, attn_heads=12, dropout=0.1):
        """
        :param vocab_size: vocab_size of total words
        :param hidden: BERT model hidden size
        :param n_layers: numbers of Transformer blocks(layers)
        :param attn_heads: number of attention heads
        :param dropout: dropout rate
        """

        super().__init__()
        self.hidden = hidden
        self.n_layers = n_layers
        self.attn_heads = attn_heads

        # paper noted they used 4*hidden_size for ff_network_hidden_size
        self.feed_forward_hidden = hidden * 4

        # embedding for BERT, sum of positional, segment, token embeddings 
        self.embedding = BERTEmbedding(vocab_size=vocab_size, embed_size=hidden)# 原始bert为可学习的embedding层,非三角函数

        # multi-layers transformer blocks, deep network
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(hidden, attn_heads, hidden * 4, dropout) for _ in range(n_layers)])

    def forward(self, x, segment_info):
        # attention masking for padded token
        # torch.ByteTensor([batch_size, 1, seq_len, seq_len)
        mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1) # 掩膜pad的部分, 此处修改可实现gpt模式

        # embedding the indexed sequence to sequence of vectors
        x = self.embedding(x, segment_info)

        # running over multiple transformer blocks
        for transformer in self.transformer_blocks:
            x = transformer.forward(x, mask)

        return x



class BERTLM(nn.Module):

    def __init__(self, config: BertConfig):
        self.config = config
        vocab_size = config.vocab_size
        hidden = config.d_model
        n_layers = config.n_layer
        attn_heads = config.attn_heads
        dropout = config.dropout

        super().__init__()
        self.bert = BERT(vocab_size = vocab_size, hidden=hidden, n_layers=n_layers, attn_heads=attn_heads, dropout=dropout)
        # self.next_sentence = NextSentencePrediction(hidden = hidden)
        self.mask_lm = MaskedLanguageModel(hidden = hidden, vocab_size = vocab_size)

    def forward(self, x, segment_label, num_last_tokens=0): 
        x = self.bert(x, segment_label)    # (batch_size, seq_len, hidden_size)
        if num_last_tokens > 0:
            return self.mask_lm(x[:, -num_last_tokens:])  # (batch_size, 2) , (batch_size, num_last_tokens, vocab_size)
        return self.mask_lm(x)
        # return self.next_sentence(x), self.mask_lm(x) # (batch_size, 2) , (batch_size, seq_len, vocab_size)
    
    @classmethod # cls 表示这个类本身。
    def from_pretrained(cls, pretrained_model_name): 
        config_data = load_config_hf(pretrained_model_name)
        config = BertConfig(**config_data)
        model = cls(config)
        model.load_state_dict(load_state_dict_hf(pretrained_model_name))
        print("loading successful:", pretrained_model_name)
        return model

    def save_pretrained(self, save_directory):
        """
        Minimal implementation of save_pretrained for MambaLMHeadModel.
        Save the model and its configuration file to a directory.
        """
        # Ensure save_directory exists
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        # Save the model's state_dict
        model_path = os.path.join(save_directory, 'pytorch_model.bin')
        torch.save(self.state_dict(), model_path)

        # Save the configuration of the model
        config_path = os.path.join(save_directory, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(self.config.__dict__, f)




"""
用于自监督微调
"""
class BERTLM_sft(BERTLM):

    @classmethod # cls 表示这个类本身。
    def from_pretrained(cls, pretrained_model_name, n_output):  # 不定长参数,参数存储为dic
        """
        功能:先读入再更换head layer
        """
        config_data = load_config_hf(pretrained_model_name)
        config = BertConfig(**config_data)
        model = cls(config)
        model.load_state_dict(load_state_dict_hf(pretrained_model_name))
        print("loading successful:", pretrained_model_name)
        model.replace_head(n_output) # n分类 n=2=二分类 
        return model

    @classmethod # cls 表示这个类本身。
    def from_funetuned(cls, pretrained_model_name, n_output):  # 不定长参数,参数存储为dic
        """
        功能:先更换head layer再读入
        """
        config_data = load_config_hf(pretrained_model_name)
        config = BertConfig(**config_data)
        model = cls(config)
        model.replace_head(n_output) # n分类 n=2=二分类 
        model.load_state_dict(load_state_dict_hf(pretrained_model_name))
        print("loading successful:", pretrained_model_name)
        return model  

    def replace_head(self, n_output): 
        """
        功能:替换head layer
        """       
        self.mask_lm = MaskedLanguageModel(hidden = self.config.d_model, vocab_size = n_output)

