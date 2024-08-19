# BiDNAMamba

This repository contains code and pre-trained weights for BiDNAMamba.


## Usage

### Requirement

```
mamba-ssm        1.2.0
causal-conv1d    1.2.0 
torch            2.1.1+cu118
```

mamba setup: see: https://blog.csdn.net/weixin_46413311/article/details/137872145?spm=1001.2014.3001.5502


## Pre-training model

### 1. Data preprocessing

From https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/001/405/GCF_000001405.25_GRCh37.p13/
Download GCF_000001405.25_GRCh37.p13_genomic.fna.gz, And unzip, for example, unzip to /data/rawdata/hg19/GCF_000001405.25_GRCh37.p13_genomic.fna

```
bash /0_pretrain/00_create_corpus.sh
```

### 2. Pretraining

```
bash /0_pretrain/01_pretraining.sh
```

## Fine-tuning model

```
bash /1_finetune_Promoter/01_finetuning.sh
```

## Interpretability analysis

```
python /5_fig/shap_mamba_.py
```