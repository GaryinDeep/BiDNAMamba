#!/bin/bash
python 00_create_corpus.py\
  --data ./data/rawdata/hg19/GCF_000001405.25_GRCh37.p13_genomic.fna\
  --ngram 3\
  --seq_size 1000\
  --chunk_size 10000\
  --seq_stride 500\
  --ngram_stride 1\
  --hg_name hg19 \
  --output_path ./data/corpus

