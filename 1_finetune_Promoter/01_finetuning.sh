# bi
CUDA_VISIBLE_DEVICES=0,1,2 python -m torch.distributed.launch --nnodes 1 --nproc_per_node=3 02_finetuningbi.py \
--task finetune \
--distributed-mode DDP \
--ngram  3 \
--seq-size 600 \
--lr 7e-5 \
--batch-size 256 \
--epochs 15 \
--verbose-steps 100 \
--save-steps-interval 1 \
--pretrain-seq-size  1000 