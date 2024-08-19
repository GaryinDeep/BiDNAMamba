#### bi
CUDA_VISIBLE_DEVICES=0,1,2 python -m torch.distributed.launch --nnodes 1 --nproc_per_node=3 02_pretrainingbi.py \
--task pretrain \
--distributed-mode DDP \
--ngram  3 \
--seq-size 1000 \
--d-model 768 \
--n-layer 24 \
--lr 1e-4 \
--batch-size 256 \
--epochs 20 \
--verbose-steps 100 \
--save-steps-interval 1