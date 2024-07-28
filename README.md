Run Command: CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train.py --dataset='CHN6' --label-ratio=0.2
