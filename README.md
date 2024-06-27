Welcome to our project MCMCNet.
In this project, the GCLM module is located in the models/Contrastive_Loss directory, the ARAM module is located in the models/AFF_Attn directory, and the RSPH structure is implemented in models/DLinkNet5 directory.

Run Command: CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train.py --dataset='CHN6' --label-ratio=0.2
