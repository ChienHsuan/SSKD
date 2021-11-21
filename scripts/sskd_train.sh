#!/bin/sh
SOURCE="dukemtmc"
TARGET="market1501"         # market1501  dukemtmc
ARCH="msanet_pos"


CUDA_VISIBLE_DEVICES=3 \
python examples/sskd_train.py -ds ${SOURCE} -dt ${TARGET} -a ${ARCH} \
	--num-instances 4 --lr 0.00035 --iters 400 -b 64 --epochs 40 \
	--weight 0.5 --weight_ms 3 --weight_tf 1.5 --dropout 0 --lambda-value 0 \
	--height 192 --width 96 --features 512 \
	--init-1 /home/lab314/HDD/Hsuan/models/peer_networks/dukemtmc/msanet_pos/1/model.pth.tar-100 \
	--init-2 /home/lab314/HDD/Hsuan/models/peer_networks/dukemtmc/msanet_pos/2/model.pth.tar-100 \
	--init-3 /home/lab314/HDD/Hsuan/models/peer_networks/dukemtmc/msanet_pos/3/model.pth.tar-100 \
	--data-dir /home/lab314/HDD/Dataset \
	--logs-dir /home/lab314/HDD/Hsuan/SSKD/log \
	--eval-step 1 \
	--print-freq 50
