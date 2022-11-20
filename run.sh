#!/bin/bash

# * laptop

# * TCL
CUDA_VISIBLE_DEVICES=0 python ./train.py --model_name tcl --dataset laptop --seed 1000 --num_epoch 40 --vocab_dir ./dataset/Laptops_corenlp --cuda 0 --losstype orthogonalloss --alpha 0.5 --beta 0.7 --gamma 0.8 --reshape --batch_size 16
# * TCL with Bert
# CUDA_VISIBLE_DEVICES=0 python ./train.py --model_name dtclbert --dataset laptop --seed 1000 --bert_lr 2e-5 --num_epoch 15 --hidden_dim 768 --max_length 100 --cuda 0 --losstype orthogonalloss --alpha 0.2 --beta 0.3 --gamma 0.7 --reshape --batch_size 16


# * restaurant

# * Triplet CL
# CUDA_VISIBLE_DEVICES=0 python ./train.py --model_name tcl --dataset restaurant --seed 1000 --num_epoch 50 --vocab_dir ./dataset/Restaurants_corenlp --cuda 0 --losstype orthogonalloss --alpha 0.1 --beta 0.5 --gamma 0.5 --reshape --batch_size 64
# * Triplet CL with Bert
# CUDA_VISIBLE_DEVICES=0 python ./train.py --model_name tclbert --dataset restaurant --seed 1000 --bert_lr 2e-5 --num_epoch 15 --hidden_dim 768 --max_length 100 --cuda 1 --losstype orthogonalloss --alpha 0.2 --beta 0.5 --gamma 0.4 --reshape --batch_size 16


# * twitter

# * Triplet CL
# CUDA_VISIBLE_DEVICES=0 python ./train.py --model_name tcl --dataset twitter --seed 1000 --num_epoch 50 --vocab_dir ./dataset/Tweets_corenlp --cuda 0 --losstype orthogonalloss --alpha 0.2 --beta 0.2 --gamma 0.7 --reshape --batch_size 16
# * Triplet CL with Bert
# CUDA_VISIBLE_DEVICES=0 python ./train.py --model_name tclbert --dataset twitter --seed 1000 --bert_lr 2e-5 --num_epoch 15 --hidden_dim 768 --max_length 100 --cuda 0 --losstype orthogonalloss --alpha 0.3 --beta 0.5 --gamma 0.7 --reshape --batch_size 32
