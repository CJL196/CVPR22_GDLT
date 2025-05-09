#! /bin/bash
# FineFS

# sp, pcs
python main.py --video-path FineFS/process/SWIN_Feature64 --train-label-path FineFS/data/annotation --model-name sp-pcs --action-type Ball --lr 1e-2 --epoch 250 --n_decoder 2 --n_query 4 --alpha 1.0 --margin 1.0 --lr-decay cos --decay-rate 0.01 --dropout 0.3 --dataset finefs --split sp --type pcs

# fs, pcs
python main.py --video-path FineFS/process/SWIN_Feature64 --train-label-path FineFS/data/annotation --model-name fs-pcs --action-type Ball --lr 1e-2 --epoch 250 --n_decoder 2 --n_query 4 --alpha 1.0 --margin 1.0 --lr-decay cos --decay-rate 0.01 --dropout 0.3 --dataset finefs --split fs --type pcs

# sp, tes
python main.py --video-path FineFS/process/SWIN_Feature64 --train-label-path FineFS/data/annotation --model-name sp-tes --action-type Ball --lr 1e-2 --epoch 250 --n_decoder 2 --n_query 4 --alpha 1.0 --margin 1.0 --lr-decay cos --decay-rate 0.01 --dropout 0.3 --dataset finefs --split sp --type tes

# fs, tes
python main.py --video-path FineFS/process/SWIN_Feature64 --train-label-path FineFS/data/annotation --model-name fs-tes --action-type Ball --lr 1e-2 --epoch 250 --n_decoder 2 --n_query 4 --alpha 1.0 --margin 1.0 --lr-decay cos --decay-rate 0.01 --dropout 0.3 --dataset finefs --split fs --type tes



