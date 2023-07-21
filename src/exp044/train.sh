python train_stage1.py --seed 5023 --model_name resnetrs50 --drop_path_rate 0.2 \
    --lr 1e-3 --batch_size 16 --image_size 256 \
    --fold 0 --gpus 4 --epochs 30 --logdir resnetrs50_cls_stage1_small --num_workers 6

python train_stage2.py --seed 5024 --model_name resnetrs50 --drop_path_rate 0.4 \
    --lr 1e-3 --batch_size 16 --image_size 256 \
    --fold 0 --gpus 4 --epochs 60 --logdir resnetrs50_unet_stage2_small_ep60 --num_workers 6 

python train_stage1_seg.py --seed 5025 --model_name resnetrs50 --drop_path_rate 0.4 \
    --lr 1e-3 --batch_size 16 --image_size 256 \
    --fold 0 --gpus 4 --epochs 30 --logdir resnetrs50_unet_stage1_small_ep30 --num_workers 6 

python eval_fp16.py --seed 2023 --batch_size 4  --fold 0 \
    --logdir_cls resnetrs50_cls_stage1_small --logdir_seg resnetrs50_unet_stage2_small_ep60 --num_workers 6

python eval_fp16_v2.py --seed 2023 --batch_size 4  --fold 0 \
    --logdir_cls resnetrs50_cls_stage1_small --logdir_seg resnetrs50_unet_stage2_small_ep60 --logdir_stage1_seg resnetrs50_unet_stage1_small_ep30 --num_workers 6

python eval_fp16_v2_sub.py --seed 2023 --batch_size 4  --fold 0 \
    --logdir_cls resnetrs50_cls_stage1_small --logdir_seg resnetrs50_unet_stage2_small_ep60 --logdir_stage1_seg resnetrs50_unet_stage1_small_ep30 --num_workers 6


python train_stage1.py --seed 2023 --model_name resnetrs101 --drop_path_rate 0.2 \
    --lr 1e-3 --batch_size 8 --image_size 384 \
    --fold 0 --gpus 4 --epochs 30 --logdir resnetrs101_cls_stage1 --num_workers 6

python train_stage2.py --seed 2023 --model_name resnetrs101 --drop_path_rate 0.4 \
    --lr 1e-3 --batch_size 8 --image_size 384 \
    --fold 0 --gpus 4 --epochs 60 --logdir resnetrs101_unet_stage2_ep60 --num_workers 6 

python train_stage1_seg.py --seed 2023 --model_name resnetrs101 --drop_path_rate 0.4 \
    --lr 1e-3 --batch_size 8 --image_size 384 \
    --fold 0 --gpus 4 --epochs 30 --logdir resnetrs101_unet_stage1_ep30 --num_workers 6 

python eval_fp16.py --seed 2023 --batch_size 16  --fold 0 \
    --logdir_cls resnetrs101_cls_stage1 --logdir_seg resnetrs101_unet_stage2_ep60 --num_workers 6

python eval_fp16_v2.py --seed 2023 --batch_size 16  --fold 0 \
    --logdir_cls resnetrs101_cls_stage1 --logdir_seg resnetrs101_unet_stage2_ep60 --logdir_stage1_seg resnetrs101_unet_stage1_ep30 --num_workers 6


python train_stage1.py --seed 6023 --model_name resnetrs50 --drop_path_rate 0.2 \
    --lr 1e-3 --batch_size 8 --image_size 512 \
    --fold 0 --gpus 4 --epochs 30 --logdir resnetrs50_cls_stage1 --num_workers 6

python train_stage2.py --seed 6024 --model_name resnetrs50 --drop_path_rate 0.4 \
    --lr 1e-3 --batch_size 8 --image_size 512 \
    --fold 0 --gpus 4 --epochs 60 --logdir resnetrs50_unet_stage2_ep60 --num_workers 6 

python train_stage1_seg.py --seed 6025 --model_name resnetrs50 --drop_path_rate 0.4 \
    --lr 1e-3 --batch_size 8 --image_size 512 \
    --fold 0 --gpus 4 --epochs 30 --logdir resnetrs50_unet_stage1_ep30 --num_workers 6 

python eval_fp16.py --seed 2023 --batch_size 4  --fold 0 \
    --logdir_cls resnetrs50_cls_stage1 --logdir_seg resnetrs50_unet_stage2_ep60 --num_workers 6

python eval_fp16_v2.py --seed 2023 --batch_size 4  --fold 0 \
    --logdir_cls resnetrs50_cls_stage1 --logdir_seg resnetrs50_unet_stage2_ep60 --logdir_stage1_seg resnetrs50_unet_stage1_ep30 --num_workers 6

python eval_fp16_v2_sub.py --seed 2023 --batch_size 4  --fold 0 \
    --logdir_cls resnetrs50_cls_stage1 --logdir_seg resnetrs50_unet_stage2_ep60 --logdir_stage1_seg resnetrs50_unet_stage1_ep30 --num_workers 6


python train_stage1.py --seed 2023 --model_name swinv2_base_window16_256.ms_in1k \
    --drop_path_rate 0.2 --backbone_lr 2e-4 --lr 1e-3 --batch_size 16 --image_size 256 \
    --fold 0 --gpus 4 --epochs 30 --logdir swinv2_base_window16_cls_stage1 --num_workers 6

python train_stage2.py --seed 2023 --model_name swinv2_base_window16_256.ms_in1k \
     --drop_path_rate 0.4 --backbone_lr 2e-4 --lr 1e-3 --batch_size 16 --image_size 256 \
    --fold 0 --gpus 4 --epochs 60 --logdir swinv2_base_window16_unet_stage2_ep60 --num_workers 6 

python train_stage1_seg.py --seed 2023 --model_name swinv2_base_window16_256.ms_in1k \
     --drop_path_rate 0.4 --backbone_lr 2e-4 --lr 1e-3 --batch_size 16 --image_size 256 \
    --fold 0 --gpus 4 --epochs 30 --logdir swinv2_base_window16_unet_stage1_ep30 --num_workers 6 

python eval_fp16.py --seed 2023 --batch_size 4  --fold 0 \
    --logdir_cls swinv2_base_window16_cls_stage1 --logdir_seg swinv2_base_window16_unet_stage2_ep60 --num_workers 6

python eval_fp16_v2.py --seed 2023 --batch_size 4  --fold 0 --logdir_cls swinv2_base_window16_cls_stage1 \
    --logdir_seg swinv2_base_window16_unet_stage2_ep60 --logdir_stage1_seg swinv2_base_window16_unet_stage1_ep30 --num_workers 6


python train_stage1.py --seed 2023 --model_name swin_base_patch4_window12_384.ms_in1k \
    --drop_path_rate 0.2 --backbone_lr 2e-4 --lr 1e-3 --batch_size 8 --image_size 384 \
    --fold 0 --gpus 4 --epochs 30 --logdir swin_base_patch4_window12_cls_stage1 --num_workers 6

python train_stage2.py --seed 2023 --model_name swin_base_patch4_window12_384.ms_in1k \
     --drop_path_rate 0.4 --backbone_lr 2e-4 --lr 1e-3 --batch_size 8 --image_size 384 \
    --fold 0 --gpus 4 --epochs 60 --logdir swin_base_patch4_window12_unet_stage2_ep60 --num_workers 6 

python train_stage1_seg.py --seed 2023 --model_name swin_base_patch4_window12_384.ms_in1k \
     --drop_path_rate 0.4 --backbone_lr 2e-4 --lr 1e-3 --batch_size 8 --image_size 384 \
    --fold 0 --gpus 4 --epochs 30 --logdir swin_base_patch4_window12_unet_stage1_ep30 --num_workers 6 

python eval_fp16.py --seed 2023 --batch_size 4  --fold 0 \
    --logdir_cls swin_base_patch4_window12_cls_stage1 --logdir_seg swin_base_patch4_window12_unet_stage2_ep60 --num_workers 6

python eval_fp16_v2.py --seed 2023 --batch_size 4  --fold 0 --logdir_cls swin_base_patch4_window12_cls_stage1 \
    --logdir_seg swin_base_patch4_window12_unet_stage2_ep60 --logdir_stage1_seg swin_base_patch4_window12_unet_stage1_ep30 --num_workers 6

python eval_fp16_v2_sub.py --seed 2023 --batch_size 16  --fold 0 --logdir_cls swin_base_patch4_window12_cls_stage1 \
    --logdir_seg swin_base_patch4_window12_unet_stage2_ep60 --logdir_stage1_seg swin_base_patch4_window12_unet_stage1_ep30 --num_workers 6


python train_stage1.py --seed 2023 --model_name convnext_base.fb_in1k --drop_path_rate 0.2 \
    --lr 1e-3 --batch_size 8 --image_size 384 \
    --fold 0 --gpus 4 --epochs 30 --logdir convnext_base_cls_stage1 --num_workers 6

python train_stage2.py --seed 2023 --model_name convnext_base.fb_in1k --drop_path_rate 0.4 \
    --lr 1e-3 --batch_size 8 --image_size 384 \
    --fold 0 --gpus 4 --epochs 60 --logdir convnext_base_unet_stage2_ep60 --num_workers 6 

python train_stage1_seg.py --seed 2023 --model_name convnext_base.fb_in1k --drop_path_rate 0.4 \
    --lr 1e-3 --batch_size 8 --image_size 384 \
    --fold 0 --gpus 4 --epochs 30 --logdir convnext_base_unet_stage1_ep30 --num_workers 6 

python eval_fp16.py --seed 2023 --batch_size 16  --fold 0 \
    --logdir_cls convnext_base_cls_stage1 --logdir_seg convnext_base_unet_stage2_ep60 --num_workers 6

python eval_fp16_v2.py --seed 2023 --batch_size 16  --fold 0 \
    --logdir_cls convnext_base_cls_stage1 --logdir_seg convnext_base_unet_stage2_ep60 --logdir_stage1_seg convnext_base_unet_stage1_ep30 --num_workers 6

python eval_fp16_v2_sub.py --seed 2023 --batch_size 16  --fold 0 \
    --logdir_cls convnext_base_cls_stage1 --logdir_seg convnext_base_unet_stage2_ep60 --logdir_stage1_seg convnext_base_unet_stage1_ep30 --num_workers 6
