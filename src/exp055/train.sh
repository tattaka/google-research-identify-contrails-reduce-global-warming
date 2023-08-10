python train_stage1.py --seed 2024 --model_name resnetrs50 --drop_path_rate 0.2 \
    --lr 1e-3 --batch_size 8 --image_size 256 \
    --fold 0 --gpus 4 --epochs 30 --logdir resnetrs50_256_cls_stage1 --num_workers 6 --disable_compile

python train_stage2.py --seed 2024 --model_name resnetrs50 --drop_path_rate 0.4 \
    --lr 1e-3 --batch_size 8 --image_size 256 \
    --fold 0 --gpus 4 --epochs 60 --logdir resnetrs50_256_unet_stage2_ep60 --num_workers 6  --disable_compile

python train_stage1_seg.py --seed 2025 --model_name resnetrs50 --drop_path_rate 0.4 \
    --lr 1e-3 --batch_size 8 --image_size 256 \
    --fold 0 --gpus 4 --epochs 30 --logdir resnetrs50_256_unet_stage1_ep30 --num_workers 6  --disable_compile

python eval_fp16.py --batch_size 16  --fold 0 \
    --logdir_cls resnetrs50_cls_stage1 --logdir_seg resnetrs50_256_unet_stage2_ep60 --num_workers 6

python eval_fp16_v2.py --batch_size 4  --fold 0 \
    --logdir_cls resnetrs50_cls_stage1 --logdir_seg resnetrs50_256_unet_stage2_ep60 --logdir_stage1_seg resnetrs50_256_unet_stage1_ep30 --num_workers 6

python eval_fp16_v2_sub.py --batch_size 4  --fold 0 \
    --logdir_cls resnetrs50_cls_stage1 --logdir_seg resnetrs50_256_unet_stage2_ep60 --logdir_stage1_seg resnetrs50_256_unet_stage1_ep30 --num_workers 6


python train_stage1.py --seed 3023 --model_name resnetrs101 --drop_path_rate 0.2 \
    --lr 1e-3 --batch_size 4 --image_size 384 --seq_len 5 \
    --fold 0 --gpus 4 --epochs 30 --logdir resnetrs101_cls_stage1 --num_workers 6 --disable_compile

python train_stage2.py --seed 3024 --model_name resnetrs101 --drop_path_rate 0.4 \
    --lr 1e-3 --batch_size 4 --image_size 384 --seq_len 5  \
    --fold 0 --gpus 4 --epochs 60 --logdir resnetrs101_384_unet_stage2_ep60 --num_workers 6 --disable_compile

python train_stage1_seg.py --seed 3026 --model_name resnetrs101 --drop_path_rate 0.4 \
    --lr 1e-3 --batch_size 4 --image_size 384 --seq_len 5  \
    --fold 0 --gpus 4 --epochs 30 --logdir resnetrs101_384_unet_stage1_ep30 --num_workers 6 --disable_compile

python eval_fp16.py --batch_size 16  --fold 0 \
    --logdir_cls resnetrs101_cls_stage1 --logdir_seg resnetrs101_384_unet_stage2_ep60 --num_workers 6 

python eval_fp16_v2.py --batch_size 16  --fold 0 \
    --logdir_cls resnetrs101_cls_stage1 --logdir_seg resnetrs101_384_unet_stage2_ep60 --logdir_stage1_seg resnetrs101_384_unet_stage1_ep30 --num_workers 6

python eval_fp16_v2_sub.py --batch_size 16  --fold 0 \
    --logdir_cls resnetrs101_cls_stage1 --logdir_seg resnetrs101_384_unet_stage2_ep60 --logdir_stage1_seg resnetrs101_384_unet_stage1_ep30 --num_workers 6


python train_stage1.py --seed  4023 --model_name swinv2_base_window16_256.ms_in1k \
    --drop_path_rate 0.2 --backbone_lr 2e-4 --lr 1e-3 --batch_size 2 --image_size 256 \
    --fold 0 --gpus 4 --epochs 30 --logdir swinv2_base_window16_cls_stage1 --num_workers 6 --disable_compile

python train_stage2.py --seed 4024 --model_name swinv2_base_window16_256.ms_in1k \
     --drop_path_rate 0.4 --backbone_lr 5e-4 --lr 1e-3 --batch_size 2 --image_size 256 \
    --fold 0 --gpus 4 --epochs 60 --logdir swinv2_base_window16_256_unet_stage2_ep60 --num_workers 6 --disable_compile

python train_stage1_seg.py --seed 4025 --model_name swinv2_base_window16_256.ms_in1k \
     --drop_path_rate 0.4 --backbone_lr 5e-4 --lr 1e-3 --batch_size 2 --image_size 256 \
    --fold 0 --gpus 4 --epochs 30 --logdir swinv2_base_window16_256_unet_stage1_ep30 --num_workers 6 --disable_compile

python eval_fp16.py --batch_size 4  --fold 0 \
    --logdir_cls swin_base_patch4_window12_cls_stage1 --logdir_seg swinv2_base_window16_256_unet_stage2_ep60 --num_workers 6

python eval_fp16_v2.py --batch_size 4  --fold 0 --logdir_cls swin_base_patch4_window12_cls_stage1 \
    --logdir_seg swinv2_base_window16_256_unet_stage2_ep60 --logdir_stage1_seg swinv2_base_window16_256_unet_stage1_ep30 --num_workers 6

python eval_fp16_v2_sub.py --batch_size 4  --fold 0 --logdir_cls swin_base_patch4_window12_cls_stage1 \
    --logdir_seg swinv2_base_window16_256_unet_stage2_ep60 --logdir_stage1_seg swinv2_base_window16_256_unet_stage1_ep30 --num_workers 6


python train_stage1.py --seed 5023 --model_name swin_base_patch4_window12_384.ms_in1k \
    --drop_path_rate 0.2 --backbone_lr 2e-4 --lr 1e-3 --batch_size 2 --image_size 384 --seq_len 5  \
    --fold 0 --gpus 4 --epochs 30 --logdir swin_base_patch4_window12_384_cls_stage1 --num_workers 6 --disable_compile

python train_stage2.py --seed 5024 --model_name swin_base_patch4_window12_384.ms_in1k \
     --drop_path_rate 0.4 --backbone_lr 5e-4 --lr 1e-3 --batch_size 2 --image_size 384 --seq_len 5  \
    --fold 0 --gpus 4 --epochs 60 --logdir swin_base_patch4_window12_384_unet_stage2_ep60 --num_workers 6 --disable_compile

python train_stage1_seg.py --seed 5025 --model_name swin_base_patch4_window12_384.ms_in1k \
     --drop_path_rate 0.4 --backbone_lr 5e-4 --lr 1e-3 --batch_size 2 --image_size 384 --seq_len 5  \
    --fold 0 --gpus 4 --epochs 30 --logdir swin_base_patch4_window12_384_unet_stage1_ep30 --num_workers 6 --disable_compile

python eval_fp16.py --batch_size 16  --fold 0 \
    --logdir_cls swin_base_patch4_window12_cls_stage1 --logdir_seg swin_base_patch4_window12_384_unet_stage2_ep60 --num_workers 6

python eval_fp16_v2.py --batch_size 16  --fold 0 --logdir_cls swin_base_patch4_window12_cls_stage1 \
    --logdir_seg swin_base_patch4_window12_384_unet_stage2_ep60 --logdir_stage1_seg swin_base_patch4_window12_384_unet_stage1_ep30 --num_workers 6

python eval_fp16_v2_sub.py --batch_size 16  --fold 0 --logdir_cls swin_base_patch4_window12_cls_stage1 \
    --logdir_seg swin_base_patch4_window12_384_unet_stage2_ep60 --logdir_stage1_seg swin_base_patch4_window12_384_unet_stage1_ep30 --num_workers 6


python train_stage1.py --seed 6023 --model_name convnext_base.fb_in1k --drop_path_rate 0.2 \
    --lr 1e-3 --batch_size 8 --image_size 256 \
    --fold 0 --gpus 4 --epochs 30 --logdir convnext_base_256_cls_stage1 --num_workers 6 --disable_compile

python train_stage2.py --seed 6024 --model_name convnext_base.fb_in1k --drop_path_rate 0.4 \
    --lr 1e-3 --batch_size 4 --image_size 256 --attention_type cbam \
    --fold 0 --gpus 4 --epochs 60 --logdir convnext_base_256_unet_cbam_stage2_ep60 --num_workers 6 --disable_compile

python train_stage1_seg.py --seed 6025 --model_name convnext_base.fb_in1k --drop_path_rate 0.4 \
    --lr 1e-3 --batch_size 4 --image_size 256 --attention_type cbam \
    --fold 0 --gpus 4 --epochs 30 --logdir convnext_base_256_unet_cbam_stage1_ep30 --num_workers 6 --disable_compile

python eval_fp16.py --batch_size 16  --fold 0 \
    --logdir_cls convnext_base_cls_stage1 --logdir_seg convnext_base_256_unet_cbam_stage2_ep60 --num_workers 6

python eval_fp16_v2.py --batch_size 16  --fold 0 \
    --logdir_cls convnext_base_cls_stage1 --logdir_seg convnext_base_256_unet_cbam_stage2_ep60 --logdir_stage1_seg convnext_base_256_unet_cbam_stage1_ep30 --num_workers 6

python eval_fp16_v2_sub.py --batch_size 16  --fold 0 \
    --logdir_cls convnext_base_cls_stage1 --logdir_seg convnext_base_256_unet_cbam_stage2_ep60 --logdir_stage1_seg convnext_base_256_unet_cbam_stage1_ep30 --num_workers 6


python train_stage1.py --seed 7024 --model_name resnest101e \
    --lr 1e-3 --batch_size 4 --image_size 320 --seq_len 5 \
    --fold 0 --gpus 4 --epochs 30 --logdir resnest101e_cls_stage1_320 --num_workers 6 --disable_compile

python train_stage2.py --seed 7024 --model_name resnest101e \
    --lr 1e-3 --batch_size 4 --image_size 320 --decoder_type FastFCNImprove --seq_len 5 \
    --fold 0 --gpus 4 --epochs 60 --logdir resnest101e_320_fastfcn_stage2_ep60 --num_workers 6  --disable_compile

python train_stage1_seg.py --seed 7025 --model_name resnest101e \
    --lr 1e-3 --batch_size 4 --image_size 320 --decoder_type FastFCNImprove --seq_len 5 \
    --fold 0 --gpus 4 --epochs 30 --logdir resnest101e_320_fastfcn_stage1_ep30 --num_workers 6  --disable_compile

python eval_fp16.py --batch_size 16  --fold 0 \
    --logdir_cls resnest101e_cls_stage1_320 --logdir_seg resnest101e_320_fastfcn_stage2_ep60 --num_workers 6

python eval_fp16_v2.py --batch_size 16  --fold 0 \
    --logdir_cls resnest101e_cls_stage1_320 --logdir_seg resnest101e_320_fastfcn_stage2_ep60 --logdir_stage1_seg resnest101e_320_fastfcn_stage1_ep30 --num_workers 6

python eval_fp16_v2_sub.py --batch_size 16  --fold 0 \
    --logdir_cls resnest101e_cls_stage1_320 --logdir_seg resnest101e_320_fastfcn_stage2_ep60 --logdir_stage1_seg resnest101e_320_fastfcn_stage1_ep30 --num_workers 6


python train_stage1.py --seed 8026 --model_name convnext_large.fb_in1k --drop_path_rate 0.2 \
    --lr 1e-3 --batch_size 2 --image_size 384 --seq_len 5  \
    --fold 0 --gpus 4 --epochs 30 --logdir convnext_large_384_cls_stage1 --num_workers 6 --disable_compile

python train_stage2.py --seed 8124 --model_name convnext_large.fb_in22k_ft_in1k_384 --drop_path_rate 0.4 \
    --lr 1e-3 --batch_size 2 --image_size 384 --seq_len 5 \
    --fold 0 --gpus 4 --epochs 40 --logdir convnext_large_384_unet_stage2_ep40 --num_workers 6 --disable_compile

python train_stage1_seg.py --seed 8027 --model_name convnext_large.fb_in22k_ft_in1k_384 --drop_path_rate 0.4 \
    --lr 1e-3 --batch_size 2 --image_size 384 --seq_len 5 \
    --fold 0 --gpus 4 --epochs 20 --logdir convnext_large_384_unet_stage1_ep20 --num_workers 6 --disable_compile

python eval_fp16.py --batch_size 16  --fold 0 \
    --logdir_cls resnetrs101_cls_stage1 --logdir_seg convnext_large_384_unet_stage2_ep40 --num_workers 6

python eval_fp16_v2.py --batch_size 16  --fold 0 \
    --logdir_cls resnetrs101_cls_stage1 --logdir_seg convnext_large_384_unet_stage2_ep40 --logdir_stage1_seg convnext_large_384_unet_stage1_ep20 --num_workers 6

python eval_fp16_v2_sub.py --batch_size 16  --fold 0 \
    --logdir_cls resnetrs101_cls_stage1 --logdir_seg convnext_large_384_unet_stage2_ep40 --logdir_stage1_seg convnext_large_384_unet_stage1_ep20 --num_workers 6


python train_stage1.py --seed 10023 --model_name resnetrs200 \
    --lr 1e-3 --batch_size 2 --image_size 384 \
    --fold 0 --gpus 4 --epochs 25 --logdir resnetrs200_384_cls_stage1 --num_workers 6 --disable_compile

python train_stage2.py --seed 10024 --model_name resnetrs200 \
    --lr 1e-3 --batch_size 2 --image_size 384 --decoder_type FastFCNImprove \
    --fold 0 --gpus 4 --epochs 50 --logdir resnetrs200_384_fastfcn_stage2_ep50 --num_workers 6  --disable_compile

python train_stage1_seg.py --seed 10025 --model_name resnetrs200 \
    --lr 1e-3 --batch_size 2 --image_size 384 --decoder_type FastFCNImprove \
    --fold 0 --gpus 4 --epochs 25 --logdir resnetrs200_384_fastfcn_stage1_ep25 --num_workers 6  --disable_compile

python eval_fp16.py --batch_size 16  --fold 0 \
    --logdir_cls resnetrs101_cls_stage1 --logdir_seg resnetrs200_384_fastfcn_stage2_ep50 --num_workers 6

python eval_fp16_v2.py --batch_size 16  --fold 0 \
    --logdir_cls resnetrs101_cls_stage1 --logdir_seg resnetrs200_384_fastfcn_stage2_ep50 --logdir_stage1_seg resnetrs200_384_fastfcn_stage1_ep25 --num_workers 6

python eval_fp16_v2_sub.py --batch_size 16  --fold 0 \
    --logdir_cls resnetrs101_cls_stage1 --logdir_seg resnetrs200_384_fastfcn_stage2_ep50 --logdir_stage1_seg resnetrs200_384_fastfcn_stage1_ep25 --num_workers 6


# python train_stage1.py --seed 3023 --model_name resnetrs101 --drop_path_rate 0.2 \
#     --lr 1e-3 --batch_size 4 --image_size 384 --seq_len 5 \
#     --fold 0 --gpus 4 --epochs 30 --logdir resnetrs101_cls_stage1 --num_workers 6 --disable_compile

python train_stage2.py --seed 14024 --model_name resnetrs101 --drop_path_rate 0.4 \
    --lr 1e-3 --batch_size 2 --image_size 512 --seq_len 5  \
    --fold 0 --gpus 4 --epochs 60 --logdir resnetrs101_512_unet_stage2_ep60 --num_workers 6 --disable_compile

python train_stage1_seg.py --seed 14027 --model_name resnetrs101 --drop_path_rate 0.4 \
    --lr 1e-3 --batch_size 2 --image_size 512 --seq_len 5  \
    --fold 0 --gpus 4 --epochs 30 --logdir resnetrs101_512_unet_stage1_ep30 --num_workers 6 --disable_compile

python eval_fp16.py --batch_size 16  --fold 0 \
    --logdir_cls resnetrs101_cls_stage1 --logdir_seg resnetrs101_512_unet_stage2_ep60 --num_workers 6 

python eval_fp16_v2.py --batch_size 16  --fold 0 \
    --logdir_cls resnetrs101_cls_stage1 --logdir_seg resnetrs101_512_unet_stage2_ep60 --logdir_stage1_seg resnetrs101_512_unet_stage1_ep30 --num_workers 6

python eval_fp16_v2_sub.py --batch_size 16  --fold 0 \
    --logdir_cls resnetrs101_cls_stage1 --logdir_seg resnetrs101_512_unet_stage2_ep60 --logdir_stage1_seg resnetrs101_512_unet_stage1_ep30 --num_workers 6


python train_stage1.py --seed 8026 --model_name convnext_large.fb_in1k --drop_path_rate 0.2 \
    --lr 1e-3 --batch_size 2 --image_size 384 --seq_len 5  \
    --fold 0 --gpus 4 --epochs 30 --logdir convnext_large_384_cls_stage1 --num_workers 6 --disable_compile

python train_stage2.py --seed 21124 --model_name resnetrs101 --drop_path_rate 0.2 \
    --lr 1e-3 --batch_size 2 --image_size 512 --seq_len 5 --decoder_type FastFCNImprove \
    --fold 0 --gpus 4 --epochs 50 --logdir resnetrs101_512_fastfcn_stage2_ep50 --num_workers 6 --disable_compile

python train_stage1_seg.py --seed 21025 --model_name resnetrs101 --drop_path_rate 0.2 \
    --lr 1e-3 --batch_size 2 --image_size 512 --seq_len 5 --decoder_type FastFCNImprove \
    --fold 0 --gpus 4 --epochs 35 --logdir resnetrs101_512_fastfcn_stage1_ep25 --num_workers 6 --disable_compile

python eval_fp16.py --batch_size 16  --fold 0 \
    --logdir_cls resnetrs101_cls_stage1 --logdir_seg resnetrs101_512_fastfcn_stage2_ep50 --num_workers 6

python eval_fp16_v2.py --batch_size 16  --fold 0 \
    --logdir_cls resnetrs101_cls_stage1 --logdir_seg resnetrs101_512_fastfcn_stage2_ep50 --logdir_stage1_seg resnetrs101_512_fastfcn_stage1_ep25 --num_workers 6

python eval_fp16_v2_sub.py --batch_size 16  --fold 0 \
    --logdir_cls resnetrs101_cls_stage1 --logdir_seg resnetrs101_512_fastfcn_stage2_ep50 --logdir_stage1_seg resnetrs101_512_fastfcn_stage1_ep25 --num_workers 6


# python train_stage1.py --seed 3023 --model_name resnetrs101 --drop_path_rate 0.2 \
#     --lr 1e-3 --batch_size 4 --image_size 384 --seq_len 5 \
#     --fold 0 --gpus 4 --epochs 30 --logdir resnetrs101_cls_stage1 --num_workers 6 --disable_compile

python train_stage2.py --seed 14024 --model_name convnext_base.fb_in1k  --drop_path_rate 0.2 \
    --lr 1e-3 --batch_size 2 --image_size 512 --seq_len 5  \
    --fold 0 --gpus 4 --epochs 50 --logdir convnext_base_512_unet_stage2_ep50 --num_workers 6 --disable_compile

python train_stage1_seg.py --seed 14028 --model_name convnext_base.fb_in1k --drop_path_rate 0.2 \
    --lr 1e-3 --batch_size 2 --image_size 512 --seq_len 5  \
    --fold 0 --gpus 4 --epochs 25 --logdir convnext_base_512_unet_stage1_ep25 --num_workers 6 --disable_compile

python eval_fp16.py --batch_size 16  --fold 0 \
    --logdir_cls resnetrs101_cls_stage1 --logdir_seg convnext_base_512_unet_stage2_ep50 --num_workers 6 

python eval_fp16_v2.py --batch_size 16  --fold 0 \
    --logdir_cls resnetrs101_cls_stage1 --logdir_seg convnext_base_512_unet_stage2_ep50 --logdir_stage1_seg convnext_base_512_unet_stage1_ep25 --num_workers 6

python eval_fp16_v2_sub.py --batch_size 16  --fold 0 \
    --logdir_cls resnetrs101_cls_stage1 --logdir_seg convnext_base_512_unet_stage2_ep50 --logdir_stage1_seg convnext_base_512_unet_stage1_ep25 --num_workers 6


# python train_stage1.py --seed 3023 --model_name resnetrs101 --drop_path_rate 0.2 \
#     --lr 1e-3 --batch_size 4 --image_size 384 --seq_len 5 \
#     --fold 0 --gpus 4 --epochs 30 --logdir resnetrs101_cls_stage1 --num_workers 6 --disable_compile

python train_stage2.py --seed 100124 --model_name convnext_base.fb_in1k --drop_path_rate 0.2 \
    --lr 1e-3 --batch_size 2 --image_size 512 --seq_len 5 --attention_type cbam  \
    --fold 0 --gpus 4 --epochs 30 --logdir convnext_base_512_unet_cbam_stage2_ep30 --num_workers 6 --disable_compile

python train_stage1_seg.py --seed 100124 --model_name convnext_base.fb_in1k --drop_path_rate 0.2 \
    --lr 1e-3 --batch_size 2 --image_size 512 --seq_len 5 --attention_type cbam  \
    --fold 0 --gpus 4 --epochs 12 --logdir convnext_base_512_unet_cbam_stage1_ep12 --num_workers 6 --disable_compile

python eval_fp16.py --batch_size 16  --fold 0 \
    --logdir_cls resnetrs101_cls_stage1 --logdir_seg convnext_base_512_unet_cbam_stage2_ep30 --num_workers 6 

python eval_fp16_v2.py --batch_size 16  --fold 0 \
    --logdir_cls resnetrs101_cls_stage1 --logdir_seg convnext_base_512_unet_cbam_stage2_ep30 --logdir_stage1_seg convnext_base_512_unet_cbam_stage1_ep12 --num_workers 6

python eval_fp16_v2_sub.py --batch_size 16  --fold 0 \
    --logdir_cls resnetrs101_cls_stage1 --logdir_seg convnext_base_512_unet_cbam_stage2_ep30 --logdir_stage1_seg convnext_base_512_unet_cbam_stage1_ep12 --num_workers 6