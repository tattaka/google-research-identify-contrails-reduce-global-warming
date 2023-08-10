import argparse
import datetime
import os
import warnings
from glob import glob

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from tqdm import tqdm
from train_stage1 import BASE_DIR  # find_threshold_percentile,
from train_stage1 import EXP_ID as EXP_ID_STAGE1
from train_stage1 import (
    ContrailsDataModule,
    ContrailsLightningModel2_5D,
    dice_score,
    find_threshold_percentile,
)
from train_stage1_seg import EXP_ID as EXP_ID_STAGE1_SEG
from train_stage2 import EXP_ID as EXP_ID_STAGE2
from train_stage2 import ContrailsLightningSegModel2_5D

"""
####### w/o postprocess ##########
resnetrs50_cls_stage1 + resnetrs50_unet_stage2_ep60 fold0: score: 0.6734030203235074 cls_score: 0.8690697082050473 seg_threshold: 0.9935546875000012(0.28173828125) cls_threshold: 0.7296875000000005(0.3288192749023464)
resnetrs101_cls_stage1 + resnetrs101_unet_stage2_ep60 fold0: score: 0.6803797379181896 cls_score: 0.870967241408277 seg_threshold: 0.9937988281250012(0.306884765625) cls_threshold: 0.7296875000000006(0.35661888122558677)
swinv2_base_window16_cls_stage1 + swinv2_base_window16_unet_stage2_ep60 fold0: score: 0.670260576127946 cls_score: 0.8709967537179062 seg_threshold: 0.9936523437500012(0.30419921875) cls_threshold: 0.7085937500000006(0.2810430526733462)
swin_base_patch4_window12_cls_stage1 + swin_base_patch4_window12_unet_stage2_ep60 fold0: score: 0.6729627094042107 cls_score: 0.8615102898435955 seg_threshold: 0.9936523437500012(0.32568359375) cls_threshold: 0.6984375000000005(0.26396560668945357)
convnext_base_cls_stage1 + convnext_base_unet_cbam_stage2_ep60 fold0: score: 0.6725025726451911 cls_score: 0.8568750835916983 seg_threshold: 0.9936523437500012(0.31298828125) cls_threshold: 0.7210937500000003(0.140380859375)
resnest101e_cls_stage1_320 + resnest101e_fastfcn_stage2_320_ep60 fold0: score: 0.6732612551544981 cls_score: 0.8704561623552697 seg_threshold: 0.9937500000000012(0.34765625) cls_threshold: 0.7195312500000006(0.2692956924438507)
resnetrs101_cls_stage1 + convnext_large_unet_stage2_ep40 fold0: score: 0.6810905239462489 cls_score: 0.8697242691291486 seg_threshold: 0.9936523437500012(0.24780350924083905) cls_threshold: 0.7101562500000007(0.2963132858276384)
resnetrs101_cls_stage1 + resnetrs200_fastfcn_stage2_384_ep50 fold0: score: 0.6805724837947459 cls_score: 0.8727942990651427 seg_threshold: 0.9937500000000012(0.302001953125) cls_threshold: 0.7171875000000006(0.28861045837402427)
resnetrs101_cls_stage1 + resnetrs101_512_unet_stage2_ep60 fold0: score: 0.6777777874812647 cls_score: 0.8692375047530722 seg_threshold: 0.9934570312500013(0.2470703125) cls_threshold: 0.7250000000000005(0.31274414062500355)
resnetrs101_cls_stage1 + convnext_base_512_unet_stage2_ep50 fold0: score: 0.6831887644362594 cls_score: 0.8698879745584054 seg_threshold: 0.9935546875000012(0.2305908203125) cls_threshold: 0.7171875000000005(0.3134765625)
####### w/o postprocess ##########
"""
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_args() -> argparse.Namespace:
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument(
        "--seed",
        default=2022,
        type=int,
        metavar="SE",
        help="seed number",
        dest="seed",
    )
    dt_now = datetime.datetime.now()
    parent_parser.add_argument(
        "--logdir_cls",
        default=f"{dt_now.strftime('%Y%m%d-%H-%M-%S')}",
    )
    parent_parser.add_argument(
        "--logdir_seg",
        default=f"{dt_now.strftime('%Y%m%d-%H-%M-%S')}",
    )
    parent_parser.add_argument(
        "--logdir_stage1_seg",
        default=f"{dt_now.strftime('%Y%m%d-%H-%M-%S')}",
    )
    parent_parser.add_argument(
        "--fold",
        type=int,
        default=0,
    )
    parser = ContrailsDataModule.add_model_specific_args(parent_parser)
    return parser.parse_args()


def main(args):
    pl.seed_everything(args.seed)
    warnings.simplefilter("ignore")
    train_df = pd.read_csv(f"{BASE_DIR}/train_metadata.csv")
    valid_df = pd.read_csv(f"{BASE_DIR}/validation_metadata.csv")
    dataloader = ContrailsDataModule(
        train_df=train_df,
        valid_df=valid_df,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
    ).test_dataloader()

    logdir_cls = f"../../logs/exp{EXP_ID_STAGE1}/{args.logdir_cls}/fold0"
    ckpt_path = glob(
        f"{logdir_cls}/**/best_dice.ckpt",
        recursive=True,
    )[0]
    print(f"ckpt_path = {ckpt_path}")
    model = ContrailsLightningModel2_5D.load_from_checkpoint(
        ckpt_path, pretrained=False
    )

    logdir_seg = f"../../logs/exp{EXP_ID_STAGE2}/{args.logdir_seg}/fold0"
    ckpt_path = glob(
        f"{logdir_seg}/**/best_dice.ckpt",
        recursive=True,
    )[0]
    print(f"ckpt_path = {ckpt_path}")
    seg_model = ContrailsLightningSegModel2_5D.load_from_checkpoint(
        ckpt_path, pretrained=False
    )

    logdir_stage1_seg = (
        f"../../logs/exp{EXP_ID_STAGE1_SEG}/{args.logdir_stage1_seg}/fold0"
    )
    ckpt_path = glob(
        f"{logdir_stage1_seg}/**/best_dice.ckpt",
        recursive=True,
    )[0]
    print(f"ckpt_path = {ckpt_path}")
    seg_model_stage1 = ContrailsLightningSegModel2_5D.load_from_checkpoint(
        ckpt_path, pretrained=False
    )

    model.eval()
    model = model.half().to(device=device)
    seg_model.eval()
    seg_model = seg_model.half().to(device=device)
    seg_model_stage1.eval()
    seg_model_stage1 = seg_model_stage1.half().to(device=device)

    cls_logits_all = []
    seg_logit_stage1_all = []
    seg_topk_all = []
    seg_logits_all = []
    label_all = []
    for batch in tqdm(dataloader):
        image, label = batch
        # image = torch.stack(
        #     [image, image.flip(2), image.flip(3), image.flip(2).flip(3)]
        # )
        image = torch.stack([image])
        tta_num, bs, c, h, w = image.shape
        image = image.reshape((tta_num * bs, c, h, w))
        with torch.no_grad():
            seg_input = image.half().to(device=device)
            seg_logits_stage1 = torch.sigmoid(
                seg_model_stage1.model_ema.module(seg_input)
            ).reshape((tta_num, bs, -1, h, w))
            # seg_logits_stage1[1] = seg_logits_stage1[1].flip(2)
            # seg_logits_stage1[2] = seg_logits_stage1[2].flip(3)
            # seg_logits_stage1[3] = seg_logits_stage1[3].flip(2).flip(3)
            seg_logits_stage1 = seg_logits_stage1.mean(0).detach().cpu().numpy()
            seg_logit_stage1_all.append(seg_logits_stage1)
            seg_topk_stage1 = np.mean(
                np.sort(seg_logits_stage1.reshape(bs, -1), axis=1)[:, -100:],
                axis=1,
                keepdims=True,
            )
            seg_topk_all.append(seg_topk_stage1)
            cls_input = image.half().to(device=device)
            cls_logits = (
                torch.sigmoid(model.model_ema.module(cls_input))
                .reshape((tta_num, bs, -1))
                .mean(0)
                .detach()
                .cpu()
                .numpy()
            )  # (bs, c)
            cls_logits_all.append(cls_logits)  # (bs, c)
            # cls_logits_all.append(seg_topk_stage1)  # (bs, c)
            seg_input = image.half().to(device=device)
            seg_logits = torch.sigmoid(seg_model.model_ema.module(seg_input)).reshape(
                (tta_num, bs, -1, h, w)
            )
            # seg_logits[1] = seg_logits[1].flip(2)
            # seg_logits[2] = seg_logits[2].flip(3)
            # seg_logits[3] = seg_logits[3].flip(2).flip(3)
            seg_logits = seg_logits.mean(0).detach().cpu().numpy()
            seg_logits_all.append(seg_logits)  # (bs, c, h, w)
        label_all.append(label)
    label_all = np.concatenate(label_all)
    cls_logits_all = np.concatenate(cls_logits_all)
    seg_topk_all = np.concatenate(seg_topk_all)
    seg_logit_stage1_all = np.concatenate(seg_logit_stage1_all)
    seg_logits_all = np.concatenate(seg_logits_all)

    cls_logits_ensemble = (cls_logits_all + seg_topk_all) / 2
    cls_threshold_percent = find_threshold_percentile(
        label_all.max((2, 3)), cls_logits_ensemble
    )
    cls_threshold = np.quantile(cls_logits_ensemble, cls_threshold_percent)
    cls_preds_ensemble = cls_logits_ensemble > cls_threshold

    seg_logits_ensemble = (seg_logit_stage1_all + seg_logits_all) / 2
    pos_idx = label_all.max((1, 2, 3)) > 0
    seg_threshold_percent = find_threshold_percentile(
        label_all[pos_idx], seg_logits_ensemble[pos_idx]
    )
    seg_threshold = np.quantile(
        seg_logits_ensemble[pos_idx],
        seg_threshold_percent,
    )
    seg_preds_ensemble = (seg_logits_ensemble > seg_threshold) * cls_preds_ensemble[
        :, :, None, None
    ]

    score = dice_score(label_all, seg_preds_ensemble)
    cls_score = dice_score(label_all.max((2, 3)), cls_preds_ensemble)
    np.save(os.path.join(logdir_cls, "cls_logits"), cls_logits_all)
    np.save(os.path.join(logdir_stage1_seg, "cls_logits"), seg_topk_all)
    os.makedirs(os.path.join(logdir_seg, "seg_logits"), exist_ok=True)
    os.makedirs(os.path.join(logdir_stage1_seg, "seg_logits"), exist_ok=True)
    for i, idx in enumerate(valid_df.record_id):
        np.save(
            os.path.join(logdir_seg, "seg_logits", f"{idx}"),
            seg_logits_all[i],
        )
        np.save(
            os.path.join(logdir_stage1_seg, "seg_logits", f"{idx}"),
            seg_logit_stage1_all[i],
        )
    print(
        f"{args.logdir_cls} + {args.logdir_seg} fold0: score: {score} cls_score: {cls_score} seg_threshold: {seg_threshold_percent}({seg_threshold}) cls_threshold: {cls_threshold_percent}({cls_threshold})"
    )


if __name__ == "__main__":
    main(get_args())
