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
resnetrs50_cls_stage1 + resnetrs50_unet_stage2_ep60 fold0: score: 0.6734186187292283 cls_score: 0.8690697082050473 seg_threshold: 0.9932617187500014 cls_threshold: 0.7296875000000005
resnetrs101_cls_stage1 + resnetrs101_unet_stage2_ep60 fold0: score: 0.6802517586112529 cls_score: 0.870967241408277 seg_threshold: 0.9932617187500012 cls_threshold: 0.7296875000000006
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
    seg_logits_all = []
    label_all = []

    train_df = pd.read_csv(f"{BASE_DIR}/train_metadata.csv")
    valid_df = pd.read_csv(f"{BASE_DIR}/validation_metadata.csv")
    dataloader = ContrailsDataModule(
        train_df=train_df,
        valid_df=valid_df,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
    ).test_dataloader()
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
            seg_logits_stage1 = seg_logits_stage1.mean(0).detach().cpu().numpy()
            seg_topk_stage1 = np.mean(
                np.sort(seg_logits_stage1.reshape(bs, -1), axis=1)[:, -100:],
                axis=1,
                keepdims=True,
            )
            cls_input = image.half().to(device=device)
            cls_logits = (
                torch.sigmoid(model.model_ema.module(cls_input))
                .reshape((tta_num, bs, -1))
                .mean(0)
                .detach()
                .cpu()
                .numpy()
            )  # (bs, c)
            cls_logits_all.append((cls_logits + seg_topk_stage1) / 2)  # (bs, c)
            # cls_logits_all.append(seg_topk_stage1)  # (bs, c)
        label_all.append(label)
    label_all = np.concatenate(label_all)
    cls_logits_all = np.concatenate(cls_logits_all)

    cls_threshold_percent = find_threshold_percentile(
        label_all.max((2, 3)), cls_logits_all
    )
    cls_threshold = np.quantile(cls_logits_all, cls_threshold_percent)
    cls_preds_all = cls_logits_all > cls_threshold

    dataloader = ContrailsDataModule(
        train_df=train_df,
        valid_df=valid_df[cls_preds_all[:, 0] > 0.5],
        num_workers=args.num_workers,
        batch_size=args.batch_size,
    ).test_dataloader()
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
            seg_input = image.half().to(device=device)
            seg_logits = torch.sigmoid(seg_model.model_ema.module(seg_input)).reshape(
                (tta_num, bs, -1, h, w)
            )
            # seg_logits[1] = seg_logits[1].flip(2)
            # seg_logits[2] = seg_logits[2].flip(3)
            # seg_logits[3] = seg_logits[3].flip(2).flip(3)
            seg_logits = seg_logits.mean(0).detach().cpu().numpy()
            seg_logits_all.append((seg_logits + seg_logits_stage1) / 2)  # (bs, c, h, w)

    seg_logits_all = np.concatenate(seg_logits_all)
    pos_idx = cls_preds_all[:, 0] > 0.5
    seg_threshold_percent = find_threshold_percentile(
        label_all[pos_idx], seg_logits_all
    )
    seg_threshold = np.quantile(
        seg_logits_all,
        seg_threshold_percent,
    )
    seg_preds_all = np.zeros((len(cls_preds_all), 1, 256, 256))
    seg_preds_all[pos_idx] = seg_logits_all > seg_threshold

    score = dice_score(label_all, seg_preds_all)
    cls_score = dice_score(label_all.max((2, 3)), cls_preds_all)
    print(
        f"{args.logdir_cls} + {args.logdir_seg} fold0: score: {score} cls_score: {cls_score} seg_threshold: {seg_threshold_percent} cls_threshold: {cls_threshold_percent}"
    )


if __name__ == "__main__":
    main(get_args())
