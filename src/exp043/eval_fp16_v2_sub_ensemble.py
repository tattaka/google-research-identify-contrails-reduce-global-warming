import argparse

# import datetime
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
fold0: score: 0.6952059140140283 cls_score: 0.8796291282478833 seg_threshold: 0.9935546875000012(0.34765625) cls_threshold: 0.7156250000000006(0.2546157836914079)
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
    parent_parser.add_argument(
        "--fold",
        type=int,
        default=0,
    )
    parser = ContrailsDataModule.add_model_specific_args(parent_parser)
    return parser.parse_args()


model_confs = {
    "logdir_cls": [
        "resnetrs101_cls_stage1",
        "convnext_base_cls_stage1",
        "swin_base_patch4_window12_cls_stage1",
        "resnest101e_cls_stage1_320",
    ],
    "logdir_stage1_seg": [
        "resnetrs50_unet_stage1_ep30",
        "resnetrs101_unet_stage1_ep30",
        "swinv2_base_window16_unet_stage1_ep30",
        "swin_base_patch4_window12_unet_stage1_ep30",
        "convnext_base_unet_cbam_stage1_ep30",
        "resnest101e_fastfcn_stage1_320_ep30",
        "convnext_large_unet_stage1_ep20",
    ],
    "logdir_seg": [
        "resnetrs50_unet_stage2_ep60",
        "resnetrs101_unet_stage2_ep60",
        "swinv2_base_window16_unet_stage2_ep60",
        "swin_base_patch4_window12_unet_stage2_ep60",
        "convnext_base_unet_cbam_stage2_ep60",
        "resnest101e_fastfcn_stage2_320_ep60",
        "convnext_large_unet_stage2_ep40",
    ],
}


def main(args):
    pl.seed_everything(args.seed)
    warnings.simplefilter("ignore")
    ckpt_path = [
        glob(f"../../logs/exp{EXP_ID_STAGE1}/{conf}/**/best_dice.ckpt", recursive=True)[
            0
        ]
        for conf in model_confs["logdir_cls"]
    ]
    print(f"ckpt_path = {ckpt_path}")
    model = [
        ContrailsLightningModel2_5D.load_from_checkpoint(pt)
        .eval()
        .half()
        .to(device=device)
        for pt in ckpt_path
    ]

    ckpt_path = [
        glob(f"../../logs/exp{EXP_ID_STAGE2}/{conf}/**/best_dice.ckpt", recursive=True)[
            0
        ]
        for conf in model_confs["logdir_seg"]
    ]
    print(f"ckpt_path = {ckpt_path}")
    seg_model = [
        ContrailsLightningSegModel2_5D.load_from_checkpoint(pt)
        .eval()
        .half()
        .to(device=device)
        for pt in ckpt_path
    ]

    ckpt_path = [
        glob(
            f"../../logs/exp{EXP_ID_STAGE1_SEG}/{conf}/**/best_dice.ckpt",
            recursive=True,
        )[0]
        for conf in model_confs["logdir_stage1_seg"]
    ]
    print(f"ckpt_path = {ckpt_path}")
    seg_model_stage1 = [
        ContrailsLightningSegModel2_5D.load_from_checkpoint(pt)
        .eval()
        .half()
        .to(device=device)
        for pt in ckpt_path
    ]

    cls_logits_all = []
    seg_logits_all = []
    label_all = []

    train_df = pd.read_csv(f"{BASE_DIR}/train_metadata.csv")
    valid_df = pd.read_csv(f"{BASE_DIR}/validation_metadata.csv")
    dataloader = ContrailsDataModule(
        train_df=train_df,
        valid_df=valid_df,
        num_workers=args.num_workers,
        batch_size=4,
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
            image = image.half().to(device=device)
            # seg_logits_stage1 = torch.sigmoid(
            #     seg_model_stage1.model_ema.module(image)
            # ).reshape((tta_num, bs, -1, h, w))
            seg_topk_stage1 = np.stack(
                [
                    np.mean(
                        np.sort(
                            torch.sigmoid(m.model_ema.module(image))
                            .reshape((tta_num, bs, -1, h, w))
                            .mean(0)
                            .detach()
                            .cpu()
                            .numpy()
                            .reshape(bs, -1),
                            axis=1,
                        )[:, -100:],
                        axis=1,
                        keepdims=True,
                    )
                    for m in seg_model_stage1
                ]
            )  # (model_len, bs, c)

            cls_logits = np.stack(
                [
                    torch.sigmoid(m.model_ema.module(image))
                    .reshape((tta_num, bs, -1))
                    .mean(0)
                    .detach()
                    .cpu()
                    .numpy()
                    for m in model
                ]
            )  # (model_len, bs, c)

            cls_logits_all.append(
                np.concatenate([cls_logits, seg_topk_stage1]).mean(0)
            )  # (bs, c)
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
        batch_size=4,
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
            image = image.half().to(device=device)
            seg_logits_stage1 = np.stack(
                [
                    torch.sigmoid(m.model_ema.module(image))
                    .reshape((tta_num, bs, -1, h, w))
                    .mean(0)
                    .detach()
                    .cpu()
                    .numpy()
                    for m in seg_model_stage1
                ]
            ).mean(0)

            seg_logits = np.stack(
                [
                    torch.sigmoid(m.model_ema.module(image))
                    .reshape((tta_num, bs, -1, h, w))
                    .mean(0)
                    .detach()
                    .cpu()
                    .numpy()
                    for m in seg_model
                ]
            ).mean(0)
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
        f"fold0: score: {score} cls_score: {cls_score} seg_threshold: {seg_threshold_percent}({seg_threshold}) cls_threshold: {cls_threshold_percent}({cls_threshold})"
    )


if __name__ == "__main__":
    main(get_args())
