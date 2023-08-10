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
from train_stage2 import EXP_ID as EXP_ID_STAGE2
from train_stage2 import ContrailsLightningSegModel2_5D

"""
####### w/o postprocess ##########
resnetrs50_cls_stage1 + resnetrs50_256_unet_stage2_ep60 fold0: score: 0.6767507655362093 cls_score: 0.8534075287599444 seg_threshold: 0.9936523437500012(0.94580078125) cls_threshold: 0.7203125000000006(0.32561492919921986)
resnetrs101_cls_stage1 + resnetrs101_384_unet_stage2_ep60 fold0: score: 0.6794366053522322 cls_score: 0.8546336476443106 seg_threshold: 0.9933593750000013(0.95166015625) cls_threshold: 0.7453125000000007(0.4987239837646501)
convnext_base_cls_stage1 + convnext_base_256_unet_cbam_stage2_ep60 fold0: score: 0.6672015078455318 cls_score: 0.8396658564509807 seg_threshold: 0.9936523437500012(0.95068359375) cls_threshold: 0.7164062500000006(0.11140966415405343)
resnest101e_cls_stage1_320 + resnest101e_320_fastfcn_stage2_ep60 fold0: score: 0.6712102127310827 cls_score: 0.845724405830093 seg_threshold: 0.9936523437500012(0.94873046875) cls_threshold: 0.7179687500000005(0.3080625534057633)
resnetrs101_cls_stage1 + convnext_large_384_unet_stage2_ep40 fold0: score: 0.6707800037759941 cls_score: 0.8546336476443106 seg_threshold: 0.9933593750000013(0.951171875) cls_threshold: 0.7453125000000007(0.4987239837646501)
resnetrs101_cls_stage1 + resnetrs101_512_unet_stage2_ep60 fold0: score: 0.6748742244726561 cls_score: 0.8546336476443106 seg_threshold: 0.9936523437500012(0.95947265625) cls_threshold: 0.7453125000000007(0.4987239837646501)
resnetrs101_cls_stage1 + resnetrs101_512_fastfcn_stage2_ep50 fold0: score: 0.6708517039604829 cls_score: 0.8546336476443106 seg_threshold: 0.9936523437500012(0.95703125) cls_threshold: 0.7453125000000007(0.4987239837646501)
resnetrs101_cls_stage1 + convnext_base_512_unet_stage2_ep50 fold0: score: 0.6669293399578479 cls_score: 0.8546336476443106 seg_threshold: 0.9935546875000012(0.95458984375) cls_threshold: 0.7453125000000007(0.4987239837646501)
resnetrs101_cls_stage1 + convnext_base_512_unet_cbam_stage2_ep30 fold0: score: 0.6825580189136958 cls_score: 0.8747623078147367 seg_threshold: 0.9937500000000012(0.7998046875) cls_threshold: 0.7296875000000006(0.45517921447754267)
####### w/o postprocess ##########
"""
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_IMG_SIZE = 256


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

    model.eval()
    model = model.half().to(device=device)
    seg_model.eval()
    seg_model = seg_model.half().to(device=device)

    cls_logits_all = []
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
            cls_input = image.half().to(device=device)
            cls_logits_all.append(
                torch.sigmoid(model.model_ema.module(cls_input))
                .reshape((tta_num, bs, -1))
                .mean(0)
                .detach()
                .cpu()
                .numpy()
            )  # (bs, c)
            seg_input = image.half().to(device=device)
            seg_logits = torch.sigmoid(seg_model.model_ema.module(seg_input)).reshape(
                (tta_num, bs, -1, h, w)
            )
            # seg_logits[1] = seg_logits[1].flip(2)
            # seg_logits[2] = seg_logits[2].flip(3)
            # seg_logits[3] = seg_logits[3].flip(2).flip(3)
            seg_logits_all.append(
                seg_logits.mean(0).detach().cpu().numpy()
            )  # (bs, c, h, w)
        label_all.append(label)
    label_all = np.concatenate(label_all)
    cls_logits_all = np.concatenate(cls_logits_all)
    seg_logits_all = np.concatenate(seg_logits_all)

    cls_threshold_percent = find_threshold_percentile(
        label_all.max((2, 3)), cls_logits_all
    )
    cls_threshold = np.quantile(cls_logits_all, cls_threshold_percent)
    cls_preds_all = cls_logits_all > cls_threshold

    pos_idx = label_all.max((1, 2, 3)) > 0
    seg_threshold_percent = find_threshold_percentile(
        label_all[pos_idx], seg_logits_all[pos_idx]
    )
    seg_threshold = np.quantile(
        seg_logits_all[pos_idx],
        seg_threshold_percent,
    )
    seg_preds_all = (seg_logits_all > seg_threshold) * cls_preds_all[:, :, None, None]

    score = dice_score(label_all, seg_preds_all)
    cls_score = dice_score(label_all.max((2, 3)), cls_preds_all)
    np.save(os.path.join(logdir_cls, "cls_logits"), cls_logits_all)
    os.makedirs(os.path.join(logdir_seg, "seg_logits"), exist_ok=True)
    for i, idx in enumerate(valid_df.record_id):
        np.save(
            os.path.join(logdir_seg, "seg_logits", f"{idx}"),
            seg_logits_all[i],
        )
    print(
        f"{args.logdir_cls} + {args.logdir_seg} fold0: score: {score} cls_score: {cls_score} seg_threshold: {seg_threshold_percent}({seg_threshold}) cls_threshold: {cls_threshold_percent}({cls_threshold})"
    )


if __name__ == "__main__":
    main(get_args())
