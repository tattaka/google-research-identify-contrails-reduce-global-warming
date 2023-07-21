import argparse
import os

# import datetime
import warnings
from glob import glob

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from timm.utils import ModelEmaV2
from torch.nn import functional as F
from tqdm import tqdm
from train_stage1 import BASE_DIR  # find_threshold_percentile,
from train_stage1 import EXP_ID as EXP_ID_STAGE1
from train_stage1 import (
    ContrailsDataModule,
    ContrailsLightningModel,
    ContrailsModel,
    dice_score,
)
from train_stage1_seg import EXP_ID as EXP_ID_STAGE1_SEG
from train_stage2 import EXP_ID as EXP_ID_STAGE2
from train_stage2 import ContrailsLightningSegModel, ContrailsSegModel

"""
####### w/o postprocess ##########
fold0: score: 0.6857790176195541 cls_score: 0.8614227199290305 seg_threshold: 0.9933593750000013(0.3466796875) cls_threshold: 0.7218750000000005(0.3066787719726567)
fold0: score: 0.7842369341505444
####### w/o postprocess ##########
"""
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ContrailsModelPL(ContrailsModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward_image_feats(self, img):
        # mean = img.mean(dim=(1, 2, 3), keepdim=True)
        # std = img.std(dim=(1, 2, 3), keepdim=True) + 1e-6
        # img = (img - mean) / std
        bs, _, h, w = img.shape
        img = (
            img.reshape(bs, 3, 8, h, w)
            .permute((0, 2, 1, 3, 4))
            .reshape(bs * 8, 3, h, w)
        )
        img = F.interpolate(
            img, size=(self.image_size, self.image_size), mode="bilinear"
        )
        img_feats = self.encoder(img)
        if self.output_fmt == "NHWC":
            img_feats = [
                img_feat.permute(0, 3, 1, 2).contiguous() for img_feat in img_feats
            ]
        return img_feats[-1]


class ContrailsLightningModelPL(ContrailsLightningModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.__build_model(
            model_name=kwargs["model_name"],
            pretrained=kwargs["pretrained"],
            drop_path_rate=kwargs["drop_path_rate"],
            in_chans=kwargs["in_chans"],
            num_class=kwargs["num_class"],
            image_size=kwargs["image_size"],
        )
        if not kwargs["disable_compile"]:
            self.__compile_model()

    def __build_model(
        self,
        model_name: str = "resnet34",
        pretrained: bool = False,
        drop_path_rate: float = 0,
        in_chans: int = 3,
        num_class: int = 1,
        image_size: int = 256,
    ):
        self.model = ContrailsModelPL(
            model_name=model_name,
            pretrained=pretrained,
            drop_path_rate=drop_path_rate,
            in_chans=in_chans,
            num_class=num_class,
            image_size=image_size,
        )
        self.model_ema = ModelEmaV2(self.model, decay=0.998)

    def __compile_model(self):
        self.model = torch.compile(self.model)
        self.model_ema = torch.compile(self.model_ema)


class ContrailsSegModelPL(ContrailsSegModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward_image_feats(self, img):
        # mean = img.mean(dim=(1, 2, 3), keepdim=True)
        # std = img.std(dim=(1, 2, 3), keepdim=True) + 1e-6
        # img = (img - mean) / std
        bs, _, h, w = img.shape
        img = (
            img.reshape(bs, 3, 8, h, w)
            .permute((0, 2, 1, 3, 4))
            .reshape(bs * 8, 3, h, w)
        )
        img = F.interpolate(
            img, size=(self.image_size, self.image_size), mode="bilinear"
        )
        img_feats = self.encoder(img)
        if self.output_fmt == "NHWC":
            img_feats = [
                img_feat.permute(0, 3, 1, 2).contiguous() for img_feat in img_feats
            ]
        return img_feats


class ContrailsLightningSegModelPL(ContrailsLightningSegModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.__build_model(
            model_name=kwargs["model_name"],
            pretrained=kwargs["pretrained"],
            drop_path_rate=kwargs["drop_path_rate"],
            decoder_type=kwargs["decoder_type"],
            center=kwargs["center"],
            attention_type=kwargs["attention_type"],
            in_chans=kwargs["in_chans"],
            num_class=kwargs["num_class"],
            image_size=kwargs["image_size"],
        )
        if not kwargs["disable_compile"]:
            self.__compile_model()

    def __build_model(
        self,
        model_name: str = "resnet34",
        pretrained: bool = False,
        drop_path_rate: float = 0,
        decoder_type: str = "UNet",  # UNet or FastFCNImprove
        center=None,
        attention_type=None,
        in_chans: int = 3,
        num_class: int = 1,
        image_size: int = 256,
    ):
        self.model = ContrailsSegModelPL(
            model_name=model_name,
            pretrained=pretrained,
            drop_path_rate=drop_path_rate,
            decoder_type=decoder_type,
            center=center,
            attention_type=attention_type,
            in_chans=in_chans,
            num_class=num_class,
            image_size=image_size,
        )
        self.model_ema = ModelEmaV2(self.model, decay=0.998)

    def __compile_model(self):
        self.model = torch.compile(self.model)
        self.model_ema = torch.compile(self.model_ema)


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
        "swinv2_base_window16_cls_stage1",
        "swin_base_patch4_window12_cls_stage1",
        "resnetrs50_cls_stage1",
    ],
    "logdir_stage1_seg": [
        "resnetrs101_unet_stage1_ep30",
        "swinv2_base_window16_unet_stage1_ep30",
        "swin_base_patch4_window12_unet_stage1_ep30",
        "resnetrs50_unet_stage1_ep30",
        "convnext_base_unet_stage1_ep30",
    ],
    "logdir_seg": [
        "resnetrs101_unet_stage2_ep60",
        "swinv2_base_window16_unet_stage2_ep60",
        "swin_base_patch4_window12_unet_stage2_ep60",
        "resnetrs50_unet_stage2_ep60",
        "convnext_base_unet_stage2_ep60",
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
        ContrailsLightningModelPL.load_from_checkpoint(pt)
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
        ContrailsLightningSegModelPL.load_from_checkpoint(pt)
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
        ContrailsLightningSegModelPL.load_from_checkpoint(pt)
        .eval()
        .half()
        .to(device=device)
        for pt in ckpt_path
    ]

    seg_preds_all = []
    label_all = []

    train_df = pd.read_csv(f"{BASE_DIR}/train_metadata.csv")
    dataloader = ContrailsDataModule(
        train_df=train_df,
        valid_df=train_df,
        num_workers=args.num_workers,
        batch_size=2,
    ).test_dataloader()
    idx = 0
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
                    .reshape((tta_num, bs * 8, -1, h, w))
                    .mean(0)
                    .detach()
                    .cpu()
                    .numpy()
                    for m in seg_model_stage1
                ]
            )
            seg_topk_stage1 = np.stack(
                [
                    np.mean(
                        np.sort(
                            logit.reshape(bs * 8, -1),
                            axis=1,
                        )[:, -100:],
                        axis=1,
                        keepdims=True,
                    )
                    for logit in seg_logits_stage1
                ]
            )  # (model_len, bs, c)

            seg_logits_stage2 = np.stack(
                [
                    torch.sigmoid(m.model_ema.module(image))
                    .reshape((tta_num, bs * 8, -1, h, w))
                    .mean(0)
                    .detach()
                    .cpu()
                    .numpy()
                    for m in seg_model
                ]
            )

            cls_logits = np.stack(
                [
                    torch.sigmoid(m.model_ema.module(image))
                    .reshape((tta_num, bs * 8, -1))
                    .mean(0)
                    .detach()
                    .cpu()
                    .numpy()
                    for m in model
                ]
            )  # (model_len, bs, c)
            cls_logits = np.concatenate([cls_logits, seg_topk_stage1]).mean(0)
            cls_preds = cls_logits > 0.3066787719726567
            seg_logits = np.concatenate([seg_logits_stage1, seg_logits_stage2]).mean(0)
            seg_preds = seg_logits > 0.3466796875
            seg_preds = (
                (seg_preds * cls_preds[:, :, None, None])
                .reshape(bs, 1, 8, 256, 256)
                .transpose((0, 3, 4, 1, 2))
            )  # (bs, h, w, 1, 8)
            seg_logits = (
                (seg_logits * cls_preds[:, :, None, None])
                .reshape(bs, 1, 8, 256, 256)
                .transpose((0, 3, 4, 1, 2))
            )
            seg_preds_all.append(seg_preds[:, :, :, :, 4])

            seg_preds[:, :, :, :, 4] = label.numpy().transpose((0, 2, 3, 1))
            seg_logits[:, :, :, :, 4] = label.numpy().transpose((0, 2, 3, 1))
            record_ids = train_df.iloc[idx : idx + len(batch)].record_id.tolist()
            for record_id, pred_mask, pred_logit in zip(
                record_ids, seg_preds, seg_logits
            ):
                os.makedirs(
                    f"../../input/ash_dataset_pseudo/train/{record_id}", exist_ok=True
                )
                np.save(
                    f"../../input/ash_dataset_pseudo/train/{record_id}/mask_pseudo.npy",
                    pred_mask,
                )
                np.save(
                    f"../../input/ash_dataset_pseudo/train/{record_id}/logit_pseudo.npy",
                    pred_logit,
                )
        idx += len(batch)
        label_all.append(label)
        # break
    label_all = np.concatenate(label_all)
    seg_preds_all = np.concatenate(seg_preds_all)
    score = dice_score(label_all.transpose((0, 2, 3, 1)), seg_preds_all)
    print(f"fold0: score: {score}")


if __name__ == "__main__":
    main(get_args())
