import argparse
import datetime
import math
import os
import warnings
from functools import partial
from typing import Callable, List, Tuple

import albumentations as albu
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import timm
import torch
from albumentations.pytorch import ToTensorV2
from decoder_utils import FastFCNImproveHead, UNetHead
from pytorch_lightning import LightningDataModule, callbacks
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities import rank_zero_info
from scipy.optimize import minimize
from timm.utils import ModelEmaV2
from torch import nn
from torch.nn import functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import get_cosine_schedule_with_warmup

EXP_ID = "055"
COMMENT = """
UNet Baseline without negative + normalize + w/o grid distortion + post_conv3d + seq_len option(default: 7) + low flip, rotate + reverse_video_aug + use indivisual mask
"""
BASE_DIR = "../../input/ash_dataset/"

img_mean = (
    np.concatenate([np.asarray([55.936405, 140.56143, 143.563]) for _ in range(8)])
    / 255.0
)
img_std = (
    np.concatenate([np.asarray([30.419374, 37.97873, 48.56115]) for _ in range(8)])
    / 255.0
)


def get_transforms(train: bool = False) -> Callable:
    if train:
        return albu.Compose(
            [
                albu.Resize(height=256, width=256),
                albu.Flip(p=0.25),
                albu.RandomRotate90(p=0.25),
                albu.ShiftScaleRotate(
                    p=0.5, scale_limit=0.3, shift_limit=0.1, rotate_limit=0
                ),
                albu.Rotate(limit=45, p=0.25),
                albu.Normalize(img_mean, img_std),
                ToTensorV2(transpose_mask=True),
            ]
        )
    else:
        return albu.Compose(
            [
                albu.Normalize(img_mean, img_std),
                ToTensorV2(transpose_mask=True),
            ]
        )


class ContrailsDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        mode: str = "train",  # "train" | "valid" | "test"
    ):
        self.df = df
        self.mode = mode
        self.train = mode == "train"
        self.transforms = get_transforms(self.train)

    def __len__(self) -> int:
        return len(self.df)

    def np_load(self, file) -> np.ndarray:
        if type(file) == str:
            file = open(file, "rb")
        header = file.read(128)
        if not header:
            return None
        descr = str(header[19:25], "utf-8").replace("'", "").replace(" ", "")
        shape = tuple(
            int(num)
            for num in str(header[60:120], "utf-8")
            .replace(", }", "")
            .replace("(", "")
            .replace(")", "")
            .split(",")
        )
        datasize = np.lib.format.descr_to_dtype(descr).itemsize
        for dimension in shape:
            datasize *= dimension
        return np.ndarray(shape, dtype=descr, buffer=file.read(datasize))

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image = (
            self.np_load(
                os.path.join(self.df.ash_data_path.iloc[idx], "img.npy")
            ).astype(np.float32)
            * 255
        )  # (h, w, c, l)
        h, w, c, d = image.shape
        image = image.reshape(h, w, c * d)
        if self.train:
            label = self.np_load(
                os.path.join(self.df.ash_data_path.iloc[idx], "mask_individual.npy")
            ).astype(np.float32)
            label = np.clip((label * 2).sum(-1) / label.shape[-1], 0, 1)
        else:
            label = self.np_load(
                os.path.join(self.df.ash_data_path.iloc[idx], "mask.npy")
            ).astype(np.float32)
        aug = self.transforms(image=image, mask=label)
        image = aug["image"]  # (c * d, h, w)
        label = aug["mask"]
        return (
            image,
            label,
        )


class ContrailsDataModule(LightningDataModule):
    def __init__(
        self,
        train_df: pd.DataFrame,
        valid_df: pd.DataFrame,
        num_workers: int = 4,
        batch_size: int = 16,
    ):
        super().__init__()

        self._num_workers = num_workers
        self._batch_size = batch_size
        self.train_df = train_df
        self.valid_df = valid_df
        self.save_hyperparameters(
            "num_workers",
            "batch_size",
        )

    def create_dataset(self, mode: str = "train") -> ContrailsDataset:
        if mode == "train":
            return ContrailsDataset(
                df=self.train_df,
                mode=mode,
            )
        else:
            return ContrailsDataset(
                df=self.valid_df,
                mode=mode,
            )

    def __dataloader(self, mode: str = "train") -> DataLoader:
        """Train/validation loaders."""
        dataset = self.create_dataset(mode)
        return DataLoader(
            dataset=dataset,
            batch_size=self._batch_size,
            num_workers=self._num_workers,
            shuffle=(mode == "train"),
            drop_last=(mode == "train"),
            pin_memory=True,
        )

    def train_dataloader(self) -> DataLoader:
        return self.__dataloader(mode="train")

    def val_dataloader(self) -> DataLoader:
        return self.__dataloader(mode="valid")

    def test_dataloader(self) -> DataLoader:
        return self.__dataloader(mode="test")

    @staticmethod
    def add_model_specific_args(
        parent_parser: argparse.ArgumentParser,
    ) -> argparse.ArgumentParser:
        parser = parent_parser.add_argument_group("ContrailsDataModule")
        parser.add_argument(
            "--num_workers",
            default=4,
            type=int,
            metavar="W",
            help="number of CPU workers",
            dest="num_workers",
        )
        parser.add_argument(
            "--batch_size",
            default=16,
            type=int,
            metavar="BS",
            help="number of sample in a batch",
            dest="batch_size",
        )
        return parent_parser


def downsample_conv(
    in_channels: int,
    out_channels: int,
    stride: int = 2,
):
    return nn.Sequential(
        *[
            nn.Conv3d(
                in_channels,
                out_channels,
                1,
                stride=(1, stride, stride),
                padding=0,
                bias=False,
            ),
            nn.BatchNorm3d(out_channels),
        ]
    )


class ResidualConv3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        out_channels: int,
        stride: int = 2,
    ):
        super().__init__()

        self.conv1 = nn.Conv3d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(mid_channels)
        self.act1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Sequential(
            nn.Conv3d(mid_channels, mid_channels, kernel_size=1, stride=1, bias=False),
            nn.Conv3d(
                mid_channels,
                mid_channels,
                kernel_size=3,
                stride=(1, stride, stride),
                padding=1,
                bias=False,
                groups=mid_channels,
            ),
        )
        self.bn2 = nn.BatchNorm3d(mid_channels)
        self.act2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv3d(mid_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(out_channels)

        self.act3 = nn.ReLU(inplace=True)
        self.downsample = downsample_conv(
            in_channels,
            out_channels,
            stride=stride,
        )
        self.stride = stride
        self.zero_init_last()

    def zero_init_last(self):
        if getattr(self.bn3, "weight", None) is not None:
            nn.init.zeros_(self.bn3.weight)

    def forward(self, x: torch.Tensor):
        shortcut = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)

        x = self.conv3(x)
        x = self.bn3(x)

        if self.downsample is not None:
            shortcut = self.downsample(shortcut)
        x += shortcut
        x = self.act3(x)

        return x


class ContrailsSegModel2_5D(nn.Module):
    def __init__(
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
        seq_len: int = 7,
    ):
        super().__init__()
        self.encoder = timm.create_model(
            model_name,
            pretrained=pretrained,
            in_chans=in_chans,
            features_only=True,
            drop_path_rate=drop_path_rate,
        )
        self.seq_len = seq_len
        self.output_fmt = getattr(self.encoder, "output_fmt", "NHCW")
        self.in_chans = in_chans
        assert image_size % 32 == 0
        self.image_size = image_size
        num_features = self.encoder.feature_info.channels()
        conv3d = []
        for ch_3d in num_features:
            conv3d.append(
                nn.Sequential(
                    *[
                        ResidualConv3D(
                            ch_3d,
                            ch_3d // 4,
                            ch_3d,
                            1,
                        )
                        for _ in range(3)
                    ]
                )
            )
        self.conv3d = nn.ModuleList(conv3d)
        if decoder_type == "UNet":
            self.head = UNetHead(
                encoder_channels=num_features,
                num_class=num_class,
                center=center,
                attention_type=attention_type,
                classification=False,
                deep_supervision=False,
            )
        elif decoder_type == "FastFCNImprove":
            self.head = FastFCNImproveHead(
                encoder_channels=num_features,
                num_class=num_class,
                attention_type=attention_type,
                classification=False,
                deep_supervision=False,
            )
        else:
            raise NotImplementedError

    def forward_image_feats(self, img):
        # img -> (bs, 3 * 8, h, w)
        img = F.interpolate(
            img, size=(self.image_size, self.image_size), mode="bilinear"
        )
        bs, _, h, w = img.shape
        img = img.reshape(bs, 3, 8, h, w)
        assert 4 - self.seq_len // 2 > 0
        img = img[:, :, 4 - self.seq_len // 2 : 4 - self.seq_len // 2 + self.seq_len]
        if self.training and np.random.rand() < 0.5:
            img = img.flip(2)
        img = img.permute((0, 2, 1, 3, 4)).reshape(bs * self.seq_len, 3, h, w)
        img_feats = self.encoder(img)
        if self.output_fmt == "NHWC":
            img_feats = [
                img_feat.permute(0, 3, 1, 2).contiguous() for img_feat in img_feats
            ]
        for i in range(len(img_feats)):
            img_feat = img_feats[i]
            _, ch, h, w = img_feat.shape  # (bs * seq_len, ch, h, w)
            img_feat = img_feat.reshape(bs, self.seq_len, ch, h, w).transpose(
                1, 2
            )  # (bs, ch, seq_len, h, w)
            img_feats[i] = self.conv3d[i](img_feat)[
                :, :, self.seq_len // 2
            ]  # (bs, ch, seq_len, h, w) -> (bs, ch, h, w)
        return img_feats

    def forward_head(self, img_feats):
        output = self.head(img_feats)
        return F.interpolate(output, size=(256, 256), mode="bilinear")

    def forward(
        self,
        img: torch.Tensor,
    ):
        """
        img: (bs, ch, h, w)
        """
        img_feats = self.forward_image_feats(img)
        return self.forward_head(img_feats)


class Mixup(object):
    def __init__(self, p: float = 0.5, alpha: float = 0.5):
        self.p = p
        self.alpha = alpha
        self.lam = 1.0
        self.do_mixup = False

    def init_lambda(self):
        if np.random.rand() < self.p:
            self.do_mixup = True
        else:
            self.do_mixup = False
        if self.do_mixup and self.alpha > 0.0:
            self.lam = np.random.beta(self.alpha, self.alpha)
        else:
            self.lam = 1.0

    def reset_lambda(self):
        self.lam = 1.0


def dice_score(targets: np.ndarray, preds: np.ndarray, smooth: float = 1e-6):
    if targets.sum() == 0 and preds.sum() == 0:
        return 1.0
    y_true_count = targets.sum()
    ctp = (preds * targets).sum()
    cfp = (preds * (1 - targets)).sum()

    c_precision = ctp / (ctp + cfp + smooth)
    c_recall = ctp / (y_true_count + smooth)
    dice = 2 * (c_precision * c_recall) / (c_precision + c_recall + smooth)

    return dice


def func_percentile(y_true: np.ndarray, y_pred: np.ndarray, t: List[float]):
    score = dice_score(
        y_true,
        (
            y_pred
            > np.quantile(
                y_pred,
                np.clip(t[0], 0, 1),
            )
        ).astype(int),
    )
    return -score


def find_threshold_percentile(y_true: np.ndarray, y_pred: np.ndarray):
    x0 = [0.5]
    threshold = minimize(
        partial(
            func_percentile,
            y_true,
            y_pred,
        ),
        x0,
        method="nelder-mead",
    ).x[0]
    return np.clip(threshold, 0, 1)


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


class GlobalDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.smooth = 1e-6

    def forward(self, preds, targets):
        preds = torch.sigmoid(preds)
        y_true_count = targets.sum()
        ctp = (preds * targets).sum()
        cfp = (preds * (1 - targets)).sum()

        c_precision = ctp / (ctp + cfp + self.smooth)
        c_recall = ctp / (y_true_count + self.smooth)
        dice = 2 * (c_precision * c_recall) / (c_precision + c_recall + self.smooth)
        return 1 - dice


class ContrailsLightningSegModel2_5D(pl.LightningModule):
    def __init__(
        self,
        model_name: str = "resnet34",
        pretrained: bool = False,
        drop_path_rate: float = 0,
        decoder_type: str = "UNet",  # UNet or FastFCNImprove
        center=None,
        attention_type=None,
        image_size: int = 256,
        in_chans: int = 3,
        num_class: int = 1,
        seq_len: int = 7,
        mixup_p: float = 0.0,
        mixup_alpha: float = 0.5,
        no_mixup_epochs: int = 0,
        dice_ratio: float = 0.25,
        lr: float = 1e-3,
        backbone_lr: float = None,
        disable_compile: bool = False,
    ) -> None:
        super().__init__()
        self.lr = lr
        self.backbone_lr = backbone_lr if backbone_lr is not None else lr
        self.dice_ratio = dice_ratio
        self.__build_model(
            model_name=model_name,
            pretrained=pretrained,
            drop_path_rate=drop_path_rate,
            decoder_type=decoder_type,
            center=center,
            attention_type=attention_type,
            in_chans=in_chans,
            num_class=num_class,
            image_size=image_size,
            seq_len=seq_len,
        )
        if not disable_compile:
            self.__compile_model()
        self.mixupper = Mixup(p=mixup_p, alpha=mixup_alpha)
        self.mixup_alpha = mixup_alpha
        self.no_mixup_epochs = no_mixup_epochs
        self.gt_val = []
        self.logit_val = []
        self.save_hyperparameters()

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
        seq_len: int = 7,
    ):
        self.model = ContrailsSegModel2_5D(
            model_name=model_name,
            pretrained=pretrained,
            drop_path_rate=drop_path_rate,
            decoder_type=decoder_type,
            center=center,
            attention_type=attention_type,
            in_chans=in_chans,
            num_class=num_class,
            image_size=image_size,
            seq_len=seq_len,
        )
        self.model_ema = ModelEmaV2(self.model, decay=0.998)
        self.criterions = {
            "bce": nn.BCEWithLogitsLoss(),
            "dice": GlobalDiceLoss(),
        }

    def __compile_model(self):
        self.model = torch.compile(self.model)
        self.model_ema = torch.compile(self.model_ema)

    def calc_loss(self, outputs: torch.Tensor, labels: torch.Tensor):
        losses = {}
        # smooth = 0.1
        # true = labels["targets"] * (1 - (smooth / 0.5)) + smooth

        losses["bce"] = self.criterions["bce"](
            outputs["logits"],
            labels["targets"].to(dtype=outputs["logits"].dtype),
        )
        losses["dice"] = self.criterions["dice"](
            outputs["logits"],
            labels["targets"].to(dtype=outputs["logits"].dtype),
        )
        losses["loss"] = losses["bce"] + self.dice_ratio * losses["dice"]
        return losses

    def training_step(self, batch, batch_idx):
        self.model_ema.update(self.model)
        step_output = {}
        outputs = {}
        loss_target = {}
        self.mixupper.init_lambda()
        image, label = batch
        if (
            self.mixupper.do_mixup
            and self.current_epoch < self.trainer.max_epochs - self.no_mixup_epochs
        ):
            image = self.mixupper.lam * image + (1 - self.mixupper.lam) * image.flip(0)
        else:
            # if (
            #     np.random.rand() < 0.5
            #     and self.current_epoch < self.trainer.max_epochs - self.no_mixup_epochs
            # ):
            #     lam = (
            #         np.random.beta(self.mixup_alpha, self.mixup_alpha)
            #         if self.mixup_alpha > 0
            #         else 0
            #     )
            #     bbx1, bby1, bbx2, bby2 = rand_bbox(volume.size(), lam)
            #     volume[:, :, bbx1:bbx2, bby1:bby2] = volume.flip(0)[
            #         :, :, bbx1:bbx2, bby1:bby2
            #     ]
            #     label[:, :, bbx1:bbx2, bby1:bby2] = label.flip(0)[
            #         :, :, bbx1:bbx2, bby1:bby2
            #     ]
            pass
        outputs["logits"] = self.model(image)

        loss_target["targets"] = label
        losses = self.calc_loss(outputs, loss_target)

        if (
            self.mixupper.do_mixup
            and self.current_epoch < self.trainer.max_epochs - self.no_mixup_epochs
        ):
            loss_target["targets"] = loss_target["targets"].flip(0)
            losses_b = self.calc_loss(outputs, loss_target)
            for key in losses:
                losses[key] = (
                    self.mixupper.lam * losses[key]
                    + (1 - self.mixupper.lam) * losses_b[key]
                )
        step_output.update(losses)
        self.log_dict(
            dict(
                train_loss=losses["loss"],
                train_bce_loss=losses["bce"],
                train_dice_loss=losses["dice"],
            )
        )
        return step_output

    def validation_step(self, batch, batch_idx):
        step_output = {}
        outputs = {}
        loss_target = {}

        image, label = batch
        outputs["logits"] = self.model_ema.module(image)

        loss_target["targets"] = label
        losses = self.calc_loss(outputs, loss_target)
        # if torch.isnan(losses["loss"]):
        #     print("Rollback previous model weight!")
        #     self.model_ema.set(self.model)
        #     image, label = batch
        #     outputs["logits"] = self.model_ema.module(image)
        #     loss_target["targets"] = label
        #     losses = self.calc_loss(outputs, loss_target)
        step_output.update(losses)
        self.logit_val.append(torch.sigmoid(outputs["logits"]).detach().cpu().numpy())
        self.gt_val.append(label.detach().cpu().numpy() > 0.5)

        self.log_dict(
            dict(
                val_loss=losses["loss"],
                val_bce_loss=losses["bce"],
                val_dice_loss=losses["dice"],
            )
        )
        return step_output

    def on_validation_epoch_end(self):
        logit_val = np.concatenate(self.logit_val)  # (len, c, h, w)
        gt_val = np.concatenate(self.gt_val)  # (len, c, h, w)
        threshold = find_threshold_percentile(gt_val, logit_val)
        pred_val = logit_val > np.quantile(logit_val, threshold)
        pred_val_05 = logit_val > 0.5

        score = dice_score(gt_val, pred_val)
        score_05 = dice_score(gt_val, pred_val_05)

        self.logit_val.clear()
        self.gt_val.clear()

        self.log_dict(
            dict(
                val_dice_score=score,
                val_threshold=threshold,
                val_dice05_score=score_05,
            ),
            sync_dist=True,
        )

    def get_optimizer_parameters(self):
        no_decay = ["bias", "gamma", "beta"]
        optimizer_parameters = [
            {
                "params": [
                    p
                    for n, p in self.model.encoder.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0,
                "lr": self.backbone_lr,
            },
            {
                "params": [
                    p
                    for n, p in self.model.encoder.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0001,
                "lr": self.backbone_lr,
            },
            {
                "params": [
                    p
                    for n, p in list(self.model.conv3d.named_parameters())
                    + list(self.model.head.named_parameters())
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0,
                "lr": self.lr,
            },
            {
                "params": [
                    p
                    for n, p in list(self.model.conv3d.named_parameters())
                    + list(self.model.head.named_parameters())
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0001,
                "lr": self.lr,
            },
        ]
        return optimizer_parameters

    def configure_optimizers(self):
        self.warmup = True
        optimizer = AdamW(self.get_optimizer_parameters())
        max_train_steps = self.trainer.estimated_stepping_batches
        warmup_steps = math.ceil((max_train_steps * 2) / 100) if self.warmup else 0
        rank_zero_info(
            f"max_train_steps: {max_train_steps}, warmup_steps: {warmup_steps}"
        )
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=max_train_steps,
        )
        scheduler = {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1,
        }
        return [optimizer], [scheduler]

    @staticmethod
    def add_model_specific_args(
        parent_parser: argparse.ArgumentParser,
    ) -> argparse.ArgumentParser:
        parser = parent_parser.add_argument_group("ContrailsLightningSegModel2_5D")
        parser.add_argument(
            "--model_name",
            default="resnet34",
            type=str,
            metavar="MN",
            help="Name (as in ``timm``) of the feature extractor",
            dest="model_name",
        )
        parser.add_argument(
            "--drop_path_rate",
            default=None,
            type=float,
            metavar="DPR",
            dest="drop_path_rate",
        )
        parser.add_argument(
            "--decoder_type",
            default="UNet",
            type=str,
            choices=["UNet", "FastFCNImprove"],
            metavar="DT",
            help="Name of the decoder_type, implemented: UNet|FastFCNImproved",
            dest="decoder_type",
        )
        parser.add_argument(
            "--center",
            default=None,
            type=str,
            choices=[None, "fpa", "aspp"],
            metavar="CT",
            help="Name of the center module, implemented: None|fpa|aspp",
            dest="center",
        )
        parser.add_argument(
            "--attention_type",
            default=None,
            type=str,
            choices=[None, "scse", "cbam"],
            metavar="AT",
            help="Name of the attention module, implemented: None|scse|cbam",
            dest="attention_type",
        )
        parser.add_argument(
            "--image_size",
            default=256,
            type=int,
            metavar="IS",
            dest="image_size",
        )
        parser.add_argument(
            "--in_chans",
            default=3,
            type=int,
            metavar="ICH",
            dest="in_chans",
        )
        parser.add_argument(
            "--num_class",
            default=1,
            type=int,
            metavar="OCL",
            dest="num_class",
        )
        parser.add_argument(
            "--seq_len",
            default=7,
            type=int,
            metavar="SL",
            dest="seq_len",
        )
        parser.add_argument(
            "--dice_ratio",
            default=0.25,
            type=float,
            metavar="DR",
            dest="dice_ratio",
        )
        parser.add_argument(
            "--mixup_p", default=0.0, type=float, metavar="MP", dest="mixup_p"
        )
        parser.add_argument(
            "--mixup_alpha", default=0.0, type=float, metavar="MA", dest="mixup_alpha"
        )
        parser.add_argument(
            "--no_mixup_epochs",
            default=0,
            type=int,
            metavar="NME",
            dest="no_mixup_epochs",
        )
        parser.add_argument(
            "--lr",
            default=1e-3,
            type=float,
            metavar="LR",
            help="initial learning rate",
            dest="lr",
        )

        parser.add_argument(
            "--backbone_lr",
            default=None,
            type=float,
            metavar="LR",
            help="initial learning rate",
            dest="backbone_lr",
        )
        parent_parser.add_argument(
            "--disable_compile",
            action="store_true",
            help="disable torch.compile",
            dest="disable_compile",
        )

        return parent_parser


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
        "--debug",
        action="store_true",
        help="1 batch run for debug",
        dest="debug",
    )
    dt_now = datetime.datetime.now()
    parent_parser.add_argument(
        "--logdir",
        default=f"{dt_now.strftime('%Y%m%d-%H-%M-%S')}",
    )
    parent_parser.add_argument(
        "--fold",
        type=int,
        default=0,
    )
    parent_parser.add_argument(
        "--gpus", type=int, default=4, help="number of gpus to use"
    )
    parent_parser.add_argument(
        "--epochs", default=10, type=int, metavar="N", help="total number of epochs"
    )
    parser = ContrailsLightningSegModel2_5D.add_model_specific_args(parent_parser)
    parser = ContrailsDataModule.add_model_specific_args(parser)
    return parser.parse_args()


def main(args):
    pl.seed_everything(args.seed, workers=True)
    if not args.debug:
        warnings.simplefilter("ignore")
    train_df = pd.read_csv(f"{BASE_DIR}/train_metadata.csv")
    valid_df = pd.read_csv(f"{BASE_DIR}/validation_metadata.csv")
    train_df = train_df[train_df.mask_sum > 0]
    valid_df = valid_df[valid_df.mask_sum > 0]
    datamodule = ContrailsDataModule(
        train_df=train_df,
        valid_df=valid_df,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
    )
    model = ContrailsLightningSegModel2_5D(
        model_name=args.model_name,
        pretrained=True,
        drop_path_rate=args.drop_path_rate,
        decoder_type=args.decoder_type,
        center=args.center,
        attention_type=args.attention_type,
        image_size=args.image_size,
        in_chans=args.in_chans,
        num_class=args.num_class,
        seq_len=args.seq_len,
        dice_ratio=args.dice_ratio,
        mixup_p=args.mixup_p,
        mixup_alpha=args.mixup_alpha,
        no_mixup_epochs=args.no_mixup_epochs,
        lr=args.lr,
        backbone_lr=args.backbone_lr,
        disable_compile=args.disable_compile,
    )

    logdir = f"../../logs/exp{EXP_ID}/{args.logdir}/fold0"
    print(f"logdir = {logdir}")
    lr_monitor = callbacks.LearningRateMonitor()
    loss_checkpoint = callbacks.ModelCheckpoint(
        filename="best_loss",
        monitor="val_loss",
        save_top_k=1,
        save_last=True,
        save_weights_only=True,
        mode="min",
    )
    dice_checkpoint = callbacks.ModelCheckpoint(
        filename="best_dice",
        monitor="val_dice_score",
        save_last=False,
        save_weights_only=True,
        mode="max",
    )
    dice05_checkpoint = callbacks.ModelCheckpoint(
        filename="best_dice05",
        monitor="val_dice05_score",
        save_last=False,
        save_weights_only=True,
        mode="max",
    )
    os.makedirs(os.path.join(logdir, "wandb"), exist_ok=True)
    if not args.debug:
        wandb_logger = WandbLogger(
            name=f"exp{EXP_ID}/{args.logdir}_fold0",
            save_dir=logdir,
            project="google-research-identify-contrails-reduce-global-warming",
        )

    trainer = pl.Trainer(
        default_root_dir=logdir,
        sync_batchnorm=True,
        gradient_clip_val=2.0,
        precision="16-mixed",
        devices=args.gpus,
        accelerator="gpu",
        strategy="ddp_find_unused_parameters_true",
        # strategy="ddp",
        max_epochs=args.epochs,
        logger=wandb_logger if not args.debug else True,
        callbacks=[
            loss_checkpoint,
            dice_checkpoint,
            dice05_checkpoint,
            lr_monitor,
        ],
        fast_dev_run=args.debug,
        num_sanity_val_steps=0,
        accumulate_grad_batches=16 // args.batch_size,
    )
    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    main(get_args())
