import os

import numpy as np
import pandas as pd
from tqdm.auto import trange

TRAIN_DIR = "../input/google-research-identify-contrails-reduce-global-warming/train"
VALIDATION_DIR = (
    "../input/google-research-identify-contrails-reduce-global-warming/validation"
)
ash_dataset_dir = "../input/ash_dataset"

data_types = {"record_id": str}
train_meta = pd.read_json(
    "../input/google-research-identify-contrails-reduce-global-warming/train_metadata.json",
    dtype=data_types,
)
validation_meta = pd.read_json(
    "../input/google-research-identify-contrails-reduce-global-warming/validation_metadata.json",
    dtype=data_types,
)


def train_get_image(base_dir, meta, i):
    # print(record_ids[i])
    with open(os.path.join(base_dir, meta.record_id.iloc[i], "band_11.npy"), "rb") as f:
        band11 = np.load(f)
    with open(os.path.join(base_dir, meta.record_id.iloc[i], "band_14.npy"), "rb") as f:
        band14 = np.load(f)
    with open(os.path.join(base_dir, meta.record_id.iloc[i], "band_15.npy"), "rb") as f:
        band15 = np.load(f)
    with open(
        os.path.join(base_dir, meta.record_id.iloc[i], "human_pixel_masks.npy"), "rb"
    ) as f:
        human_pixel_mask = np.load(f)
    with open(
        os.path.join(base_dir, meta.record_id.iloc[i], "human_individual_masks.npy"),
        "rb",
    ) as f:
        human_individual_mask = np.load(f)

    _T11_BOUNDS = (243, 303)
    _CLOUD_TOP_TDIFF_BOUNDS = (-4, 5)
    _TDIFF_BOUNDS = (-4, 2)

    def normalize_range(data, bounds):
        """Maps data to the range [0, 1]."""
        return (data - bounds[0]) / (bounds[1] - bounds[0])
        # return data

    r = normalize_range(band15 - band14, _TDIFF_BOUNDS)
    g = normalize_range(band14 - band11, _CLOUD_TOP_TDIFF_BOUNDS)
    b = normalize_range(band14, _T11_BOUNDS)
    false_color = np.clip(np.stack([r, g, b], axis=2), 0, 1)
    return (
        false_color.astype(np.float16),
        human_pixel_mask > 0.5,
        human_individual_mask > 0.5,
    )


def val_get_image(base_dir, meta, i):
    with open(os.path.join(base_dir, meta.record_id.iloc[i], "band_11.npy"), "rb") as f:
        band11 = np.load(f)
    with open(os.path.join(base_dir, meta.record_id.iloc[i], "band_14.npy"), "rb") as f:
        band14 = np.load(f)
    with open(os.path.join(base_dir, meta.record_id.iloc[i], "band_15.npy"), "rb") as f:
        band15 = np.load(f)
    with open(
        os.path.join(base_dir, meta.record_id.iloc[i], "human_pixel_masks.npy"), "rb"
    ) as f:
        human_pixel_mask = np.load(f)

    _T11_BOUNDS = (243, 303)
    _CLOUD_TOP_TDIFF_BOUNDS = (-4, 5)
    _TDIFF_BOUNDS = (-4, 2)

    def normalize_range(data, bounds):
        """Maps data to the range [0, 1]."""
        return (data - bounds[0]) / (bounds[1] - bounds[0])

    r = normalize_range(band15 - band14, _TDIFF_BOUNDS)
    g = normalize_range(band14 - band11, _CLOUD_TOP_TDIFF_BOUNDS)
    b = normalize_range(band14, _T11_BOUNDS)
    false_color = np.clip(np.stack([r, g, b], axis=2), 0, 1)
    return (
        false_color.astype(np.float16),
        human_pixel_mask > 0.5,
    )


mask_sum = []
ash_data_path = []
for i in trange(len(train_meta)):
    img, mask, mask_individual = train_get_image(TRAIN_DIR, train_meta, i)
    out_dir = os.path.join(ash_dataset_dir, "train", train_meta.record_id.iloc[i])
    os.makedirs(out_dir, exist_ok=True)
    mask_sum.append(mask.sum())
    ash_data_path.append(os.path.abspath(out_dir))
    np.save(os.path.join(out_dir, "img"), img)
    np.save(os.path.join(out_dir, "mask"), mask)
    np.save(os.path.join(out_dir, "mask_individual"), mask_individual)

train_meta["mask_sum"] = mask_sum
train_meta["ash_data_path"] = ash_data_path
train_meta.to_csv(os.path.join(ash_dataset_dir, "train_metadata.csv"), index=False)

mask_sum = []
ash_data_path = []
for i in trange(len(validation_meta)):
    img, mask = val_get_image(VALIDATION_DIR, validation_meta, i)
    out_dir = os.path.join(
        ash_dataset_dir, "validation", validation_meta.record_id.iloc[i]
    )
    os.makedirs(out_dir, exist_ok=True)
    mask_sum.append(mask.sum())
    ash_data_path.append(os.path.abspath(out_dir))
    np.save(os.path.join(out_dir, "img"), img)
    np.save(os.path.join(out_dir, "mask"), mask)

validation_meta["mask_sum"] = mask_sum
validation_meta["ash_data_path"] = ash_data_path
validation_meta.to_csv(
    os.path.join(ash_dataset_dir, "validation_metadata.csv"), index=False
)
