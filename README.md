# Iris Abnormality Segmentation

Binary semantic segmentation of pathology-associated abnormal regions in iris images using U-Net.

## Target definition

The network uses a single foreground class:

- `0`: background / non-abnormal
- `1`: abnormal

The dataset is composed of:

- **Geometry**: the annotated Geometry region is treated as `abnormal`.
- **Tissue**: the annotated pathological tissue region is treated as `abnormal`.
- **Healthy**: no abnormal region; the ground-truth mask is completely empty.

Geometry and Tissue therefore share the same output label (`abnormal`) in the segmentation model.

## Important split rule

Images such as:

```text
0011_R_IG_1_1.png
0011_R_IG_2_1.png
```

are treated as images from the same subject (`0011`).

All images from the same subject are assigned to exactly one of:

- train
- validation
- test

This prevents subject leakage.

## Raw data structure

```text
raw_dataset/
├── Geometry/
│   ├── images/
│   └── masks/
├── Tissue/
│   ├── images/
│   └── masks/
└── Healthy/
    └── images/
```

Geometry and Tissue masks are exported from CVAT.
Healthy masks are created automatically by `prepare_dataset.py`.

## Prepared data structure

```text
prepared_dataset/
├── Geometry/
│   ├── train/
│   │   ├── images/
│   │   └── masks/
│   ├── val/
│   │   ├── images/
│   │   └── masks/
│   └── test/
│       ├── images/
│       └── masks/
├── Tissue/
│   ├── train/
│   ├── val/
│   └── test/
└── Healthy/
    ├── train/
    ├── val/
    └── test/
```

No metadata CSV is used.

## Installation

```bash
pip install -r requirements.txt
```

## Configuration

Edit:

```text
configs/unet.yaml
```

and specify the raw dataset paths and output paths.

## Prepare dataset

```bash
python scripts/prepare_dataset.py \
    --config configs/unet.yaml
```

## Check dataset

```bash
python scripts/check_dataset.py \
    --config configs/unet.yaml
```

## Train

```bash
python scripts/train_unet.py \
    --config configs/unet.yaml
```

## Evaluate

```bash
python scripts/evaluate.py \
    --config configs/unet.yaml
```

Evaluation is reported overall and separately for:

- Geometry
- Tissue
- Healthy

For Healthy images, false-positive rate is especially useful because the correct mask is empty.

## Predict one image

```bash
python scripts/predict.py \
    --config configs/unet.yaml \
    --image /path/to/image.png
```

The predicted mask and overlay are saved under the configured prediction directory.
