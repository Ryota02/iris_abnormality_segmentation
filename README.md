# Iris Abnormality Segmentation

This repository contains experiments for binary segmentation of abnormal iris regions using a ResNet34-based U-Net.

The current experiments focus on **Tissue** and **Healthy** iris images and evaluate:

- Baseline training
- Individual image augmentation methods
- Oversampling
- Augmentation + oversampling
- Synthetic Tissue data
- Synthetic-only training
- Quantitative result reports
- Qualitative prediction reports

---

## Requirements

Install the required Python packages:

```bash
pip install torch torchvision opencv-python numpy pandas matplotlib pillow tqdm pyyaml reportlab
```

Check whether CUDA is available:

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

Run all commands from the project root.

---

## Data Assumptions

The task uses binary segmentation masks:

```text
0   = background / non-abnormal region
255 = abnormal region
```

Healthy images use all-zero masks.

Synthetic images, augmentation, and oversampling must be used **only during training**.

Validation and test sets must contain only real images.

If multiple images belong to the same subject, all images from that subject must remain in the same split to avoid subject leakage.

---

## Model

The default model is a ResNet34-based U-Net:

```yaml
model:
  name: unet_resnet
  backbone: resnet34
  pretrained: true
  in_channels: 1
  out_channels: 1
```

Typical training settings are:

```yaml
training:
  image_size: 512
  epochs: 100
  batch_size: 8
  learning_rate: 0.0001
  weight_decay: 0.0001
  num_workers: 0
  threshold: 0.5
```

The current segmentation loss is:

```text
0.5 × BCEWithLogitsLoss + 0.5 × DiceLoss
```

---

# 1. Baseline Training

Run:

```bash
python scripts/train_unet.py     --config configs/unet_resnet34.yaml
```

This trains the model without synthetic data, augmentation, or oversampling unless explicitly enabled in the configuration.

---

# 2. Augmentation and Oversampling Ablation

This experiment compares:

- Baseline
- One augmentation method at a time
- Oversampling only
- One augmentation method + oversampling

The evaluated augmentation methods are:

```text
horizontal_flip
rotation
brightness
contrast
gamma
gaussian_noise
gaussian_blur
```

Only one augmentation method is enabled in each augmentation experiment.

The full set of conditions is:

```text
baseline

horizontal_flip
rotation
brightness
contrast
gamma
gaussian_noise
gaussian_blur

oversampling

horizontal_flip_oversampling
rotation_oversampling
brightness_oversampling
contrast_oversampling
gamma_oversampling
gaussian_noise_oversampling
gaussian_blur_oversampling
```

Run all conditions with:

```bash
python scripts/run_augmentation_ablation.py     --config configs/augmentation_ablation.yaml
```

Oversampling is controlled by the category multiplier in the configuration. For example:

```yaml
sampling:
  enabled: false

  category_multiplier:
    Healthy: 1
    Tissue: 6
```

If the training set contains 5 Tissue images and 32 Healthy images, this produces approximately:

```text
Tissue  : 5 × 6 = 30
Healthy : 32 × 1 = 32
```

Oversampling affects only the training set.

---

# 3. Augmentation Ablation Report

After the augmentation experiment finishes, create the quantitative report with:

```bash
python scripts/create_augmentation_report.py     --config configs/augmentation_ablation.yaml
```

The report summarizes:

- Validation Dice and IoU
- Test Dice and IoU
- Tissue-specific Dice and IoU
- Baseline vs oversampling
- Augmentation-only vs augmentation + oversampling
- Results across all augmentation methods

Tissue-specific metrics should be inspected separately because Healthy masks contain no abnormal pixels.

---

# 4. Synthetic Tissue Ablation

Synthetic Tissue images are added only to the training set.

A typical synthetic experiment configuration includes:

```yaml
synthetic:
  enabled: false
  use_num: 0
  shuffle_seed: 42
  include_real_tissue: true
```

A fixed `shuffle_seed` ensures that smaller synthetic subsets are reproducible and nested within larger subsets.

Example experimental conditions:

```yaml
synthetic_ablation:
  experiments:

    - name: baseline
      synthetic_count: 0
      include_real_tissue: true

    - name: synthetic_10
      synthetic_count: 10
      include_real_tissue: true

    - name: synthetic_20
      synthetic_count: 20
      include_real_tissue: true

    - name: synthetic_balanced
      synthetic_count: 27
      include_real_tissue: true

    - name: synthetic_31
      synthetic_count: 31
      include_real_tissue: true

    - name: synthetic_only_31
      synthetic_count: 31
      include_real_tissue: false
```

`synthetic_balanced` should be chosen so that:

```text
Real Tissue + Synthetic Tissue ≈ Real Healthy
```

For example:

```text
Real Tissue      = 5
Synthetic Tissue = 27
Healthy          = 32
```

Run all synthetic experiments with:

```bash
python scripts/run_synthetic_ablation.py     --config configs/synthetic_ablation.yaml
```

For the synthetic ablation, augmentation and oversampling should normally remain disabled so that the effect of synthetic data can be evaluated independently.

---

# 5. Synthetic Ablation Report

After all synthetic experiments finish, create the quantitative PDF report with:

```bash
python scripts/create_synthetic_report.py     --config configs/synthetic_ablation.yaml
```

The main metrics are:

```text
best_val_dice
best_val_iou
test_dice
test_iou
tissue_test_dice
tissue_test_iou
```

Validation metrics should be used for checkpoint and experiment selection.

Test metrics should be treated as final evaluation results and should not be used to choose the best training condition.

---

# 6. Prediction Visualization

Each prediction can be visualized using five images:

```text
Original
Ground Truth
Prediction
Segmentation Error Map
Probability Heat Map
```

The segmentation error map uses:

```text
Green = True Positive
Blue  = False Positive
Red   = False Negative
Black = True Negative
```

The probability heat map represents the pixel-wise sigmoid probability of the abnormal class.

It is **not Grad-CAM**.

---

# 7. Prediction for One Synthetic Experiment

Example: baseline

```bash
python scripts/predict.py     --config configs/synthetic_ablation.yaml     --experiment baseline     --split test
```

Example: synthetic_20

```bash
python scripts/predict.py     --config configs/synthetic_ablation.yaml     --experiment synthetic_20     --split test
```

---

# 8. Prediction for All Synthetic Experiments

Run prediction for all synthetic-ablation checkpoints with:

```bash
python scripts/predict.py     --config configs/synthetic_ablation.yaml     --all-experiments     --split test
```

Each experiment stores its prediction results separately.

---

# 9. Combined Synthetic Prediction PDF

After generating predictions for all synthetic experiments, create one combined qualitative PDF with:

```bash
python scripts/create_synthetic_prediction_report.py     --config configs/synthetic_ablation.yaml     --split test
```

The PDF compares the five visualization panels for each test case across the selected synthetic experiments.

---

# 10. Recommended Experiment Order

Run the experiments in the following order.

### Step 1: Baseline

```bash
python scripts/train_unet.py     --config configs/unet_resnet34.yaml
```

### Step 2: Augmentation / Oversampling Ablation

```bash
python scripts/run_augmentation_ablation.py     --config configs/augmentation_ablation.yaml
```

### Step 3: Augmentation Report

```bash
python scripts/create_augmentation_report.py     --config configs/augmentation_ablation.yaml
```

### Step 4: Synthetic Ablation

```bash
python scripts/run_synthetic_ablation.py     --config configs/synthetic_ablation.yaml
```

### Step 5: Synthetic Report

```bash
python scripts/create_synthetic_report.py     --config configs/synthetic_ablation.yaml
```

### Step 6: Generate Predictions for All Synthetic Conditions

```bash
python scripts/predict.py     --config configs/synthetic_ablation.yaml     --all-experiments     --split test
```

### Step 7: Create the Combined Prediction PDF

```bash
python scripts/create_synthetic_prediction_report.py     --config configs/synthetic_ablation.yaml     --split test
```

---

# 11. Experimental Rules

Use the following protocol for all comparisons:

```text
Training:
- Real Tissue
- Real Healthy
- Synthetic Tissue, if enabled
- Augmentation, if enabled
- Oversampling, if enabled

Validation:
- Real Tissue only
- Real Healthy only

Test:
- Real Tissue only
- Real Healthy only
```

Keep the same train/validation/test split across all ablation conditions.

For very small Tissue datasets, subject-level cross-validation is preferable to relying on a validation set containing only one Tissue image.