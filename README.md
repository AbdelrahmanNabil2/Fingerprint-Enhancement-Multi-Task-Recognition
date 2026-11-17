# Fingerprint Recognition and Gender Classification

A multi-task convolutional network that reads a fingerprint image and predicts the person, the gender, and the finger.

Preprocessing was done separately, before training. This repository contains the training code, the desktop app, and the result figures. The trained weights file is not included.

## What is included

| Path | Purpose |
|---|---|
| `final_code.py` | Desktop app. Upload a print, apply filters, then classify it. |
| `model3.ipynb` | Training and evaluation. It saves a new weights file when training finishes. |
| `metrics/` | Accuracy curves, confusion matrices, and sample predictions from the trained run. |

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the app

From this folder:

```bash
python final_code.py
```

Classification looks for `model/fingerprint_model_final_20250509_1311.h5` next to the script. That file is not in this folder. Train with `model3.ipynb`, then place the saved weights on that path before using **Test Fingerprint Model**. The filters still run without it.

## Dataset

The images are not in this repository. Training used the Sokoto Coventry Fingerprint Dataset (SOCOFing): 600 people, 477 male and 123 female, ten fingers each.

- **Real prints:** 6,000 images. A file is named `100__M_Left_index_finger.BMP`.
- **Easy alterations:** 17,931 images. Each real print has up to three altered copies: central rotation (`CR`), obliteration (`Obl`), and Z-cut (`Zcut`). A file is named `100__M_Left_index_finger_CR.BMP`.
- Most images are 96×103 pixels. The filename supplies the person id, the gender (`M` or `F`), and the finger.

Those training images were preprocessed in a separate step, outside this code: contrast stretch, histogram equalization, sharpen, and segmentation. The notebook only opens the finished images, resizes them to 128×128, and scales pixels to the range 0–1.

## Train again

Point `train_folder` and `test_folder` at the start of `model3.ipynb` to your local image folders, then run the notebook. The notebook writes a new `.h5` file when it finishes.
