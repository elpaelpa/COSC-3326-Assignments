# Deepfake Detection

COSC 3326 (Introduction to Image Processing and Computer Vision), Group 4 project.

Classifies cropped faces as real or fake by fine-tuning a pretrained ResNet-18 on the
[OpenForensics](https://zenodo.org/records/5528418) dataset (Le et al., ICCV 2021, CC BY 4.0).

## Results

Test split: 2,280 held-out faces (718 real, 1,562 fake).

| Model | Accuracy | ROC AUC |
|---|---|---|
| Majority class (always "fake") | 68.51% | 0.5 |
| ResNet-18, ImageNet-pretrained | 98.99% | 0.9994 |
| ResNet-18, from scratch (baseline) | 97.15% | 0.9954 |

Train, validation and test all come from the OpenForensics Val set (split by image, 70/15/15),
so these numbers show in-dataset performance only. Each model's metrics and figures are in
its own folder under `results/`.

## Layout

```
notebooks/train_deepfake_colab.ipynb   download data, crop faces, train, evaluate (run on Colab)
notebooks/baseline_scratch_colab.ipynb same pipeline, ResNet-18 trained from scratch (baseline)
scripts/download_openforensics.py      standalone dataset downloader (local use)
results/resnet18_pretrained/           metrics and figures for the main model
results/resnet18_scratch/              metrics for the from-scratch baseline
docs/                                  proposal and results slides, OpenForensics paper
models/                                trained weights, best.pt (not in git)
```

## Run it

1. Open `notebooks/train_deepfake_colab.ipynb` in Google Colab (or the VS Code Colab extension).
2. Select a GPU runtime and run the cells in order. The notebook downloads the Val set (about 3 GB),
   trains for 5 epochs and writes metrics and figures to `/content/results/<model>/`.
3. Copy that folder and the checkpoint (`best.pt` or `baseline_scratch.pt`) somewhere permanent
   (e.g. Google Drive) before the runtime disconnects, because Colab's disk is erased. Put the
   files in the matching `results/<model>/` folder here.

To download the dataset locally instead:

```
pip install -r requirements.txt
python scripts/download_openforensics.py --data-dir data/OpenForensics
```
