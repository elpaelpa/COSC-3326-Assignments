# Deepfake Detection

COSC 3326 (Introduction to Image Processing and Computer Vision), Group 4 project.

Classifies cropped faces as real or fake by fine-tuning a pretrained ResNet-18 on the
[OpenForensics](https://zenodo.org/records/5528418) dataset (Le et al., ICCV 2021, CC BY 4.0).

## Results

Test split: 2,280 held-out faces (718 real, 1,562 fake).

| Metric | Value |
|---|---|
| Accuracy | 98.99% |
| ROC AUC | 0.9994 |

Train, validation and test all come from the OpenForensics Val set (split by image, 70/15/15),
so these numbers show in-dataset performance only. Details: `results/metrics.txt`.

## Layout

```
notebooks/train_deepfake_colab.ipynb   download data, crop faces, train, evaluate (run on Colab)
scripts/download_openforensics.py      standalone dataset downloader (local use)
results/                               metrics and figures from the notebook
docs/                                  proposal and results slides
models/                                trained weights, best.pt (not in git)
```

## Run it

1. Open `notebooks/train_deepfake_colab.ipynb` in Google Colab (or the VS Code Colab extension).
2. Select a GPU runtime and run the cells in order. The notebook downloads the Val set (about 3 GB),
   trains for 5 epochs and writes metrics and figures to `/content/results`.
3. Copy `/content/results` and `best.pt` somewhere permanent (e.g. Google Drive) before the runtime
   disconnects, because Colab's disk is erased.

To download the dataset locally instead:

```
pip install -r requirements.txt
python scripts/download_openforensics.py --data-dir data/OpenForensics
```
