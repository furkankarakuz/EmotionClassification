# TurkishEmotionClassification

<p align="center">
  <img src="https://github.com/user-attachments/assets/f98ff5da-8f77-4f83-ac27-d063377171e0"/>
</p>

Fine-tuned BERTurk model for **Turkish emotion classification** (6 emotions) — training notebook, reusable source modules, and a Gradio app.

🤗 **Model:** [furkankarakuz/turkish-emotion-classifier](https://huggingface.co/furkankarakuz/turkish-emotion-classifier)

## Overview

This project fine-tunes [`dbmdz/bert-base-turkish-cased`](https://huggingface.co/dbmdz/bert-base-turkish-cased)
to classify short Turkish text into one of six emotions. The model is trained on
**raw text** (no special preprocessing needed) and reaches **~0.97 accuracy and
macro-F1** on the held-out test set.

## Emotions

The model predicts one of six classes. Label ids follow `model.config.id2label`:

| id | Label | Emotion |
|----|---------|---|
| 0 | `anger` | 😠 |
| 1 | `disgust` | 🤢 |
| 2 | `fear` | 😨 |
| 3 | `joy` | 😊 |
| 4 | `sadness` | 😢 |
| 5 | `surprise` | 😲 |

## Project structure

```
EmotionClassification/
├── data/         # Dataset (CSV)
├── notebooks/    # Training notebook (data prep, fine-tuning, evaluation)
├── src/          # Reusable Python modules (data, model, training, evaluation)
├── ui.py         # Gradio app to try the model interactively
├── requirements.txt
├── LICENSE
└── README.md
```

## Installation

```bash
git clone https://github.com/furkankarakuz/EmotionClassification.git
cd EmotionClassification
pip install -r requirements.txt
```

## Usage

### Use the model directly

```python
from transformers import pipeline

pipe = pipeline("text-classification", model="furkankarakuz/turkish-emotion-classifier")
print(pipe("Bu film gerçekten harikaydı!"))
# [{'label': 'joy', 'score': 0.98}]
```

### Run the Gradio app

```bash
python ui.py
```

### Train it yourself

Open the notebook under `notebooks/` and run the cells. It covers data cleaning,
a stratified train/validation/test split, class-weighted fine-tuning, and
evaluation on the test set.

## Model details

| | |
|--------------------|------------------------------------------|
| Base model | `dbmdz/bert-base-turkish-cased` |
| Task | Single-label, multi-class classification |
| Language | Turkish (`tr`) |
| Classes | 6 |
| Max sequence length | 64 (dynamic padding) |
| Batch size | 32 |
| Learning rate | 2e-5 (linear, warmup ratio 0.1) |
| Epochs | 10 (early stopping, patience 2) |
| Loss | class-weighted cross-entropy |
| Best-model metric | macro-F1 (best at epoch 8) |
| Seed | 42 |

## Results

### Test set (3,520 samples)

| Metric | Value |
|--------|-------|
| Accuracy | 0.9690 |
| F1 (macro) | 0.9683 |
| Precision (macro) | 0.9676 |
| Recall (macro) | 0.9691 |
| Loss | 0.1880 |

### Per-class results (test set)

| Label | Precision | Recall | F1 | Support |
|-------|----------:|-------:|---:|--------:|
| anger | 0.96 | 0.96 | 0.96 | 616 |
| disgust | 0.96 | 0.98 | 0.97 | 419 |
| fear | 0.98 | 0.97 | 0.97 | 623 |
| joy | 0.98 | 0.98 | 0.98 | 783 |
| sadness | 0.96 | 0.97 | 0.96 | 662 |
| surprise | 0.96 | 0.96 | 0.96 | 417 |

### Validation results per epoch

Training ran for 10 epochs; the best checkpoint (**epoch 8**, by validation
macro-F1) was restored as the final model.

| Epoch | Train Loss | Val Loss | Accuracy | F1 (macro) | Precision (macro) | Recall (macro) |
|------:|-----------:|---------:|---------:|-----------:|------------------:|---------------:|
| 1 | 0.1866 | 0.1858 | 0.9460 | 0.9460 | 0.9481 | 0.9442 |
| 2 | 0.1412 | 0.1436 | 0.9591 | 0.9583 | 0.9575 | 0.9592 |
| 3 | 0.0766 | 0.1639 | 0.9619 | 0.9617 | 0.9621 | 0.9619 |
| 4 | 0.0364 | 0.1749 | 0.9622 | 0.9616 | 0.9627 | 0.9608 |
| 5 | 0.0269 | 0.1892 | 0.9631 | 0.9620 | 0.9609 | 0.9635 |
| 6 | 0.0132 | 0.1901 | 0.9656 | 0.9650 | 0.9652 | 0.9649 |
| 7 | 0.0141 | 0.1980 | 0.9642 | 0.9635 | 0.9648 | 0.9625 |
| **8** | **0.0011** | **0.1990** | **0.9676** | **0.9673** | **0.9670** | **0.9676** |
| 9 | 0.0063 | 0.2036 | 0.9665 | 0.9657 | 0.9661 | 0.9656 |
| 10 | 0.0043 | 0.2022 | 0.9670 | 0.9664 | 0.9667 | 0.9662 |

## Limitations

The model is trained on short, single-sentence text and predicts a single
dominant emotion. For long, multi-sentence input, split into sentences and
classify each separately. It is not intended for languages other than Turkish or
for emotions outside the six trained classes.
