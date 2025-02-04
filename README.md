# Clinical BERT Pretraining

This repository provides a PyTorch implementation of **Clinical BERT**, a transformer-based model pre-trained on electronic health records (EHR) using **Masked Language Modeling (MLM)** and **Next Sentence Prediction (NSP)** tasks. The model is built on **BlueBERT** and can be further fine-tuned for downstream medical NLP tasks.

## Features
- **Pretraining with Clinical Texts**: Uses EHR notes and other clinical documents.
- **Masked Language Model (MLM)**: Randomly masks tokens and predicts them.
- **Next Sentence Prediction (NSP)**: Determines if one sentence follows another.
- **Based on BlueBERT**: Supports `bionlp/bluebert_pubmed_uncased_L-12_H-768_A-12`.
- **Efficient Training**: Includes `DataLoader`, `train.py`, and model saving/loading utilities.

## Installation

Ensure you have Python 3.8+ and install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### 1️⃣ **Dataset Preparation**
Ensure you have a dataset with **clinical notes**. Modify `train.py` to load your dataset:
```python
dataset = ClinicalTextDataset("path/to/your_dataset.csv", tokenizer)
```

### 2️⃣ **Training the Model**
Run the training script:
```bash
python train.py
```
This will train Clinical BERT using **MLM & NSP** and save the model checkpoint.

### 3️⃣ **Fine-Tuning for Classification**
Use `EHRBERTClassifier` to fine-tune on clinical classification tasks:
```python
from ehr_bert_pytorch import EHRBERTClassifier
model = EHRBERTClassifier("bionlp/bluebert_pubmed_uncased_L-12_H-768_A-12", num_labels=2)
```

### 4️⃣ **Evaluation**
```bash
python evaluate.py --model_path clinical_bert_pretrained.pth
```

## Requirements
See `requirements.txt` for dependencies.

## Model Architecture
- **BERT-base architecture** with:
  - Hidden size: `768`
  - Number of layers: `12`
  - Attention heads: `12`
  - Vocab size: `30522`
- MLM & NSP heads added for pretraining

## Citation
If you use this work, please cite:
```
@article{ClinicalBERT2025,
  title={Pretraining Clinical BERT on EHR Notes},
  author={Yi-Shan Lan},
  year={2025}
}
```

## Contact
For questions, open an issue or contact `r08b46009@gmail.com`.

