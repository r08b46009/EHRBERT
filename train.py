import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import random
import json
from ehr_bert_pytorch import EHRBERTPretrain, mask_tokens, save_model, load_model

# ===========================
# Configuration for Clinical BERT
# ===========================
class ClinicalBERTConfig:
    def __init__(self):
        self.vocab_size = 30522
        self.hidden_size = 768
        self.num_hidden_layers = 12
        self.num_attention_heads = 12
        self.intermediate_size = 3072
        self.hidden_act = "gelu"
        self.hidden_dropout_prob = 0.1
        self.attention_probs_dropout_prob = 0.1
        self.max_position_embeddings = 512
        self.type_vocab_size = 2
        self.initializer_range = 0.02

# ===========================
# Dataset for MLM & NSP
# ===========================
class ClinicalTextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.texts = texts
    
    def __len__(self):
        return len(self.texts) - 1  # NSP needs at least two sentences
    
    def __getitem__(self, idx):
        text_a = self.texts[idx]
        text_b = self.texts[idx + 1] if random.random() > 0.5 else self.texts[random.randint(0, len(self.texts) - 1)]
        label = 1 if text_b == self.texts[idx + 1] else 0  # NSP Label
        
        encoding = self.tokenizer(text_a, text_b, max_length=self.max_length, padding='max_length', truncation=True, return_tensors='pt')
        input_ids, attention_mask, token_type_ids = encoding['input_ids'].squeeze(0), encoding['attention_mask'].squeeze(0), encoding['token_type_ids'].squeeze(0)
        
        masked_input_ids, masked_lm_labels = mask_tokens(input_ids.clone(), self.tokenizer)
        
        return masked_input_ids, attention_mask, token_type_ids, masked_lm_labels, torch.tensor(label)

# ===========================
# Training Function
# ===========================
def train(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    for batch in dataloader:
        input_ids, attention_mask, token_type_ids, masked_lm_labels, next_sentence_labels = [t.to(device) for t in batch]
        optimizer.zero_grad()
        loss, _, _ = model(input_ids, attention_mask, token_type_ids, masked_lm_labels, next_sentence_labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

# ===========================
# Evaluation Function
# ===========================
def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids, attention_mask, token_type_ids, masked_lm_labels, next_sentence_labels = [t.to(device) for t in batch]
            loss, _, _ = model(input_ids, attention_mask, token_type_ids, masked_lm_labels, next_sentence_labels)
            total_loss += loss.item()
    return total_loss / len(dataloader)

# ===========================
# Main Training Script
# ===========================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained("bionlp/bluebert_pubmed_mimic_uncased")

    # Simulated clinical text data
    sample_texts = ["Patient reports severe chest pain.", "ECG shows signs of myocardial infarction.", "Administered aspirin and oxygen therapy.", "Blood pressure is stable."]
    dataset = ClinicalTextDataset(sample_texts, tokenizer)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    # Load Model
    model = EHRBERTPretrain().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=5e-5)
    
    # Training Loop
    num_epochs = 3
    for epoch in range(num_epochs):
        train_loss = train(model, dataloader, optimizer, device)
        val_loss = evaluate(model, dataloader, device)
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")
    
    # Save Model
    save_model(model, "clinical_bert_pretrained.pth")
    print("Model training complete and saved.")