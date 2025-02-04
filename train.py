import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import random
import json
# from ehr_bert_pytorch import EHRBERTPretrain, mask_tokens, save_model, load_model
import torch
import torch.nn as nn
import torch.optim as optim
import json
import os
import random
from transformers import BertModel, BertTokenizer, BertForSequenceClassification, BertForQuestionAnswering

# ===========================
# configs.py (Model Configuration)
# ===========================
class BertConfig:
    def __init__(self, vocab_size=30522, hidden_size=128, num_hidden_layers=1,
                 num_attention_heads=4, intermediate_size=128, hidden_act="gelu",
                 hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1,
                 max_position_embeddings=512, type_vocab_size=16, initializer_range=0.02):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range

    def to_json(self):
        return json.dumps(self.__dict__, indent=2)

    @classmethod
    def from_json(cls, json_str):
        return cls(**json.loads(json_str))

# ===========================
# model_saving_utils.py (Model Saving & Loading)
# ===========================
def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_model(model, path):
    model.load_state_dict(torch.load(path))
    return model

# ===========================
# model_training_utils.py (Training Utilities)
# ===========================
def train_model(model, dataloader, optimizer, device):
    model.train()
    for batch in dataloader:
        input_ids, attention_mask, token_type_ids, labels = [t.to(device) for t in batch]
        optimizer.zero_grad()
        loss, _ = model(input_ids, attention_mask, token_type_ids, labels)
        loss.backward()
        optimizer.step()
    return loss.item()

def evaluate_model(model, dataloader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids, attention_mask, token_type_ids, labels = [t.to(device) for t in batch]
            loss, _ = model(input_ids, attention_mask, token_type_ids, labels)
            total_loss += loss.item()
    return total_loss / len(dataloader)

# ===========================
# bert_models_edit.py (Edited BERT Model with Masking)
# ===========================
def mask_tokens(inputs, tokenizer, mask_probability=0.15):
    """Applies Masking for MLM task."""
    labels = inputs.clone()

    # 確保 `inputs` 是列表 (避免 `TypeError`)
    if isinstance(inputs, torch.Tensor):
        inputs_list = inputs.tolist()
    else:
        inputs_list = inputs  # 直接使用原始輸入

    probability_matrix = torch.full(labels.shape, mask_probability)

    # 修正：確保 `get_special_tokens_mask()` 正確運行
    special_tokens_mask = [tokenizer.get_special_tokens_mask([val], already_has_special_tokens=True)[0] for val in inputs_list]
    
    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    
    labels[~masked_indices] = -100  # 只計算 `MASK` Token 的損失
    inputs[masked_indices] = tokenizer.convert_tokens_to_ids("[MASK]")

    return inputs, labels

class EHRBERTPretrain(nn.Module):
    def __init__(self, bert_model_name="bert-base-uncased", vocab_size=30522):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.mlm_head = nn.Linear(self.bert.config.hidden_size, vocab_size)
        self.nsp_head = nn.Linear(self.bert.config.hidden_size, 2)
        self.loss_layer = nn.CrossEntropyLoss()
    
    def forward(self, input_ids, attention_mask, token_type_ids, masked_lm_labels, next_sentence_labels):
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output, pooled_output = outputs.last_hidden_state, outputs.pooler_output
        lm_logits = self.mlm_head(sequence_output)
        nsp_logits = self.nsp_head(pooled_output)
        loss = self.loss_layer(lm_logits.view(-1, self.bert.config.vocab_size), masked_lm_labels.view(-1))
        return loss, lm_logits, nsp_logits

# ===========================
# bert_models_test.py (Unit Testing)
# ===========================
def test_bert_pretrain():
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = EHRBERTPretrain()
    input_ids = torch.randint(0, 30522, (2, 128))
    attention_mask = torch.ones_like(input_ids)
    token_type_ids = torch.zeros_like(input_ids)
    masked_input_ids, masked_lm_labels = mask_tokens(input_ids, tokenizer)
    next_sentence_labels = torch.randint(0, 2, (2, 1))
    loss, _, _ = model(masked_input_ids, attention_mask, token_type_ids, masked_lm_labels, next_sentence_labels)
    assert loss is not None
    print("BERT Pretraining with Masking Test Passed!")


import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertModel, BertTokenizer, BertForSequenceClassification, BertForQuestionAnswering

class BertPretrainLossAndMetricLayer(nn.Module):
    """Computes Masked LM and Next Sentence Prediction loss."""
    def __init__(self, vocab_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, lm_output_logits, sentence_output_logits, lm_labels, lm_label_weights, sentence_labels=None):
        # Masked LM Loss
        lm_loss = self.cross_entropy(lm_output_logits.view(-1, self.vocab_size), lm_labels.view(-1))
        
        # Next Sentence Prediction Loss
        if sentence_labels is not None:
            sentence_loss = self.cross_entropy(sentence_output_logits.view(-1, 2), sentence_labels.view(-1))
            total_loss = lm_loss + sentence_loss
        else:
            total_loss = lm_loss
        
        return total_loss

class EHRBERTPretrain(nn.Module):
    """Pretraining model with MLM & NSP"""
    def __init__(self, bert_model_name="bert-base-uncased", vocab_size=30522):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.mlm_head = nn.Linear(self.bert.config.hidden_size, vocab_size)
        self.nsp_head = nn.Linear(self.bert.config.hidden_size, 2)
        self.loss_layer = BertPretrainLossAndMetricLayer(vocab_size)

    def forward(self, input_ids, attention_mask, token_type_ids, masked_lm_labels, next_sentence_labels):
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output, pooled_output = outputs.last_hidden_state, outputs.pooler_output
        
        lm_logits = self.mlm_head(sequence_output)
        nsp_logits = self.nsp_head(pooled_output)
        loss = self.loss_layer(lm_logits, nsp_logits, masked_lm_labels, attention_mask, next_sentence_labels)
        
        return loss, lm_logits, nsp_logits

class EHRBERTClassifier(nn.Module):
    """Fine-tuning model for classification."""
    def __init__(self, bert_model_name="bert-base-uncased", num_labels=2):
        super().__init__()
        self.bert = BertForSequenceClassification.from_pretrained(bert_model_name, num_labels=num_labels)
    
    def forward(self, input_ids, attention_mask, token_type_ids, labels):
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=labels)
        return outputs.loss, outputs.logits

class EHRBERTQuestionAnswering(nn.Module):
    """Fine-tuning model for SQuAD-style question answering."""
    def __init__(self, bert_model_name="bert-base-uncased"):
        super().__init__()
        self.bert = BertForQuestionAnswering.from_pretrained(bert_model_name)
    
    def forward(self, input_ids, attention_mask, token_type_ids, start_positions, end_positions):
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                            start_positions=start_positions, end_positions=end_positions)
        return outputs.loss, outputs.start_logits, outputs.end_logits

# Example usage
def train(model, dataloader, optimizer, device):
    model.train()
    for batch in dataloader:
        input_ids, attention_mask, token_type_ids, labels = [t.to(device) for t in batch]
        optimizer.zero_grad()
        loss, _ = model(input_ids, attention_mask, token_type_ids, labels)
        loss.backward()
        optimizer.step()
    return loss.item()


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
# ===========================
# Dataset for MLM & NSP
# ===========================
class ClinicalTextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.texts = texts
    
    def __len__(self):
        return len(self.texts) - 1  # NSP 需要至少兩個句子
    
    def __getitem__(self, idx):
        text_a = self.texts[idx]
        text_b = self.texts[idx + 1] if random.random() > 0.5 else self.texts[random.randint(0, len(self.texts) - 1)]
        label = 1 if text_b == self.texts[idx + 1] else 0  # NSP Label
        
        encoding = self.tokenizer(text_a, text_b, max_length=self.max_length, padding='max_length', truncation=True, return_tensors='pt')
        input_ids, attention_mask, token_type_ids = encoding['input_ids'].squeeze(0), encoding['attention_mask'].squeeze(0), encoding['token_type_ids'].squeeze(0)

        # 🛠️ DEBUG：檢查 input_ids 形狀
        print(f"DEBUG: input_ids shape: {input_ids.shape}, type: {type(input_ids)}")

        # 應用 Masking
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
    tokenizer = BertTokenizer.from_pretrained("bionlp/bluebert_pubmed_uncased_L-12_H-768_A-12")
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
