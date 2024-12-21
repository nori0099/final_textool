#!/usr/bin/env python3

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import shap

# Load the dataset
df = pd.read_csv("combine_complete.csv")

# Fill missing with an empty string
df['processed_text'] = df['processed_text'].fillna("")

# Split, train and test sets
train_texts, test_texts, train_scores, test_scores = train_test_split(
    df['processed_text'].tolist(), df['Score'].tolist(), test_size=0.2, random_state=42
)


tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Tokenize data
def tokenize_function(texts):
    return tokenizer(texts, padding="max_length", truncation=True, max_length=512, return_tensors="pt")

train_encodings = tokenize_function(train_texts)
test_encodings = tokenize_function(test_texts)

# Create a custom dataset
class PopularityDataset(Dataset):
    def __init__(self, encodings, scores):
        self.encodings = encodings
        self.scores = torch.tensor(scores, dtype=torch.float)

    def __len__(self):
        return len(self.scores)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = self.scores[idx]
        return item

train_dataset = PopularityDataset(train_encodings, train_scores)
test_dataset = PopularityDataset(test_encodings, test_scores)

# regression
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=1)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",          
    num_train_epochs=3,              
    per_device_train_batch_size=4,   
    per_device_eval_batch_size=4,    
    eval_strategy="epoch",    
    save_strategy="epoch",
    logging_dir="./logs",            
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
)

# Define a Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

# Train the model
trainer.train()

# Save the model
trainer.save_model("./bert_popularity_model")

# Predict on test data
predictions = trainer.predict(test_dataset)
predicted_scores = predictions.predictions.squeeze()

# Evaluate the model
mse = mean_squared_error(test_scores, predicted_scores)
rmse = np.sqrt(mse)

print(f"Mean Squared Error: {mse}")
print(f"Mean Absolute Error: {mae}")
print(f"Root Mean Squared Error: {rmse}")





