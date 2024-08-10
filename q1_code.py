# Import necessary libraries
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import pandas as pd
from sklearn.model_selection import train_test_split

# Load data from CSV file
df = pd.read_csv('data_Q1.csv')

# Map labels to integers
label_to_id = {'Inclusive Education': 0, 'Quality Learning': 1, 'Non-SDG Aligned': 2}
df['label_id'] = df['label'].map(label_to_id)

# Split into training and validation sets
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['text'].tolist(), df['label_id'].tolist(), test_size=0.2, random_state=42
)

# Custom Dataset class for tokenization
class SdgDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        encoding = self.tokenizer.encode_plus(
            self.texts[idx], add_special_tokens=True, max_length=self.max_len, 
            padding='max_length', truncation=True, return_attention_mask=True, return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

# Initialize tokenizer and create datasets and dataloaders
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
train_dataset = SdgDataset(train_texts, train_labels, tokenizer)
val_dataset = SdgDataset(val_texts, val_labels, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4)


# Load pre-trained model
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=3)

# Training loop
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
epochs = 3
for epoch in range(epochs):
    model.train()
    total_train_loss = 0
    correct_predictions = 0

    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        logits = outputs.logits
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()
        correct_predictions += torch.sum(torch.argmax(logits, dim=1) == labels)
    
    avg_train_loss = total_train_loss / len(train_loader)
    train_accuracy = correct_predictions.double() / len(train_loader.dataset)

    print(f'Epoch {epoch+1}: Train loss = {avg_train_loss}, Train accuracy = {train_accuracy}')


# Evaluation
def evaluate_model(model, data_loader):
    model.eval()
    correct_predictions = 0
    predictions = []
    real_values = []
    
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['labels']
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            correct_predictions += torch.sum(torch.argmax(logits, dim=1) == labels)
            predictions.extend(torch.argmax(logits, dim=1).cpu().numpy())
            real_values.extend(labels.cpu().numpy())
    
    accuracy = correct_predictions.double() / len(data_loader.dataset)
    precision, recall, f1, _ = precision_recall_fscore_support(real_values, predictions, average='weighted')
    return accuracy, precision, recall, f1

# Evaluate the model after training
val_accuracy, val_precision, val_recall, val_f1 = evaluate_model(model, val_loader)
print(f'Validation accuracy: {val_accuracy}, Precision: {val_precision}, Recall: {val_recall}, F1 Score: {val_f1}')