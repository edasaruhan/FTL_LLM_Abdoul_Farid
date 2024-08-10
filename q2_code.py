import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from transformers import DistilBertForSequenceClassification
from transformers import AdamW, get_linear_schedule_with_warmup

# Load the dataset
df = pd.read_csv('data_Q2.csv')

# Basic EDA
print(df.head())
print(df['label'].value_counts())

# Visualize the distribution of classes
df['label'].value_counts().plot(kind='bar')
plt.title('Class Distribution')
plt.xlabel('Class')
plt.ylabel('Frequency')
plt.show()

# Split the dataset into training and validation sets
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])

from transformers import DistilBertTokenizerFast
from torch.utils.data import Dataset, DataLoader

# Initialize the tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

# Custom Dataset class for tokenization
class SDG9Dataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        encoding = self.tokenizer.encode_plus(
            self.texts[idx], 
            add_special_tokens=True, 
            max_length=self.max_len, 
            padding='max_length', 
            truncation=True, 
            return_attention_mask=True, 
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

# Prepare datasets
train_dataset = SDG9Dataset(train_df['text'].tolist(), train_df['label'].tolist(), tokenizer)
val_dataset = SDG9Dataset(val_df['text'].tolist(), val_df['label'].tolist(), tokenizer)

# DataLoader
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

# Load a pre-trained DistilBERT model
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=3)

# Fine tune Process
# Define the optimizer and learning rate scheduler
optimizer = AdamW(model.parameters(), lr=2e-5)
total_steps = len(train_loader) * 3  # Assuming 3 epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

# Training loop
for epoch in range(3):
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
        scheduler.step()

        total_train_loss += loss.item()
        correct_predictions += (logits.argmax(dim=1) == labels).sum().item()
    
    train_loss = total_train_loss / len(train_loader)
    train_accuracy = correct_predictions / len(train_loader.dataset)
    
    print(f"Epoch {epoch+1}: Train loss = {train_loss:.4f}, Train accuracy = {train_accuracy:.4f}")


# Evaluation function
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
            correct_predictions += (logits.argmax(dim=1) == labels).sum().item()
            predictions.extend(logits.argmax(dim=1).cpu().numpy())
            real_values.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(real_values, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(real_values, predictions, average='weighted')
    return accuracy, precision, recall, f1

# Evaluate the fine-tuned model
val_accuracy, val_precision, val_recall, val_f1 = evaluate_model(model, val_loader)
print(f"Validation Accuracy: {val_accuracy:.4f}")
print(f"Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1 Score: {val_f1:.4f}")

# Before fine-tuning evaluation
pretrained_model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=3).to(device)
pretrained_val_accuracy, pretrained_val_precision, pretrained_val_recall, pretrained_val_f1 = evaluate_model(pretrained_model, val_loader)
print(f"\nBefore Fine-Tuning - Validation Accuracy: {pretrained_val_accuracy:.4f}")
print(f"Precision: {pretrained_val_precision:.4f}, Recall: {pretrained_val_recall:.4f}, F1 Score: {pretrained_val_f1:.4f}")