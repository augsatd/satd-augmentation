import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from utils import load_data, split_data, bert_tokenizer, tokenizer
import config
from model_builder import BertClassifier
from train_model import initialize_model, train_model
from predict import predict, plot_confusion_matrix
from sklearn.metrics import classification_report

# Load and preprocess data
df = load_data('commit_augmented.csv')
X_train, X_valid, X_test, y_train, y_valid, y_test = split_data(df)

# Tokenize data
train_inputs, train_masks = bert_tokenizer(X_train, tokenizer, config.MAX_LEN)
val_inputs, val_masks = bert_tokenizer(X_valid, tokenizer, config.MAX_LEN)
test_inputs, test_masks = bert_tokenizer(X_test, tokenizer, config.MAX_LEN)

train_labels = torch.tensor(y_train)
val_labels = torch.tensor(y_valid)
test_labels = torch.tensor(y_test)

# Create DataLoaders
train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=config.BATCH_SIZE)

val_data = TensorDataset(val_inputs, val_masks, val_labels)
val_sampler = SequentialSampler(val_data)
val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=config.BATCH_SIZE)

test_data = TensorDataset(test_inputs, test_masks, test_labels)
test_sampler = SequentialSampler(test_data)
test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=config.BATCH_SIZE)

# Initialize and train model
model, optimizer, scheduler = initialize_model(train_dataloader, epochs=config.EPOCHS)
train_model(model, train_dataloader, val_dataloader, epochs=config.EPOCHS)

# Make predictions on test data
predictions = predict(model, test_dataloader)

# Classification report and confusion matrix
print('Classification Report for BERT:\n', classification_report(y_test, predictions, target_names=config.labels, digits=3))
plot_confusion_matrix(y_test, predictions, config.labels)
