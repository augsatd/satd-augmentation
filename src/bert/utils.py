import pandas as pd
import numpy as np
import random
import torch
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
import config

# Set seed for reproducibility
seed_value = 2042
random.seed(seed_value)
np.random.seed(seed_value)
torch.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)

# Load and preprocess dataset
def load_data(filepath):
    df = pd.read_csv(filepath, sep=',', encoding='utf-8')
    df = df.rename(columns={'text': 'text', 'classification': 'label'})
    df = df[~df.duplicated()]
    df.text = df.text.astype(str)
    df.label = df.label.astype(str)
    df = df[df["label"] != "non_debt"]
    df['label'] = df['label'].replace({'code_debt': 'code/design_debt', 'design_debt': 'code/design_debt'})
    df['label'] = df['label'].replace({'documentation_debt': 0, 'requirement_debt': 1, 'test_debt': 2, 'code/design_debt': 3})
    df['text_len'] = [len(text.split()) for text in df.text]
    df = df[df['text_len'] < df['text_len'].quantile(0.995)]
    return df

# Tokenization function
def bert_tokenizer(data, tokenizer, max_len):
    input_ids = []
    attention_masks = []
    for sent in data:
        encoded_sent = tokenizer.encode_plus(
            text=sent,
            add_special_tokens=True,
            max_length=max_len,
            pad_to_max_length=True,
            return_attention_mask=True
        )
        input_ids.append(encoded_sent.get('input_ids'))
        attention_masks.append(encoded_sent.get('attention_mask'))
    return torch.tensor(input_ids), torch.tensor(attention_masks)

# Split data into train, validation, and test sets
def split_data(df, test_size=0.1, valid_size=0.1, random_state=2042):
    X = df['text'].values
    y = df['label'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=valid_size, stratify=y_train, random_state=random_state)
    return X_train, X_valid, X_test, y_train, y_valid, y_test

# Configuration settings
MAX_LEN = 256
BATCH_SIZE = 32
EPOCHS = 10
device = 'cuda' if torch.cuda.is_available() else 'cpu'
labels = ["documentation_debt", "requirement_debt", "test_debt", "code-design_debt"]

# Initialize tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
