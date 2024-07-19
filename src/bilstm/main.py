from data_preparation import prepare_data
from embedding_loader import load_glove_embeddings, create_embedding_matrix
from model_builder import build_bilstm_model
from train_model import train_bilstm_model
from evaluate_model import evaluate_model
import pandas as pd

# Import Dataset
df = pd.read_csv('./commit_augmented.csv', sep=',', encoding='utf-8')

# Prepare Data
texts, labels, padded_sequences, word_index, df = prepare_data(df)

# Load GloVe Embeddings
embedding_dim = 300
glove_path = './glove.840B.300d.txt'
embedding_index = load_glove_embeddings(glove_path)
embedding_matrix = create_embedding_matrix(word_index, embedding_index, embedding_dim)

# Train/Test Split
from sklearn.model_selection import train_test_split
X = padded_sequences
y = df.classification.values
y = np.asarray(y).astype("float64")
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.1, random_state=1)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, stratify=y_train, test_size=0.1, random_state=1)

# One-hot encoding labels
import pandas as pd
y_train_enc = pd.get_dummies(y_train).to_numpy()
y_val_enc = pd.get_dummies(y_val).to_numpy()
y_test_enc = pd.get_dummies(y_test).to_numpy()

# Build Model
num_words = len(word_index) + 1
max_sequence_length = 256
model = build_bilstm_model(num_words, embedding_dim, embedding_matrix, max_sequence_length)

# Train Model
history_embedding = train_bilstm_model(model, X_train, y_train_enc, X_val, y_val_enc)

# Evaluate Model
evaluate_model(model, X_test, y_test_enc, history_embedding)
