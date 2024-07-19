import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def prepare_data(df):
    df.dropna(inplace=True)
    df['text'].dropna(inplace=True)
    texts = df['text'].tolist()
    
    # Filter out unwanted classifications
    df = df[df["classification"]!="architecture_debt"]
    df = df[df["classification"]!="build_debt"]
    df = df[df["classification"]!="defect_debt"]
    
    # Replace classifications with binary labels
    df['classification'].replace({
        'non_debt': 0.0, 'design_debt': 1.0, 'code_debt': 1.0, 
        'requirement_debt': 1.0, 'test_debt': 1.0, 
        'documentation_debt': 1.0, 'code-design_debt': 1.0
    }, inplace=True)

    # Tokenize and pad the text data
    max_sequence_length = 256
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts)
    word_index = tokenizer.word_index
    sequences = tokenizer.texts_to_sequences(texts)
    padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length)
    
    return texts, df['classification'].tolist(), padded_sequences, word_index, df
