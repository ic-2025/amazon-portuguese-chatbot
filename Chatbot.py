import os
import PyPDF2
import random

def extract_text_from_pdfs(pdf_folder):
    all_text = ''
    for filename in os.listdir(pdf_folder):
        if filename.endswith('.pdf'):
            path = os.path.join(pdf_folder, filename)
            with open(path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page in reader.pages:
                    all_text += page.extract_text() + '\n'
    return all_text

pdf_folder = './pdfs'  # altere para o caminho real
corpus_text = extract_text_from_pdfs(pdf_folder)

# %%
import re
import nltk

def preprocess_text(text):
    text = text.lower()
    # text = re.sub(r'\n+', ' ', text)
    # # MANTÉM . ! ? para detecção de fim de frase
    # text = re.sub(r'[^a-záéíóúàãõç\.\!\?\s]', '', text)
    # return text
    text = text.replace('\xa0', ' ')
    text = re.sub(r'\n+', '\n', text)  # mantém quebras de parágrafo
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r' +\n', '\n', text)
    return text.strip()

# corpus_text = "Belém é a capital do Pará. É conhecida pelo Círio de Nazaré! Você já visitou?"
processed_text = preprocess_text(corpus_text)
# clean_text = sent_tokenize(processed_text, language='portuguese')

# %%
from tensorflow.keras.models import load_model

# Load the model
model = load_model('portuguese_chatbot_vl.h5')

# Verify the model is loaded
model.summary()

import pickle

# Carrega o arquivo pickle
with open('tokenizer_vl.pickle', 'rb') as file:
    data = pickle.load(file)

print("Conteúdo carregado:", type(data))

tokenizer = data

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, BatchNormalization, Bidirectional, RepeatVector
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import re
import unicodedata



# Preprocessing functions
def preprocess_text(text):
    # Normalize and clean text
    text = unicodedata.normalize('NFKD', text.lower()).encode('ASCII', 'ignore').decode('ASCII')
    text = re.sub(r'[^\w\s]', '', text)
    return text

max_length=35

def generate_response(input_text, temperature=0.8):
    """
    Generate a response to the input text using the trained model.
    
    Args:
        input_text: User input string
        temperature: Controls randomness of predictions (higher = more random)
        
    Returns:
        Generated response string
    """
    # Preprocess input
    cleaned_input = preprocess_text(input_text)
    input_seq = tokenizer.texts_to_sequences([cleaned_input])
    input_padded = pad_sequences(input_seq, maxlen=max_length, padding='post')
    
    # Initialize response
    response_seq = np.zeros((1, max_length))
    response_seq[0, 0] = tokenizer.word_index['<OOV>']  # Start token
    
    for i in range(1, max_length):
        # Predict next word
        predictions = model.predict([input_padded, response_seq], verbose=0)[0][i-1]
        
        # Apply temperature for diversity
        predictions = np.log(predictions) / temperature
        exp_preds = np.exp(predictions)
        predictions = exp_preds / np.sum(exp_preds)
        
        # Sample from predictions
        next_word_idx = np.random.choice(len(predictions), p=predictions)
        response_seq[0, i] = next_word_idx
        
        # Stop if end token is predicted
        if next_word_idx == 0:
            break
    
    # Convert sequence to text
    response_tokens = []
    for idx in response_seq[0]:
        if idx == 0:  # Skip padding
            continue
        word = tokenizer.index_word.get(idx, '')
        if word == '<OOV>':  # Skip OOV tokens
            continue
        response_tokens.append(word)
    
    return ' '.join(response_tokens)

# # Test the chatbot
# test_inputs = [
#     "oi",
#     "conte-me sobre o folclore",
#     "quais frutas tem na amazônia?",
#     "quem é o curupira?",
#     "o que é Ver-o-Peso"
# ]

# for input_text in test_inputs:
#     response = generate_response(input_text)
#     print(f"User: {input_text}")
#     print(f"Chatbot: {response}\n")

while True:
    input_text = input("User: ")
    response = generate_response(input_text)
    print(f"Chatbot: {response}\n")


