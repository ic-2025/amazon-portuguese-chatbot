import re
import nltk
from nltk.tokenize import sent_tokenize
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np

# Download NLTK punkt data (only needs to be done once)
nltk.download('punkt', download_dir='./nltk_data', quiet=True)


import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, BatchNormalization, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import re
import unicodedata

# Set random seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Sample dataset (in a real scenario, you'd have a much larger dataset)
# # data = {
#     "input": [
#         "oi", "olá", "bom dia", "boa tarde", "boa noite", 
#         "conte-me sobre pará", "fale sobre a amazônia", 
#         "quais são os folclores da região?", "o que é o carimbó?",
#         "quem é o curupira?", "o que significa boi-bumbá?",
#         "quais animais vivem na floresta?", "como é o clima no pará?",
#         "quais frutas típicas da amazônia?", "o que é lendário na região?"
#     ],
#     "response": [
#         "olá! como posso ajudar?", "oi! tudo bem?", "bom dia! em que posso ajudar?", 
#         "boa tarde! pergunte sobre o pará e a amazônia!", "boa noite! vamos conversar sobre o folclore amazônico?",
#         "o pará é um estado incrível no norte do brasil, cheio de cultura e natureza!", 
#         "a amazônia é a maior floresta tropical do mundo, com biodiversidade única!",
#         "a região tem folclores ricos como curupira, boi-bumbá, vitória-régia e mais!",
#         "carimbó é uma dança tradicional paraense com influências indígenas, africanas e europeias!",
#         "curupira é um protetor das florestas com pés virados para trás que confunde caçadores!",
#         "boi-bumbá é uma festa folclórica que conta a história de um boi que ressuscita!",
#         "onças, araras, botos, sucuris e milhões de espécies vivem na amazônia!",
#         "o pará tem clima equatorial, quente e úmido o ano todo com muitas chuvas!",
#         "açaí, cupuaçu, bacuri, taperebá e pupunha são frutas deliciosas da região!",
#         "a região é cheia de lendas como iara, boto cor-de-rosa e mapinguari!"
#     ]
# }
df = pd.read_csv('para_tourism_dataset.csv')

# df = pd.DataFrame(data)


# Preprocessing functions
def preprocess_text(text):
    # Normalize and clean text
    text = unicodedata.normalize('NFKD', text.lower()).encode('ASCII', 'ignore').decode('ASCII')
    text = re.sub(r'[^\w\s]', '', text)
    return text

# Apply preprocessing
df['input_clean'] = df['input'].apply(preprocess_text)
df['response_clean'] = df['response'].apply(preprocess_text)

# Tokenizer setup
tokenizer = Tokenizer(filters='', oov_token='<OOV>')
tokenizer.fit_on_texts(pd.concat([df['input_clean'], df['response_clean']]))

vocab_size = len(tokenizer.word_index) + 1
print(f"Vocabulary size: {vocab_size}")

# Convert texts to sequences
input_sequences = tokenizer.texts_to_sequences(df['input_clean'])
response_sequences = tokenizer.texts_to_sequences(df['response_clean'])

# Pad sequences
max_length = max(max(len(seq) for seq in input_sequences), max(len(seq) for seq in response_sequences))
input_padded = pad_sequences(input_sequences, maxlen=max_length, padding='post')
response_padded = pad_sequences(response_sequences, maxlen=max_length, padding='post')

# Split data
X_train, X_val, y_train, y_val = train_test_split(input_padded, response_padded, test_size=0.5, random_state=42)



def preprocess_text(text):
    """Enhanced text preprocessing with NLTK support"""
    text = text.lower()
    text = text.replace('\xa0', ' ')
    text = re.sub(r'\n+', '\n', text)  # preserve paragraph breaks
    text = re.sub(r'[ \t]+', ' ', text)  # collapse multiple spaces
    text = re.sub(r' +\n', '\n', text)  # remove spaces before newlines
    text = re.sub(r'[^\w\s\.\!\?]', '', text)  # keep basic punctuation
    return text.strip()

# Load and preprocess dataset
df = pd.read_csv('para_tourism_dataset.csv')

# Apply enhanced preprocessing
df['input_clean'] = df['input'].apply(preprocess_text)
df['response_clean'] = df['response'].apply(preprocess_text)

# Tokenization with NLTK sentence-aware processing
def tokenize_with_sentences(text):
    sentences = sent_tokenize(text, language='portuguese')
    return [sentence.strip() for sentence in sentences if sentence.strip()]

# Example usage of sentence tokenization (optional for your use case)
# corpus_text = "Belém é a capital do Pará. É conhecida pelo Círio de Nazaré! Você já visitou?"
# processed_text = preprocess_text(corpus_text)
# clean_sentences = tokenize_with_sentences(processed_text)

# Main tokenizer for sequences
tokenizer = Tokenizer(filters='', oov_token='<OOV>')
all_text = pd.concat([df['input_clean'], df['response_clean']]).tolist()
tokenizer.fit_on_texts(all_text)
vocab_size = len(tokenizer.word_index) + 1

# Sequence preparation
input_sequences = tokenizer.texts_to_sequences(df['input_clean'])
response_sequences = tokenizer.texts_to_sequences(df['response_clean'])

max_length = max(max(len(seq) for seq in input_sequences), 
                max(len(seq) for seq in response_sequences))

input_padded = pad_sequences(input_sequences, maxlen=max_length, padding='post')
response_padded = pad_sequences(response_sequences, maxlen=max_length, padding='post')

# Rest of the model implementation remains the same as previous...
# [Include all the model architecture, training, and generation code from earlier]
# Improved Model Architecture
embedding_dim = 128  # Reduced from 256 to prevent overfitting
lstm_units = 256     # Reduced from 512

model = Sequential([
    Embedding(vocab_size, embedding_dim, input_length=max_length),
    BatchNormalization(),  # Helps stabilize training
    Bidirectional(LSTM(lstm_units, return_sequences=True)),
    Dropout(0.3),  # Randomly disable 30% of neurons to prevent over-reliance
    BatchNormalization(),
    Bidirectional(LSTM(lstm_units, return_sequences=True)),
    Dropout(0.3),
    BatchNormalization(),
    Dense(vocab_size, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Prepare targets by shifting response sequences
y_train_shifted = np.zeros_like(y_train)
y_train_shifted[:, :-1] = y_train[:, 1:]
y_val_shifted = np.zeros_like(y_val)
y_val_shifted[:, :-1] = y_val[:, 1:]

# Early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

# Train the model
history = model.fit(
    X_train, y_train_shifted,
    validation_data=(X_val, y_val_shifted),
    epochs=200,
    batch_size=32,
    # callbacks=[early_stopping]
)

# Plot training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy over epochs')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss over epochs')
plt.legend()
plt.show()


# Enhanced response generation with sentence-aware processing
def generate_response_with_sentences(input_text, temperature=0.7):
    cleaned_input = preprocess_text(input_text)
    input_seq = tokenizer.texts_to_sequences([cleaned_input])
    input_padded = pad_sequences(input_seq, maxlen=max_length, padding='post')
    
    response_seq = np.zeros((1, max_length))
    response_seq[0, 0] = tokenizer.word_index['<OOV>']
    
    for i in range(1, max_length):
        predictions = model.predict([input_padded, response_seq], verbose=0)[0][i-1]
        predictions = np.log(predictions) / temperature
        exp_preds = np.exp(predictions)
        predictions = exp_preds / np.sum(exp_preds)
        
        next_word_idx = np.random.choice(len(predictions), p=predictions)
        response_seq[0, i] = next_word_idx
        
        if next_word_idx == 0:
            break
    
    response_tokens = [tokenizer.index_word.get(idx, '') for idx in response_seq[0] if idx != 0]
    raw_response = ' '.join(response_tokens)
    
    # Post-process with sentence tokenization for better formatting
    sentences = sent_tokenize(raw_response, language='portuguese')
    return ' '.join(s.strip() for s in sentences if s.strip())

# Test the enhanced version
test_questions = [
    "oi",
    "quais os melhores pontos turísticos de Belém?",
    "conte-me sobre o carimbó",
    "o que é pato no tucupi?"
]

print("Enhanced Chatbot with NLTK Processing:")
for question in test_questions:
    print(f"Usuário: {question}")
    print(f"Chatbot: {generate_response_with_sentences(question)}\n")