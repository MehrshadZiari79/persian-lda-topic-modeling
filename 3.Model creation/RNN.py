import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN custom operations

import time
import re
from dadmatools.datasets import SnappfoodSentiment
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from dadmatools.normalizer import Normalizer
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout, LayerNormalization, GRU
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping
from tensorflow.keras.regularizers import l2

# Define text cleaning and normalization functions
def clean_text(text):
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = text.lower()  # Convert to lowercase
    return text

def normalize_text(text):
    normalizer = Normalizer()
    return normalizer.normalize(text)

def preprocess_text(text):
    text = clean_text(text)
    text = normalize_text(text)
    return text

# Measure the start time
start_time = time.time()

# Load the SnappfoodSentiment dataset
snpfood_sa = SnappfoodSentiment()

# Extract comments and labels
comments = []
labels = []

for item in snpfood_sa.train:
    comment = item['comment']
    label = item['label']
    comment = preprocess_text(comment)  # Preprocess comment
    comments.append(comment)
    labels.append(label)

# Encode the labels
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)

# Tokenize the comments
tokenizer = Tokenizer()
tokenizer.fit_on_texts(comments)
sequences = tokenizer.texts_to_sequences(comments)

# Pad the sequences
max_length = max(len(seq) for seq in sequences)
X = pad_sequences(sequences, maxlen=max_length)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

# Load pre-trained word embeddings (Placeholder for actual embeddings)
embedding_dim = 128
embedding_matrix = tf.keras.initializers.GlorotUniform()(shape=(len(tokenizer.word_index) + 1, embedding_dim))

# Define the improved model architecture
model = tf.keras.Sequential([
    Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=embedding_dim, weights=[embedding_matrix], trainable=True),
    Bidirectional(LSTM(128, return_sequences=True)),
    LayerNormalization(),
    Dropout(0.4),
    Bidirectional(GRU(64, kernel_regularizer=l2(0.01))),
    LayerNormalization(),
    Dropout(0.4),
    Dense(128, activation='relu', kernel_regularizer=l2(0.01)),
    Dropout(0.4),
    Dense(1, activation='sigmoid')
])

# Compile the model with a dynamic learning rate scheduler
def lr_schedule(epoch, lr):
    return lr * 0.9 if epoch > 1 else lr

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model with early stopping and learning rate scheduler
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
lr_scheduler = LearningRateScheduler(lr_schedule)

history = model.fit(X_train, y_train, epochs=20, batch_size=64, validation_split=0.2, callbacks=[early_stopping, lr_scheduler])

# Evaluate the model
y_pred = (model.predict(X_test) > 0.5).astype("int32")
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Example comments for sentiment prediction
new_comments = [
    "این محصول عالیه! خیلی دوست داشتم.",
    "بسیار بد بود، اصلا راضی نبودم.",
    "متوسط بود، نه خیلی خوب و نه خیلی بد."
]

# Preprocess and tokenize new comments
new_comments = [preprocess_text(comment) for comment in new_comments]
new_sequences = tokenizer.texts_to_sequences(new_comments)
new_padded = pad_sequences(new_sequences, maxlen=max_length)

# Predict sentiment for new comments
predictions = (model.predict(new_padded) > 0.5).astype("int32")

# Decode predictions
predictions = label_encoder.inverse_transform(predictions.flatten())

# Print predictions
for comment, prediction in zip(new_comments, predictions):
    print(f'Comment: {comment}\nSentiment: {prediction}\n')

# Measure the end time
end_time = time.time()
total_time = end_time - start_time
print(f'Total processing time: {total_time:.2f} seconds')
