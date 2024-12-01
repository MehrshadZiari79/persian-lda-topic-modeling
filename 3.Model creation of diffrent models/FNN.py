import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import re
from dadmatools.datasets import SnappfoodSentiment
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from dadmatools.normalizer import Normalizer

# Define text cleaning and normalization functions
def clean_text(text):
    """Remove punctuation and convert text to lowercase."""
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower()
    return text

def normalize_text(text):
    """Normalize the text using DadmaTools Normalizer."""
    normalizer = Normalizer()
    return normalizer.normalize(text)

def preprocess_text(text):
    """Preprocess the text by cleaning and normalizing it."""
    text = clean_text(text)
    text = normalize_text(text)
    return text

# Load the SnappfoodSentiment dataset
snpfood_sa = SnappfoodSentiment()

# Extract and preprocess comments and labels
comments = []
labels = []

for item in snpfood_sa.train:
    comment = item['comment']
    label = item['label']
    # Preprocess the comment
    comments.append(preprocess_text(comment))
    labels.append(label)

# Encode the labels
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)

# Tokenize the comments
tokenizer = Tokenizer()
tokenizer.fit_on_texts(comments)
sequences = tokenizer.texts_to_sequences(comments)

# Pad the sequences to ensure consistent input length
max_length = max(len(seq) for seq in sequences)
X = pad_sequences(sequences, maxlen=max_length)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

# Define the Feedforward Neural Network (FNN) model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=128, input_length=max_length),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Evaluate the model on the test set
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

# Decode predictions back to labels
predictions = label_encoder.inverse_transform(predictions.flatten())

# Print predictions
for comment, prediction in zip(new_comments, predictions):
    print(f'Comment: {comment}\nSentiment: {prediction}\n')
