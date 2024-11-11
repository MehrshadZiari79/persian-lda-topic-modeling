import re
import logging
from dadmatools.datasets import SnappfoodSentiment
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
from dadmatools.normalizer import Normalizer
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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

logging.info('Loading dataset...')
start_time = time.time()

# Load the SnappfoodSentiment dataset
snpfood_sa = SnappfoodSentiment()

logging.info(f'Dataset loaded in {time.time() - start_time:.2f} seconds.')

# Extract comments and labels
logging.info('Extracting and preprocessing comments and labels...')
start_time = time.time()

comments = []
labels = []

for item in snpfood_sa.train:
    comment = item['comment']
    label = item['label']
    comment = preprocess_text(comment)  # Preprocess comment
    comments.append(comment)
    labels.append(label)

logging.info(f'Comments and labels extracted and preprocessed in {time.time() - start_time:.2f} seconds.')

# Split the dataset into training and testing sets
logging.info('Splitting dataset into training and testing sets...')
start_time = time.time()

X_train, X_test, y_train, y_test = train_test_split(comments, labels, test_size=0.2, random_state=42)

logging.info(f'Dataset split in {time.time() - start_time:.2f} seconds.')

# Create a pipeline
logging.info('Creating and training the pipeline...')
start_time = time.time()

pipeline = make_pipeline(TfidfVectorizer(), RandomForestClassifier(n_estimators=100, random_state=42))
pipeline.fit(X_train, y_train)

logging.info(f'Pipeline created and trained in {time.time() - start_time:.2f} seconds.')

# Evaluate the model
logging.info('Evaluating the model...')
start_time = time.time()

y_pred = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

logging.info(f'Model evaluated in {time.time() - start_time:.2f} seconds.')
print(f'Accuracy: {accuracy * 100:.2f}%')

# Example comments for sentiment prediction
new_comments = [
    "این محصول عالیه! خیلی دوست داشتم.",
    "بسیار بد بود، اصلا راضی نبودم.",
    "متوسط بود، نه خیلی خوب و نه خیلی بد."
]

# Preprocess new comments
logging.info('Preprocessing new comments...')
start_time = time.time()

new_comments = [preprocess_text(comment) for comment in new_comments]

logging.info(f'New comments preprocessed in {time.time() - start_time:.2f} seconds.')

# Predict sentiment for new comments
logging.info('Predicting sentiment for new comments...')
start_time = time.time()

predictions = pipeline.predict(new_comments)

logging.info(f'Sentiment prediction done in {time.time() - start_time:.2f} seconds.')

# Print predictions
for comment, prediction in zip(new_comments, predictions):
    print(f'Comment: {comment}\nSentiment: {prediction}\n')
