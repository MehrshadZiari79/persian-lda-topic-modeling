import re
import time
from dadmatools.datasets import SnappfoodSentiment
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from dadmatools.normalizer import Normalizer
import dadmatools.pipeline.language as language

# Define functions for text cleaning, normalization, and lemmatization
def clean_text(text):
    """Remove punctuation and convert text to lowercase."""
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text.lower()

def normalize_text(text):
    """Normalize text using DadmaTools' Normalizer."""
    normalizer = Normalizer()
    return normalizer.normalize(text)

def lemmatize_text(text, nlp):
    """Lemmatize text using DadmaTools pipeline."""
    doc = nlp(text)  # Process the text with DadmaTools
    lemmatized_tokens = []

    # Extract lemmatized tokens from the 'tokens' list in 'sentences'
    for sentence in doc.get('sentences', []):
        for token in sentence.get('tokens', []):
            lemmatized_tokens.append(token['lemma'])

    return ' '.join(lemmatized_tokens)

def preprocess_text(text, nlp):
    """Apply text cleaning, normalization, and lemmatization in sequence."""
    text = clean_text(text)
    text = normalize_text(text)
    return lemmatize_text(text, nlp)

# Initialize the DadmaTools pipeline for lemmatization
pips = 'lem'
nlp = language.Pipeline(pips)

# Start the timer to measure total processing time
start_time = time.time()

# Load the SnappfoodSentiment dataset
snpfood_sa = SnappfoodSentiment()
print("Dataset loaded.")

# Extract and preprocess  comments and labels
comments, labels = [], []
for i, item in enumerate(snpfood_sa.train):
    comment = preprocess_text(item['comment'], nlp)  # Preprocess each comment
    comments.append(comment)
    labels.append(item['label'])
print("Comments and labels extracted and preprocessed.")

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(comments, labels, test_size=0.2, random_state=42)
print("Data split into training and testing sets.")

# Define a pipeline with TF-IDF vectorization and Logistic Regression
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LogisticRegression(max_iter=1000))
])
print("Pipeline created.")

# Set up the parameter grid for GridSearchCV
param_grid = {
    'tfidf__max_df': [0.5, 0.75, 1.0],
    'tfidf__min_df': [1, 3, 5],
    'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3)],
    'tfidf__use_idf': [True, False],
    'clf__C': [0.01, 0.1, 1, 10, 100],
    'clf__penalty': ['l2'],
    'clf__solver': ['liblinear', 'saga']  # Solvers for Logistic Regression
}
print("Parameter grid defined.")

# Initialize GridSearchCV with cross-validation and parallel processing
grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, verbose=2)
print("GridSearchCV initialized.")

# Train the model using GridSearchCV
grid_search.fit(X_train, y_train)
print("Model training completed.")

# Evaluate the best model found by GridSearchCV
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Best Model Accuracy: {accuracy * 100:.2f}%')
print(f'Best Parameters: {grid_search.best_params_}')

# Define new comments for sentiment prediction
new_comments = [
    "این محصول عالیه! خیلی دوست داشتم.",
    "بسیار بد بود، اصلا راضی نبودم.",
    "متوسط بود، نه خیلی خوب و نه خیلی بد."
]

# Preprocess new comments
new_comments = [preprocess_text(comment, nlp) for comment in new_comments]
print("New comments preprocessed.")

# Predict sentiment for the new comments
predictions = best_model.predict(new_comments)

# Display predictions
for comment, prediction in zip(new_comments, predictions):
    print(f'Comment: {comment}\nSentiment: {prediction}\n')

# Calculate and print the total processing time
total_time = time.time() - start_time
print(f'Total time spent: {total_time:.2f} seconds')
