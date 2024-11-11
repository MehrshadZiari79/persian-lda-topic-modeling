import re
import time
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from dadmatools.normalizer import Normalizer
from dadmatools.datasets import SnappfoodSentiment

# Define text cleaning function to remove punctuation
def clean_text(text):
    """Clean text by removing punctuation and converting it to lowercase."""
    if not isinstance(text, str):
        text = ''  # Handle non-string values
    return re.sub(r'[^\w\s]', '', text).lower()

# Define normalization function using DadmaTools
def normalize_text(text):
    """Normalize text using DadmaTools' Normalizer."""
    normalizer = Normalizer()
    return normalizer.normalize(text)

# Preprocess text by cleaning and normalizing
def preprocess_text(text):
    """Apply text cleaning and normalization to the input text."""
    text = clean_text(text)
    return normalize_text(text)

# Function to log and time the start of a task
def time_part(part_name):
    """Log the start of a task and return the start time."""
    start = time.time()
    print(f"\nStarting {part_name}...")
    return start

# Function to log and display the elapsed time for a task
def end_part(start_time, part_name):
    """Log the end of a task and display the elapsed time."""
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"{part_name} completed in {elapsed_time:.2f} seconds.")

# Start the overall timer
start_time = time.time()

# Load the SnappfoodSentiment dataset
load_data_start = time_part("Loading Data")
snpfood_sa = SnappfoodSentiment()
end_part(load_data_start, "Loading Data")

# Extract comments and labels from dataset, applying preprocessing
data_prep_start = time_part("Preparing Data")
comments = [preprocess_text(item['comment']) for item in snpfood_sa.train]
labels = [item['label'] for item in snpfood_sa.train]
X_train, X_test, y_train, y_test = train_test_split(comments, labels, test_size=0.2, random_state=42)
end_part(data_prep_start, "Preparing Data")

# Define a pipeline with TF-IDF and Logistic Regression
model_training_start = time_part("Training Model")
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LogisticRegression(max_iter=1000))
])

# Define parameter grid for hyperparameter tuning
param_grid = {
    'tfidf__max_df': [0.75, 1.0],
    'tfidf__min_df': [1, 5],
    'tfidf__ngram_range': [(1, 1), (1, 2)],
    'clf__C': [0.1, 1, 10],
    'clf__penalty': ['l2']
}

# Initialize and run GridSearchCV for hyperparameter optimization
grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)
end_part(model_training_start, "Training Model")

# Evaluate the best model from GridSearch
evaluation_start = time_part("Evaluating Model")
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Best Model Accuracy: {accuracy * 100:.2f}%')
print(f'Best Parameters: {grid_search.best_params_}')
end_part(evaluation_start, "Evaluating Model")

# Load additional comments from CSV file for sentiment prediction
file_loading_start = time_part("Loading Comments from CSV")
csv_file_path = 'cms_grouped.csv'
df = pd.read_csv(csv_file_path)
end_part(file_loading_start, "Loading Comments from CSV")

# Preprocess comments from CSV file
preprocessing_start = time_part("Preprocessing Comments")
processed_comments = [preprocess_text(comment) for comment in df['comment'].tolist()]
end_part(preprocessing_start, "Preprocessing Comments")

# Predict sentiment for each preprocessed comment
prediction_start = time_part("Predicting Sentiments")
predictions = best_model.predict(processed_comments)
end_part(prediction_start, "Predicting Sentiments")

# Add predictions to the DataFrame and save results to CSV
save_results_start = time_part("Saving Results")
df['Sentiment'] = predictions
output_file_path = 'analysed_cm.csv'
df.to_csv(output_file_path, index=False)
end_part(save_results_start, "Saving Results")

# Print results with numbering for each comment
print("\nSentiment Analysis Results:")
for index, (comment, prediction) in enumerate(zip(df['comment'], predictions), start=1):
    print(f"{index}. Comment: {comment}\n   Sentiment: {prediction}\n")

# Display the total time spent
total_time = time.time() - start_time
print(f'Total time spent: {total_time:.2f} seconds')
