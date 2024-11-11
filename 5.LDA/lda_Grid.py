import os
import shutil
import pandas as pd
import dadmatools.pipeline.language as language
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import GridSearchCV, ParameterGrid
import re
import numpy as np
import time
from openpyxl import Workbook
import openpyxl

# Start the timer
start_time = time.time()

# Load stopword list
with open('Stopwords_shokristop_words.txt', 'r', encoding='utf-8') as file:
    stopwords = file.read().splitlines()


# Function to clean and preprocess text
def preprocess_text(text):
    text = re.sub(r'\u200c', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[A-Za-z]', '', text)
    words = [word for word in text.split() if word not in stopwords]
    return ' '.join(words)


# Initialize dadmatools with a custom model path
custom_models_dir = ''
cache_dir = os.path.join(os.path.expanduser("~"), "cache", "dadmatools")
pipelines = 'lem,chunk'
nlp = language.Pipeline(pipelines=pipelines)


# Function to extract noun-adjective pairs
def extract_noun_adj_pairs(text):
    doc = nlp(text)
    noun_adj_pairs = []
    sentences = doc['sentences']
    for sentence in sentences:
        tokens = sentence['tokens']
        for i in range(len(tokens) - 1):
            if tokens[i]['upos'] == 'NOUN' and tokens[i + 1]['upos'] == 'ADJ':
                noun_adj_pairs.append(f"{tokens[i]['text']} {tokens[i + 1]['text']}")
    return noun_adj_pairs


# Function to process comments
def process_comments(sentiment, sample_fraction=1):
    filtered_comments = df[df['Sentiment'] == sentiment]['comment'].dropna().sample(frac=sample_fraction,
                                                                                    random_state=42)
    processed_comments = []
    for comment in filtered_comments:
        noun_adj_pairs = extract_noun_adj_pairs(comment)
        cleaned_comment = preprocess_text(' '.join(noun_adj_pairs))
        processed_comments.append(cleaned_comment)
    return processed_comments


# Perform LDA with GridSearch for multiple parameters
def perform_lda_with_grid_search(comments):
    # Define parameter grid
    param_grid = {
        'n_components': [5, 7, 9],  # Number of topics
        'max_iter': [100, 300, 500],  # Number of iterations
        'learning_method': ['batch'],
        'doc_topic_prior': [0.01, 0.05, 0.1],  # Dirichlet prior for document-topic
        'topic_word_prior': [0.01, 0.05, 0.1]  # Dirichlet prior for topic-word
    }

    # Define CountVectorizer grid for text processing
    vectorizer_params = {
        'max_df': [0.5, 0.6, 0.7],
        'min_df': [5, 10, 15],
        'ngram_range': [(1, 2), (1, 3)]
    }

    # Perform Grid Search with CountVectorizer and LDA
    best_score = -1
    best_params = None
    best_lda_model = None
    best_vectorizer = None

    for vec_params in ParameterGrid(vectorizer_params):
        vectorizer = CountVectorizer(
            stop_words=stopwords,
            **vec_params
        )
        dtm = vectorizer.fit_transform(comments)

        # Check if there are enough terms
        if dtm.shape[1] == 0:
            continue

        lda = LatentDirichletAllocation(random_state=42)
        grid_search = GridSearchCV(lda, param_grid, cv=3, n_jobs=-1)
        grid_search.fit(dtm)

        if grid_search.best_score_ > best_score:
            best_score = grid_search.best_score_
            best_params = {**vec_params, **grid_search.best_params_}
            best_lda_model = grid_search.best_estimator_
            best_vectorizer = vectorizer

    print(f"Best parameters found: {best_params}")
    return best_lda_model, best_vectorizer


# Process comments and perform Grid Search for optimal model
df = pd.read_csv('analysed_cm.csv')
happy_comments = process_comments('HAPPY')
sad_comments = process_comments('SAD')

lda_happy, vectorizer_happy = perform_lda_with_grid_search(happy_comments)
lda_sad, vectorizer_sad = perform_lda_with_grid_search(sad_comments)


# Extract and save topic information to an Excel file
def get_topic_names_and_importance(model, feature_names, no_top_words=3):
    topics = {}
    for topic_idx, topic in enumerate(model.components_):
        topic_words = [feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]
        topics[topic_idx] = ' '.join(topic_words)
    return topics


# Create an Excel workbook and sheet
wb = Workbook()
ws = wb.active
ws.title = "LDA_Topics"

# Add headers
ws.append(["Sentiment", "Topic Number", "Top Words", "Best Parameters"])

# Save topics for 'happy' comments
if lda_happy and vectorizer_happy:
    happy_topics = get_topic_names_and_importance(lda_happy, vectorizer_happy.get_feature_names_out())
    for topic_num, words in happy_topics.items():
        ws.append(["Happy", topic_num, words, str(lda_happy.get_params())])

# Save topics for 'sad' comments
if lda_sad and vectorizer_sad:
    sad_topics = get_topic_names_and_importance(lda_sad, vectorizer_sad.get_feature_names_out())
    for topic_num, words in sad_topics.items():
        ws.append(["Sad", topic_num, words, str(lda_sad.get_params())])

# Save the Excel workbook
wb.save("LDA_Topics.xlsx")

# End timer and print runtime
end_time = time.time()
print(f"Execution Time: {end_time - start_time:.2f} seconds")
