import shutil
import dadmatools.pipeline.language as language
import re
import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import arabic_reshaper
from bidi.algorithm import get_display
import multiprocessing as mp
import time
import psutil

# Timer to measure the execution time
start_time = time.time()

# Define the path to the stopwords file
stopword_file = 'Stopwords_shokristop_words.txt'

# Check if stopword file exists
if not os.path.exists(stopword_file):
    raise FileNotFoundError(f"Stopword file '{stopword_file}' not found.")

# Read the stopwords from the file
with open(stopword_file, 'r', encoding='utf-8') as file:
    stopwords = [word.strip() for word in file if word.strip()]

# Preprocess and clean each text comment
def pre_clean_text(text):
    """Removes extra spaces and skips very short comments."""
    text = text.strip()  # Remove leading/trailing whitespace
    if len(text.split()) < 3:  # Skip comments that are too short
        return ""
    return text

# Perform text preprocessing, including normalization and stopword removal
def preprocess_text(text):
    """Cleans and normalizes Persian text."""
    replacements = {
        'ی': 'ي',  # Normalize Persian 'ye' character
        'ک': 'ك',  # Normalize Persian 'kaf' character
        '\u0640': ''  # Remove Tatweel (Arabic script elongation)
    }
    # Apply character replacements
    for old, new in replacements.items():
        text = text.replace(old, new)
    # Remove unnecessary characters
    text = re.sub(r'\u200c', '', text)  # Remove ZWNJ
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\d+', '', text)  # Remove digits
    text = re.sub(r'[A-Za-z]', '', text)  # Remove Latin characters
    words = [word for word in text.split() if word not in stopwords]  # Remove stopwords
    return ' '.join(words)

# Directory paths for custom models and cache
custom_models_dir = 'E:\\payan\\WORDPR\\dadma\\xlm-roberta-base'
cache_dir = os.path.join(os.path.expanduser("~"), "cache", "dadmatools")

# Function to calculate the optimal number of parallel jobs based on system resources
def calculate_n_jobs():
    """Calculates the number of parallel processes based on CPU cores and available memory."""
    cpu_cores = mp.cpu_count()  # Get number of CPU cores
    memory_per_process = 500  # Set estimated memory per process (in MB)
    available_memory = psutil.virtual_memory().available / 1024 / 1024  # Get available memory in MB
    max_processes_by_memory = int(available_memory / memory_per_process)  # Calculate max processes by memory
    n_jobs = min(cpu_cores - 1, max_processes_by_memory)  # Use all but 1 core
    return int(n_jobs * 0.9)  # Use 90% of available CPU/Memory

# Function to copy model files to cache if they do not exist
def copy_model_if_not_exists(custom_dir, cache_subdir):
    """Copies model files from custom directory to cache if not already present."""
    if not os.path.exists(cache_subdir):
        os.makedirs(cache_subdir)  # Create cache directory if it doesn't exist
    for filename in os.listdir(custom_dir):  # Iterate over files in custom model directory
        custom_path = os.path.join(custom_dir, filename)
        cache_path = os.path.join(cache_subdir, filename)
        if os.path.isfile(custom_path) and not os.path.isfile(cache_path):
            shutil.copy(custom_path, cache_path)  # Copy missing model files

# Copy models to cache if necessary
copy_model_if_not_exists(custom_models_dir, cache_dir)

# Initialize DadmaTools pipeline with lemmatization and chunking
pipelines = 'lem,chunk'
nlp = language.Pipeline(pipelines=pipelines)

# Extract noun-adjective sequences from a text using the NLP pipeline
def extract_noun_adj_sequences(text):
    """Extracts noun-adjective sequences from text using NLP pipeline."""
    doc = nlp(text)
    noun_adj_sequences = []
    for sentence in doc['sentences']:
        tokens = sentence['tokens']
        current_sequence = []
        for token in tokens:
            if token['upos'] in ['NOUN', 'ADJ']:  # Check if the token is a noun or adjective
                current_sequence.append(token['lemma'])
            else:
                if len(current_sequence) > 1:  # Add sequence if it's more than 1 word
                    noun_adj_sequences.append(' '.join(current_sequence))
                current_sequence = []
        if len(current_sequence) > 1:  # Add remaining sequence
            noun_adj_sequences.append(' '.join(current_sequence))
    return noun_adj_sequences

# Perform Latent Dirichlet Allocation (LDA) to extract topics from text
def perform_lda(comments, num_topics=5, max_iter=2000):
    """Performs LDA topic modeling on comments."""
    vectorizer = CountVectorizer(
        max_df=0.7,  # Maximum document frequency
        min_df=max(2, len(comments) // 50),  # Minimum document frequency to prevent noise
        ngram_range=(1, 2),  # Bigrams and unigrams
        stop_words=stopwords,  # Stopwords removal
    )
    dtm = vectorizer.fit_transform(comments)  # Create Document-Term Matrix
    if dtm.shape[1] == 0:  # Ensure the DTM is not empty
        raise ValueError("No terms remain after pruning. Adjust min_df or max_df and try again.")
    lda = LatentDirichletAllocation(
        n_components=num_topics,  # Number of topics
        random_state=42,
        doc_topic_prior=0.5,
        topic_word_prior=0.5,
        max_iter=max_iter
    )
    lda.fit(dtm)  # Fit the LDA model
    return lda, vectorizer

# Get topic names and their importance (sum of words in topic)
def get_topic_names_and_importance(model, feature_names, no_top_words=10):
    """Extracts the most important words for each topic in the LDA model."""
    topics = {}
    for topic_idx, topic in enumerate(model.components_):
        topic_words = [feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]
        topics[' '.join(topic_words)] = topic.sum()  # Sum importance of the words in the topic
    return topics

# Function to process data for each room type and sentiment
def process_room_type(args):
    """Processes comments for each room type and sentiment, applies LDA, and returns the topics."""
    df, room_type, sentiment, sample_fraction = args  # Unpack arguments
    try:
        # Filter and sample comments based on room type and sentiment
        filtered_comments = df[(df['room_type'] == room_type) & (df['Sentiment'] == sentiment)]['comment'].dropna()
        sampled_comments = filtered_comments.sample(frac=sample_fraction, random_state=42)

        processed_comments = []
        for comment in sampled_comments:
            cleaned_comment = pre_clean_text(comment)
            if not cleaned_comment:
                continue
            noun_adj_sequences = extract_noun_adj_sequences(cleaned_comment)
            final_comment = ' '.join(noun_adj_sequences)
            if final_comment:
                processed_comments.append(final_comment)

        if processed_comments:
            lda_model, vectorizer = perform_lda(processed_comments)
            topics = get_topic_names_and_importance(lda_model, vectorizer.get_feature_names_out(), no_top_words=5)
            return room_type, sentiment, topics
        else:
            return room_type, sentiment, {}
    except Exception as e:
        print(f"Error processing room '{room_type}' and sentiment '{sentiment}': {e}")
        return room_type, sentiment, {}

# Main execution block
if __name__ == '__main__':
    start_time = time.time()

    # Load DataFrame from CSV file (adjust path as necessary)
    df = pd.read_csv("analysed_cm.csv")

    room_types = df['room_type'].unique()  # Get unique room types
    sentiments = ['HAPPY', 'SAD']  # Define sentiments to filter by
    sample_fraction = 1  # Fraction of comments to sample for testing

    # Prepare tasks for parallel processing
    tasks = [(df, room_type, sentiment, sample_fraction) for room_type in room_types for sentiment in sentiments]

    # Calculate the number of parallel jobs based on available system resources
    n_jobs = calculate_n_jobs()
    print(f"Using {n_jobs} parallel jobs for processing.")

    # Use multiprocessing Pool to process tasks
    with mp.Pool(processes=n_jobs) as pool:
        results = pool.map(process_room_type, tasks)

    # Save results to Excel
    with pd.ExcelWriter('output_room_topics.xlsx', engine='openpyxl') as writer:
        for room_type, sentiment, topics in results:
            if topics:
                topic_df = pd.DataFrame(topics.items(), columns=['Topic', 'Importance'])
                topic_df.to_excel(writer, sheet_name=f"{room_type}_{sentiment}")

    print("Execution Time: %.2f seconds" % (time.time() - start_time))
