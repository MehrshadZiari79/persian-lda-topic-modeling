import shutil  # Used for file operations like copying
import dadmatools.pipeline.language as language  # NLP tools
import re  # Regular expressions for text cleaning
import os  # File and directory operations
import pandas as pd  # Data handling
from sklearn.feature_extraction.text import CountVectorizer  # For text feature extraction
from sklearn.decomposition import LatentDirichletAllocation  # For LDA topic modeling
from sklearn.metrics.pairwise import cosine_similarity  # For similarity metrics
import matplotlib.pyplot as plt  # For visualizations
import arabic_reshaper  # Arabic text reshaping for display
from bidi.algorithm import get_display  # BiDi algorithm for proper Arabic display
import multiprocessing as mp  # Parallel processing
import time  # Timing execution
import psutil  # System resource monitoring

# Timer to measure the execution time
start_time = time.time()

# Define the path to the stopwords file
stopword_file = 'Stopwords_shokristop_words.txt'

# Check if the stopword file exists
if not os.path.exists(stopword_file):
    raise FileNotFoundError(f"Stopword file '{stopword_file}' not found.")

# Read stopwords from the file
with open(stopword_file, 'r', encoding='utf-8') as file:
    stopwords = [word.strip() for word in file if word.strip()]

# Pre-clean text to remove unnecessary spaces or too-short comments
def pre_clean_text(text):
    """Removes extra spaces and skips very short comments."""
    text = text.strip()  # Remove leading/trailing whitespace
    if len(text.split()) < 3:  # Skip comments that are too short
        return ""
    return text

# Preprocess and clean the text
def preprocess_text(text):
    """Cleans and normalizes Persian text."""
    replacements = {
        'ی': 'ي',  # Normalize Persian 'ye' character
        'ک': 'ك',  # Normalize Persian 'kaf' character
        '\u0640': ''  # Remove Tatweel (Arabic script elongation)
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = re.sub(r'\u200c', '', text)  # Remove ZWNJ
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\d+', '', text)  # Remove digits
    text = re.sub(r'[A-Za-z]', '', text)  # Remove Latin characters
    words = [word for word in text.split() if word not in stopwords]
    return ' '.join(words)

# Paths for custom models and cache
custom_models_dir = 'E:\\payan\\WORDPR\\dadma\\xlm-roberta-base'
cache_dir = os.path.join(os.path.expanduser("~"), "cache", "dadmatools")

# Determine optimal parallel jobs
def calculate_n_jobs():
    """Calculates the number of parallel processes based on CPU cores and available memory."""
    cpu_cores = mp.cpu_count()
    memory_per_process = 500  # Estimated memory per process (in MB)
    available_memory = psutil.virtual_memory().available / 1024 / 1024  # Convert to MB
    max_processes_by_memory = int(available_memory / memory_per_process)
    n_jobs = min(cpu_cores - 1, max_processes_by_memory)  # Use all but 1 core
    return int(n_jobs * 0.9)  # Use 90% of capacity

# Copy model files to cache if necessary
def copy_model_if_not_exists(custom_dir, cache_subdir):
    """Copies model files from the custom directory to the cache if they are missing."""
    if not os.path.exists(cache_subdir):
        os.makedirs(cache_subdir)
    for filename in os.listdir(custom_dir):
        custom_path = os.path.join(custom_dir, filename)
        cache_path = os.path.join(cache_subdir, filename)
        if os.path.isfile(custom_path) and not os.path.isfile(cache_path):
            shutil.copy(custom_path, cache_path)

# Copy models to cache
copy_model_if_not_exists(custom_models_dir, cache_dir)

# Initialize DadmaTools pipeline
pipelines = 'lem,chunk'
nlp = language.Pipeline(pipelines=pipelines)

# Extract noun-adjective sequences
def extract_noun_adj_sequences(text):
    """Extracts noun-adjective sequences using NLP pipeline."""
    doc = nlp(text)
    noun_adj_sequences = []
    for sentence in doc['sentences']:
        tokens = sentence['tokens']
        current_sequence = []
        for token in tokens:
            if token['upos'] in ['NOUN', 'ADJ']:
                current_sequence.append(token['lemma'])
            else:
                if len(current_sequence) > 1:
                    noun_adj_sequences.append(' '.join(current_sequence))
                current_sequence = []
        if len(current_sequence) > 1:
            noun_adj_sequences.append(' '.join(current_sequence))
    return noun_adj_sequences

# Perform LDA
def perform_lda(comments, num_topics=4, max_iter=2000):
    """Performs LDA topic modeling."""
    vectorizer = CountVectorizer(
        max_df=0.6,
        min_df=max(2, len(comments) // 50),
        ngram_range=(1, 2),
        stop_words=stopwords,
    )
    dtm = vectorizer.fit_transform(comments)
    if dtm.shape[1] == 0:
        raise ValueError("No terms remain after pruning. Adjust min_df or max_df and try again.")
    lda = LatentDirichletAllocation(
        n_components=num_topics,
        random_state=42,
        doc_topic_prior=0.5,
        topic_word_prior=0.5,
        max_iter=max_iter
    )
    lda.fit(dtm)
    return lda, vectorizer

# Extract topic importance
def get_topic_names_and_importance(model, feature_names, no_top_words=2):
    """Extracts the most important words for each topic in LDA model."""
    topics = {}
    for topic_idx, topic in enumerate(model.components_):
        topic_words = [feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]
        topics[' '.join(topic_words)] = topic.sum()
    return topics

# Process data
def process_room_type(args):
    """Processes room type data."""
    df, room_type, sentiment, sample_fraction = args
    try:
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
            topics = get_topic_names_and_importance(lda_model, vectorizer.get_feature_names_out(), no_top_words=2)
            return room_type, sentiment, topics
        else:
            return room_type, sentiment, {}
    except Exception as e:
        print(f"Error processing room '{room_type}' and sentiment '{sentiment}': {e}")
        return room_type, sentiment, {}

# Main block
if __name__ == '__main__':
    df = pd.read_csv("analysed_cm.csv")
    room_types = df['room_type'].unique()
    sentiments = ['HAPPY', 'SAD']
    sample_fraction = 1

    tasks = [(df, room_type, sentiment, sample_fraction) for room_type in room_types for sentiment in sentiments]
    n_jobs = calculate_n_jobs()
    print(f"Using {n_jobs} parallel jobs for processing.")

    with mp.Pool(processes=n_jobs) as pool:
        results = pool.map(process_room_type, tasks)

    with pd.ExcelWriter('output_room_topics.xlsx', engine='openpyxl') as writer:
        for room_type, sentiment, topics in results:
            if topics:
                topic_df = pd.DataFrame(topics.items(), columns=['Topic', 'Importance'])
                topic_df.to_excel(writer, sheet_name=f"{room_type}_{sentiment}")

    print("Execution Time: %.2f seconds" % (time.time() - start_time))
