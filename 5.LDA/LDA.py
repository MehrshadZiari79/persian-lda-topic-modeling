import os
import shutil
import pandas as pd
import dadmatools.pipeline.language as language
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics.pairwise import cosine_similarity
import re
import numpy as np
import matplotlib.pyplot as plt
from bidi.algorithm import get_display
import arabic_reshaper
import time
from multiprocessing import Pool

# Start the timer for execution timing
start_time = time.time()

# Load stopwords
stopword_file = 'Stopwords_shokristop_words.txt'
if not os.path.exists(stopword_file):
    raise FileNotFoundError(f"Stopword file '{stopword_file}' not found.")
with open(stopword_file, 'r', encoding='utf-8') as file:
    stopwords = [word.strip() for word in file if word.strip()]

# Text cleaning function to remove invalid or short comments
def pre_clean_text(text):
    text = text.strip()  # Remove leading/trailing whitespace
    if len(text.split()) < 3:  # Skip comments with fewer than 3 words
        return ""
    return text

# Text preprocessing function: replaces characters, removes unwanted elements
def preprocess_text(text):
    replacements = {
        'ی': 'ي',
        'ک': 'ك',
        '\u0640': ''  # Remove Tatweel
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = re.sub(r'\u200c', '', text)  # Remove ZWNJ
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\d+', '', text)  # Remove digits
    text = re.sub(r'[A-Za-z]', '', text)  # Remove Latin characters
    words = [word for word in text.split() if word not in stopwords]
    return ' '.join(words)

# Ensure DadmaTools models are available in the cache directory
custom_models_dir = 'E:\\payan\\WORDPR\\dadma\\xlm-roberta-base'
cache_dir = os.path.join(os.path.expanduser("~"), "cache", "dadmatools")

def copy_model_if_not_exists(custom_dir, cache_subdir):
    if not os.path.exists(cache_subdir):
        os.makedirs(cache_subdir)
    for filename in os.listdir(custom_dir):
        custom_path = os.path.join(custom_dir, filename)
        cache_path = os.path.join(cache_subdir, filename)
        if os.path.isfile(custom_path) and not os.path.isfile(cache_path):
            shutil.copy(custom_path, cache_path)

copy_model_if_not_exists(custom_models_dir, cache_dir)

# Initialize DadmaTools pipeline
pipelines = 'lem,chunk'
nlp = language.Pipeline(pipelines=pipelines)

# Read CSV file containing comments
csv_file = 'analysed_cm.csv'
if not os.path.exists(csv_file):
    raise FileNotFoundError(f"CSV file '{csv_file}' not found.")
df = pd.read_csv(csv_file)

# Extract noun and adjective sequences using DadmaTools pipeline
def extract_noun_adj_sequences(text):
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

# Process comments for a specific sentiment
def process_comments(sentiment, sample_fraction=1):
    try:
        filtered_comments = df[df['Sentiment'] == sentiment]['comment'].dropna().sample(frac=sample_fraction, random_state=42)
        processed_comments = []
        for comment in filtered_comments:
            cleaned_comment = pre_clean_text(comment)
            if not cleaned_comment:
                continue  # Skip invalid text
            noun_adj_sequences = extract_noun_adj_sequences(cleaned_comment)
            final_comment = preprocess_text(' '.join(noun_adj_sequences))
            if final_comment:
                processed_comments.append(final_comment)
        return processed_comments
    except Exception as e:
        print(f"Error processing comments for sentiment '{sentiment}': {e}")
        return []

# Perform LDA topic modeling
def perform_lda(comments, num_topics=6, max_iter=5000):
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

# Retrieve topic names and their importance
def get_topic_names_and_importance(model, feature_names, no_top_words=10):
    topics = {}
    for topic_idx, topic in enumerate(model.components_):
        topic_words = [feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]
        topics[' '.join(topic_words)] = topic.sum()
    return topics

# Process comments and apply LDA
def process_and_lda(sentiment):
    comments = process_comments(sentiment)
    lda, vectorizer = perform_lda(comments)
    if lda and vectorizer:
        topics = get_topic_names_and_importance(lda, vectorizer.get_feature_names_out(), 4)
        return topics
    return None

# Main block for multiprocessing
if __name__ == '__main__':
    with Pool(processes=7) as pool:
        results = pool.map(process_and_lda, ['HAPPY', 'SAD'])

    happy_topics = results[0]
    sad_topics = results[1]

    # Print and visualize topics for both sentiments
    if happy_topics:
        print("\nHAPPY Topics:")
        for topic, importance in happy_topics.items():
            print(f"Topic: {topic}, Importance: {importance}")

    if sad_topics:
        print("\nSAD Topics:")
        for topic, importance in sad_topics.items():
            print(f"Topic: {topic}, Importance: {importance}")

    # Visualization for topics
    reshaped_happy_topics = [get_display(arabic_reshaper.reshape(topic)) for topic in happy_topics.keys()]
    reshaped_sad_topics = [get_display(arabic_reshaper.reshape(topic)) for topic in sad_topics.keys()]

    plt.barh(reshaped_happy_topics, list(happy_topics.values()), color='skyblue')
    plt.title("HAPPY Topics")
    plt.xlabel("Importance")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig("happy_topics_visual.png", dpi=300)
    plt.show()

    plt.barh(reshaped_sad_topics, list(sad_topics.values()), color='salmon')
    plt.title("SAD Topics")
    plt.xlabel("Importance")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig("sad_topics_visual.png", dpi=300)
    plt.show()

    # Save topics to an Excel file
    with pd.ExcelWriter('output_topics.xlsx', engine='openpyxl') as writer:
        if happy_topics:
            pd.DataFrame(happy_topics.items(), columns=['Topic', 'Importance']).to_excel(writer, sheet_name='HAPPY', index=False)
        if sad_topics:
            pd.DataFrame(sad_topics.items(), columns=['Topic', 'Importance']).to_excel(writer, sheet_name='SAD', index=False)

    print("Execution completed in:", time.time() - start_time, "seconds")
