import os
import shutil
import pandas as pd
import dadmatools.pipeline.language as language
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import re
import time
from multiprocessing import Pool

# Start the timer
start_time = time.time()

# Load stopword file
stopword_file = 'Stopwords_shokristop_words.txt'
if not os.path.exists(stopword_file):
    raise FileNotFoundError(f"Stopword file '{stopword_file}' not found.")
with open(stopword_file, 'r', encoding='utf-8') as file:
    stopwords = [word.strip() for word in file if word.strip()]

# Pre-clean and preprocess text functions
def pre_clean_text(text):
    text = text.strip()
    if len(text.split()) < 3:  # Skip very short comments
        return ""
    return text

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

# Function to copy custom models if they do not exist
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

# Initialize the DadmaTools pipeline
pipelines = 'lem,chunk'
nlp = language.Pipeline(pipelines=pipelines)

# Load CSV file with comments
csv_file = 'analysed_cm.csv'
if not os.path.exists(csv_file):
    raise FileNotFoundError(f"CSV file '{csv_file}' not found.")
df = pd.read_csv(csv_file)

# Function to extract noun-adjective sequences from text
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

# Function to filter and process comments by room type
def process_comments_by_room_type(room_type, sentiment=None):
    if sentiment:
        filtered_comments = df[(df['room_type'] == room_type) & (df['Sentiment'] == sentiment)]['comment'].dropna()
    else:
        filtered_comments = df[df['room_type'] == room_type]['comment'].dropna()
    processed_comments = []
    for comment in filtered_comments:
        cleaned_comment = pre_clean_text(comment)
        if not cleaned_comment:
            continue
        noun_adj_sequences = extract_noun_adj_sequences(cleaned_comment)
        final_comment = preprocess_text(' '.join(noun_adj_sequences))
        if final_comment:
            processed_comments.append(final_comment)
    return processed_comments

# Function to perform LDA on comments
def perform_lda(comments, num_topics=6, max_iter=10000):
    vectorizer = CountVectorizer(
        max_df=0.6,
        min_df=max(2, len(comments) // 50),
        ngram_range=(1, 3),
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

# Function to get topic names and their importance
def get_topic_names_and_importance(model, feature_names, no_top_words=4):
    topics = {}
    for topic_idx, topic in enumerate(model.components_):
        topic_words = [feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]
        topics[' '.join(topic_words)] = topic.sum()
    return topics

# Main function to process and save LDA results by room type
def process_and_save_by_room_type(room_type):
    happy_comments = process_comments_by_room_type(room_type, sentiment='HAPPY')
    sad_comments = process_comments_by_room_type(room_type, sentiment='SAD')

    results = {}
    if happy_comments:
        lda, vectorizer = perform_lda(happy_comments)
        results['HAPPY'] = get_topic_names_and_importance(lda, vectorizer.get_feature_names_out())
    if sad_comments:
        lda, vectorizer = perform_lda(sad_comments)
        results['SAD'] = get_topic_names_and_importance(lda, vectorizer.get_feature_names_out())
    return room_type, results

# Main execution
if __name__ == '__main__':
    room_types = df['room_type'].unique()  # Get unique room types
    all_results = []

    # Use multiprocessing Pool to process each room type
    with Pool(processes=5) as pool:
        results = pool.map(process_and_save_by_room_type, room_types)

    # Save results to Excel
    with pd.ExcelWriter('output_topics_by_room_type2.xlsx', engine='openpyxl') as writer:
        for room_type, topics in results:
            for sentiment, topic_data in topics.items():
                df_topics = pd.DataFrame(topic_data.items(), columns=['Topic', 'Importance'])
                sheet_name = f'{room_type}_{sentiment}'
                df_topics.to_excel(writer, sheet_name=sheet_name, index=False)

    print("Execution completed in:", time.time() - start_time, "seconds")
