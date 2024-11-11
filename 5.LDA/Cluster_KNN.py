import os
import shutil
import pandas as pd
import dadmatools.pipeline.language as language
from transformers import AutoModel, AutoTokenizer
import torch
from sklearn.cluster import KMeans
import numpy as np
from collections import Counter
import re
import openpyxl
from bidi.algorithm import get_display
import arabic_reshaper
import matplotlib.pyplot as plt

# Load stopword list from a text file
with open('Stopwords_shokristop_words.txt', 'r', encoding='utf-8') as file:
    stopwords = file.read().splitlines()

# Define custom path to downloaded models
custom_models_dir = ''
cache_dir = os.path.join(os.path.expanduser("~"), "cache", "dadmatools")

# Function to copy models if they don't exist in the cache
def copy_model_if_not_exists(custom_dir, cache_subdir):
    if not os.path.exists(cache_subdir):
        os.makedirs(cache_subdir)
    for filename in os.listdir(custom_dir):
        custom_path = os.path.join(custom_dir, filename)
        cache_path = os.path.join(cache_subdir, filename)
        if os.path.isdir(custom_path):
            continue
        if not os.path.isfile(cache_path):
            print(f"Copying {filename} to cache...")
            shutil.copy(custom_path, cache_path)
        else:
            print(f"Model {filename} already exists in cache.")

copy_model_if_not_exists(custom_models_dir, cache_dir)

# Initialize DadmaTools pipeline for tokenization and noun extraction
pipelines = 'lem,chunk'
nlp = language.Pipeline(pipelines=pipelines)

# Initialize the Transformer model and tokenizer
model_name = "xlm-roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Read CSV file and filter comments by sentiment
df = pd.read_csv('analysed_cm.csv')

# Function to clean and preprocess text
def preprocess_text(text):
    text = re.sub(r'\u200c', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[A-Za-z]', '', text)
    words = [word for word in text.split() if word not in stopwords]
    return ' '.join(words)

# Function to extract nouns from text
def extract_nouns(text):
    doc = nlp(text)
    nouns = [token['text'] for sentence in doc['sentences'] for token in sentence['tokens'] if token['upos'] == 'NOUN']
    return ' '.join(nouns)

# Preprocess and extract nouns for each comment
def process_comments(sentiment, sample_fraction=0.1):
    filtered_comments = df[df['Sentiment'] == sentiment]['comment'].dropna().sample(frac=sample_fraction, random_state=42)
    processed_comments = []
    for comment in filtered_comments:
        nouns = extract_nouns(comment)
        cleaned_comment = preprocess_text(nouns)
        processed_comments.append(cleaned_comment)
    return processed_comments

# Generate embeddings for each comment using the Transformer model
def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    outputs = model(**inputs)
    embedding = outputs.last_hidden_state.mean(dim=1)  # Mean pooling
    return embedding.squeeze().detach().numpy()

# Get embeddings for each preprocessed comment
def process_comments_embeddings(sentiment):
    comments = process_comments(sentiment)
    embeddings = [get_embedding(comment) for comment in comments]
    return np.array(embeddings), comments

# Perform KMeans clustering on the embeddings
def perform_kmeans_clustering(embeddings, num_topics=6):
    kmeans = KMeans(n_clusters=num_topics, random_state=42)
    clusters = kmeans.fit_predict(embeddings)
    return kmeans, clusters

# Get the top words in each cluster
def get_top_words_per_cluster(comments, clusters, num_words=5):
    cluster_topics = {}
    for i in range(len(set(clusters))):
        cluster_comments = [comments[idx] for idx, cluster in enumerate(clusters) if cluster == i]
        words = ' '.join(cluster_comments).split()
        word_counts = Counter(words)
        top_words = [word for word, _ in word_counts.most_common(num_words)]
        cluster_topics[f"Cluster {i + 1}"] = top_words
    return cluster_topics

# Process embeddings and cluster them for both sentiments
happy_embeddings, happy_comments = process_comments_embeddings('HAPPY')
sad_embeddings, sad_comments = process_comments_embeddings('SAD')

# Perform clustering
happy_kmeans, happy_clusters = perform_kmeans_clustering(happy_embeddings)
sad_kmeans, sad_clusters = perform_kmeans_clustering(sad_embeddings)

# Get top words per cluster
happy_topic_words = get_top_words_per_cluster(happy_comments, happy_clusters)
sad_topic_words = get_top_words_per_cluster(sad_comments, sad_clusters)

# Display and save the topics
print("Happy Topics:", happy_topic_words)
print("Sad Topics:", sad_topic_words)

# Prepare data for visualization and Excel output
def save_and_visualize_topics(topics, sentiment):
    sorted_topics = sorted(topics.items(), key=lambda x: -len(x[1]))  # Sort by word frequency
    topic_names, topic_words = zip(*sorted_topics)

    # Reshape topic names for display
    reshaped_topics = [get_display(arabic_reshaper.reshape(' '.join(words))) for words in topic_words]

    # Save results to Excel
    df = pd.DataFrame({'Topic': topic_names, 'Top Words': reshaped_topics})
    with pd.ExcelWriter("separated_topic_importance.xlsx", mode='a', if_sheet_exists="replace") as writer:
        df.to_excel(writer, sheet_name=f'{sentiment}_Topics', index=False)

    # Plot topics
    fig, ax = plt.subplots(figsize=(10, len(reshaped_topics) * 0.7))
    ax.barh(range(len(reshaped_topics)), [len(words) for words in topic_words], color='skyblue')
    ax.set_yticks(range(len(reshaped_topics)))
    ax.set_yticklabels(reshaped_topics)
    ax.set_title(f"Important Topics in {sentiment} Comments")
    ax.invert_yaxis()  # Highest importance at the top
    plt.tight_layout()
    plt.savefig(f"{sentiment.lower()}_topic_importance.png", format="png", dpi=300)
    plt.show()

# Save and visualize topics for happy and sad comments
save_and_visualize_topics(happy_topic_words, 'Happy')
save_and_visualize_topics(sad_topic_words, 'Sad')
#Happy Topics: {'Cluster 1': ['نظافت', 'برخورد', 'نظافتش', 'امکانات', 'موقعیت'], 'Cluster 2': ['وقت'], 'Cluster 3': ['برخورد', 'امکانات', 'تمیز', 'رفتار', 'محیط'], 'Cluster 4': ['برخورد', 'تمیز', 'قیمت', 'امکانات', 'دسترسی'], 'Cluster 5': ['برخورد', 'امکانات', 'نظافت', 'حیاط', 'رفتار'], 'Cluster 6': ['برخورد', 'دسترسی', 'موقعیت', 'محیط', 'قیمت']}
#Sad Topics: {'Cluster 1': ['قیمت', 'سرویس', 'پارکینگ', 'عکس', 'جاده'], 'Cluster 2': ['نظافت', 'سرویس', 'آب', 'تحویل', 'تخت'], 'Cluster 3': [], 'Cluster 4': ['نظافت', 'تحویل', 'برخورد', 'عکس', 'کثیف'], 'Cluster 5': ['آب', 'وسایل', 'نظافت', 'برخورد', 'کثیف'], 'Cluster 6': ['کثیف', 'نظافت', 'تمیز', 'امکانات', 'آب']}
