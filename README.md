# Persian LDA Topic Modeling

This repository contains a pipeline for processing Persian text data, performing sentiment analysis, and applying Latent Dirichlet Allocation (LDA) topic modeling to extract meaningful topics from customer feedback.

## Project Structure

The project is organized into different stages, each focusing on a particular task in the data pipeline:


### Project Workflow
The following are the main steps of the project pipeline:

1. Data Gathering
This step involves collecting raw Persian text data (e.g., customer reviews) that will be used for further processing. The data is located in the 1.Data Gathering directory.

2. Data Cleaning and Adjustment
The collected data needs to be cleaned and preprocessed. This includes:

Removing irrelevant characters.
Handling missing values.
Tokenizing and normalizing text.
Scripts related to data cleaning are found in the 2.Data Cleaning and Adjustment directory.

3. Model Creation
Here, the necessary models are created, including the setup of Dadmatools for text processing and the setup for LDA. The 3.Model Creation directory contains scripts for this phase.

4. Sentiment Analysis
Sentiment analysis is performed to classify customer feedback as "Happy" or "Sad". This step extracts important features from the data and processes the comments.

The sentiment analysis script is located in the 4.Sentiment Analysis folder.

5. LDA Topic Modeling
This stage applies the Latent Dirichlet Allocation (LDA) model to identify topics in the "Happy" and "Sad" comments. The model is fine-tuned using GridSearch for parameter optimization. The results are saved to LDA_Topics.xlsx.

LDA-related scripts are in the 5.LDA folder.

Running the Project
Once the setup is complete, you can run the following scripts:

Sentiment Analysis: To perform sentiment analysis on the data:

bash
Copy code
python 4.Sentiment Analysis/sentiment_analysis.py
LDA Topic Modeling: To apply LDA topic modeling and find topics in the feedback:

bash
Copy code
python 5.LDA/lda_modeling.py
Outputs
The output of the LDA topic modeling will be saved in LDA_Topics.xlsx, which includes the topics and the top words associated with each topic.
Sentiment analysis results will be outputted to a CSV or other format, depending on the script configuration.
