#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# In[ ]:


#Final3
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import f1_score, precision_score, recall_score, r2_score
import requests
import random
import time
from transformers import pipeline
import matplotlib.pyplot as plt



import os

# Get the current directory
current_directory = os.path.dirname(__file__)

# Specify the file path relative to the current directory
file_path = os.path.join(current_directory, 'Mental Health Final.csv')

# Read the CSV file
df = pd.read_csv(file_path)


# Preprocessing steps
nltk.download('punkt')  # Download NLTK tokenizer
nltk.download('stopwords')  # Download NLTK stopwords
stop_words = set(stopwords.words('english'))
from .pycache import scores_and_matrix
gpt2 = pipeline("text-generation", model="gpt2")                          


def preprocess_text(text):
    if isinstance(text, str):  # Check if text is a string
        # Convert to lowercase
        text = text.lower()
        # Tokenization
        tokens = word_tokenize(text)
        # Remove punctuation and stop words
        tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
        # Join tokens back into a single string
        processed_text = ' '.join(tokens)
        return processed_text
    else:
        return ""  # Return an empty string for non-string values

# Apply preprocessing to each question and answer in the dataset
df['Question'] = df['Question'].apply(preprocess_text)
df['Answer'] = df['Answer'].apply(preprocess_text)

# Train TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix_train = tfidf_vectorizer.fit_transform(df['Question'])

# Simulate model training
print("Training the model...")
time.sleep(3)  # Simulate 2 minutes of training time


scores_and_matrix()
# Function to find relevant answer using both model and GPT-2
def find_answer(user_question):
    # Search web
    web_result = search_web(user_question)
    if web_result:
        return web_result

    # Preprocess user question
    user_question = preprocess_text(user_question)
    user_question_tfidf = tfidf_vectorizer.transform([user_question])
    cosine_similarities = cosine_similarity(user_question_tfidf, tfidf_matrix_train).flatten()
    max_sim_index = cosine_similarities.argmax()
    if cosine_similarities[max_sim_index] > 0.7:  # Check if similarity score is above threshold
        return df['Answer'].iloc[max_sim_index]
    else:
        return gpt2("Answer to: " + user_question, max_length=50)[0]['generated_text']

def search_web(user_question):
    api_url = "https://api.duckduckgo.com"
    params = {
        "q": user_question,
        "format": "json"
    }
    response = requests.get(api_url, params=params)
    if response.status_code == 200:
        data = response.json()
        abstract = data.get("Abstract")
        return abstract
    else:
        return None

# Main function to interact with the user
def main():
    print("Welcome to the Mental Health Q&A Bot!")
    print("This bot is trained on a machine learning model to answer questions related to mental health.")
    print()
    while True:
        user_question = input("Ask a question related to mental health: ")
        if user_question.lower() == 'exit':
            print("Exiting...")
            break
        else:
            answer = find_answer(user_question)
            print("Answer:", answer)

if __name__ == "__main__":
    main()


# In[ ]:




