import nltk
from gensim.models import KeyedVectors
import gensim.downloader as api
from matplotlib import pyplot as plt
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import contractions
import re
import string
import unidecode
import math
import pickle



word2vec_path = 'GoogleNews-vectors-negative300.bin'
def load_word2vec_model(word2vec_path):
    # use KeyedVectors to load the Word2Vec binary model
    word_vectors = KeyedVectors.load_word2vec_format(word2vec_path, binary=True, limit=20000)

    return word_vectors

def word2vec_to_dict(word_vectors):
    word2vec_dict = {}
    # iterate over each word vector in the Word2Vec model
    for word in word_vectors.key_to_index.keys():
        # get the word vector
        vector = word_vectors[word]
        # add the word and its corresponding vector to the dictionary
        word2vec_dict[word] = vector
    return word2vec_dict

def compute_avg_sentence_vector(sentence, word2vec_dict):
    # use the provided preprocessing function to process the sentence and get the tokens
    processed_words = preprocess_text(sentence)

    # collect the vectors of words that exist in word2vec_dict
    vectors = [word2vec_dict[word] for word in processed_words if word in word2vec_dict]

    # if no word vectors exist in word2vec_dict for the words in the sentence, return a zero vector
    if not vectors:
        return np.zeros(next(iter(word2vec_dict.values())).shape)

    # calculate and return the average of the vectors
    return np.mean(vectors, axis=0)

# Download necessary NLTK components
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize the lemmatizer which can be used to reduce words to their base form
lemmatizer = WordNetLemmatizer()

# Initialize the stop words
stop_words = set(stopwords.words('english'))

# Function to preprocess text
def preprocess_text(text):
    # Remove extra spaces
    text = re.sub(' +', ' ', text)
    # Convert to lowercase
    text = text.lower()
    # Expand contractions
    text = contractions.fix(text)
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove unicode characters
    text = unidecode.unidecode(text)
    # Tokenize the text
    tokens = word_tokenize(text)
    # Lemmatize the tokens and remove stop words
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
    return tokens

# Function to load the dataset
def load_dataset(filepath1, filepath2):
   train_dataset = pd.read_csv(filepath1, sep='\t', header=0)
   validation_dataset = pd.read_csv(filepath2, sep='\t', header=0)
   return train_dataset, validation_dataset


def generate_embedding_with_word2vec(df_train, df_validation, word2vec_model):
    dic = {}

    # let's process the queries and passages in the training and validation datasets
    queries = pd.concat([df_train[['qid', 'queries']], df_validation[['qid', 'queries']]]).drop_duplicates()
    passages = pd.concat([df_train[['pid', 'passage']], df_validation[['pid', 'passage']]]).drop_duplicates()

    # process queries
    for index, row in queries.iterrows():
        qid = row['qid']
        query = row['queries']
        # use
        vec = compute_avg_sentence_vector(query, word2vec_model)
        dic[qid] = vec

    # process passages
    for index, row in passages.iterrows():
        pid = row['pid']
        passage = row['passage']
        # use
        vec = compute_avg_sentence_vector(passage, word2vec_model)
        dic[pid] = vec

    # save the embeddings to a pickle file
    with open('limited20000_task2_embedding.pkl', 'wb') as file:
        pickle.dump(dic, file, protocol=pickle.HIGHEST_PROTOCOL)

    return dic

def main():
    # Step 1: Load training and validation datasets
    df_train, df_validation = load_dataset('train_data_subdataset.tsv', 'validation_data_subdataset.tsv')
    # Step 2: Load Word2Vec model
    print("Loading Word2Vec model...")
    word2vec_model = load_word2vec_model('GoogleNews-vectors-negative300.bin')
    word2vec_dict = word2vec_to_dict(word2vec_model)

    # Step 3: Generate embeddings for queries and passages
    print("Generating embeddings...")
    embedding_dict = generate_embedding_with_word2vec(df_train, df_validation, word2vec_dict)

if __name__ == "__main__":
    main()
