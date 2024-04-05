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
def load_dataset(filepath):
    # Load the dataset
    df = pd.read_csv(filepath, sep='\t', header=0)
    # extract the query including the query id and the query text
    df_query = df[['qid', 'queries']].drop_duplicates().reset_index(drop=True)
    # extract the passage including the passage id and the passage text
    df_passage = df[['pid', 'passage']].drop_duplicates().reset_index(drop=True)
    return df, df_query, df_passage

# Function to create the inverted index
def build_inverted_index(queries, passages):
    vocabularies = {}
    total_terms = 0
    for passage in passages['passage']:
        terms = preprocess_text(passage)
        total_terms += len(terms)
        unique_terms = set(terms)
        for single_term in unique_terms:
            vocabularies[single_term] = vocabularies.get(single_term, 0) + 1
    for query in queries['queries']:
        terms = preprocess_text(query)
        total_terms += len(terms)
        unique_terms = set(terms)
        for single_term in unique_terms:
            vocabularies[single_term] = vocabularies.get(single_term, 0) + 1
    N = len(passages)
    avg_len = total_terms / N
    for single_term, ni in vocabularies.items():
        idf = math.log((N - ni + 0.5) / (ni + 0.5) + 1)
        vocabularies[single_term] = (ni, idf)
    return vocabularies, avg_len

# Function to compute the BM25 score
def compute_bm25_model_scores(query, passage, vocabularies, avg_len, k1=1.2, k2 = 100, b=0.75):
    q_t = preprocess_text(query)
    p_t = preprocess_text(passage)
    score = 0
    for query_term in q_t:
        if query_term in vocabularies:
            ni, idf = vocabularies[query_term]
            f = p_t.count(query_term)
            doc_len = len(p_t)
            K = k1 * ((1 - b) + b * (doc_len / avg_len))
            score += idf * ((f * (k1 + 1)) / (f + K)) * ((1 + k2) / (1 + k2))
    return score

# function to compute average precision
def prepare_data(model_ranked_file):

    try:
        rank = pd.read_csv(model_ranked_file, header=None, sep='\t')
    except Exception as e:
        print(f"error when reading file: {e}")
        return None

    # check whether the DataFrame contains the expected three columns
    if len(rank.columns) != 3:
        print(f"Wrong {len(rank.columns)} 列。")
        return None

    # set the column names
    rank.columns = ['qid', 'pid', 'score']

    # read the validation data
    validation_data = pd.read_csv('validation_data.tsv', sep='\t', header=0)

    # according to qid and pid, merge rank and validation_data
    rank_rel = pd.merge(rank, validation_data, how="left", on=["qid", "pid"])
    return rank_rel


def compute_mAP(rank_rel, cutoff):
    avg_precision_list = []
    qid_list = rank_rel['qid'].unique()
    for qid in qid_list:
        data_per_query = rank_rel[rank_rel['qid'] == qid]
        data_per_query = data_per_query.sort_values(by='score', ascending=False).iloc[:cutoff]
        if len(data_per_query) == 0 or data_per_query['relevancy'].sum() == 0:
            avg_precision = 0
        else:
            precisions = [data_per_query['relevancy'].iloc[:i+1].mean() for i in range(min(len(data_per_query), cutoff))]
            avg_precision = np.mean([p for p, rel in zip(precisions, data_per_query['relevancy']) if rel > 0])
        avg_precision_list.append(avg_precision)
    mAP = np.mean(avg_precision_list)
    return mAP

def compute_NDCG(rank_rel, cutoff):
    NDCG_list = []
    qid_list = rank_rel['qid'].unique()
    for qid in qid_list:
        data_per_query = rank_rel[rank_rel['qid'] == qid]
        data_per_query = data_per_query.sort_values(by='score', ascending=False).iloc[:cutoff]
        DCG = sum([(2**rel - 1) / math.log2(idx + 2) for idx, rel in enumerate(data_per_query['relevancy'])])
        IDCG = sum([(2**rel - 1) / math.log2(idx + 2) for idx, rel in enumerate(sorted(data_per_query['relevancy'], reverse=True))])
        NDCG = DCG / IDCG if IDCG > 0 else 0
        NDCG_list.append(NDCG)
    avg_NDCG = np.mean(NDCG_list)
    return avg_NDCG

# use bm25 model to validate the dataset
def validate_bm25_model(df, df_query, df_passage, vocabularies, avg_len, output_filename="NN.txt"):
    # save the scores of all queries and passages
    all_scores = []

    for index, row in df_query.iterrows():
        qid = row['qid']
        query = row['queries']
        passages = df[df['qid'] == qid].copy()

        # compute the BM25 model scores for each query and passage pair
        passages['bmscore'] = passages.apply(
            lambda x: compute_bm25_model_scores(query, x['passage'], vocabularies, avg_len=avg_len), axis=1)

        for _, passage_row in passages.iterrows():
            all_scores.append((qid, passage_row['pid'], passage_row['bmscore']))

    # transform the scores to a DataFrame and save it to a file
    scores_df = pd.DataFrame(all_scores, columns=['qid', 'pid', 'score'])
    scores_df.to_csv(output_filename, index=False, sep="\t", header=False)

    # prepare the data for evaluation
    rank_rel = prepare_data(output_filename)
    mAP = compute_mAP(rank_rel, 100)
    NDCG = compute_NDCG(rank_rel, 100)

    return mAP, NDCG

def main():
    # Load the dataset
    df, df_query, df_passage = load_dataset('validation_data.tsv')
    # Build the inverted index
    vocabularies, avg_len = build_inverted_index(df_query, df_passage)
    # Validate the BM25 model
    mAP, NDCG = validate_bm25_model(df, df_query, df_passage, vocabularies, avg_len, "ranked_results.txt")

    print("Evaluation Result:")
    print(f"The mAP value of the BM25 model is {mAP}.")
    print(f"The NDCG value of the BM25 model is {NDCG}.")

if __name__ == "__main__":
    main()