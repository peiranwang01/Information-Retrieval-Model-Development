import math

import numpy as np
import pandas as pd
import nltk
from gensim.models import KeyedVectors
from matplotlib import pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import contractions
import re
import string
import unidecode
import pickle

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

def negative_sampling(data, k):
    qid_list = np.unique(np.asarray(data['qid']))
    samples = []
    for qid in qid_list:
        pos_temp = data[(data['qid'] == qid) & (data['relevancy'] == 1)]
        neg_temp = data[(data['qid'] == qid) & (data['relevancy'] == 0)]
        samples.append(pos_temp.sample(n=1, random_state=1))
        if len(neg_temp) < k:
            samples.append(neg_temp)
        else:
            samples.append(neg_temp.sample(n=k, random_state=1))
    new_data = pd.concat(samples)
    return new_data.reset_index(drop=True)


# Function to load the dataset
def load_dataset(filepath1, filepath2):
   train_dataset = pd.read_csv(filepath1, sep='\t', header=0)
   validation_dataset = pd.read_csv(filepath2, sep='\t', header=0)
   return train_dataset, validation_dataset

def load_embedding_pickle():
    with open('limited20000_task2_embedding.pkl', 'rb') as f:
        embedding_dict = pickle.load(f)
    return embedding_dict

# realise logisticmodel
class LogisticRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.cost_history = []

    def _initialize_weights(self, n_features):
        self.weights = np.zeros(n_features)
        self.bias = 0

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def _compute_cost(self, y, predicted):
        # cost = -1 / n * sum(y * log(predicted) + (1 - y) * log(1 - predicted))
        n_samples = len(y)
        cost = -(1 / n_samples) * np.sum(y * np.log(predicted) + (1 - y) * np.log(1 - predicted))
        return cost

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self._initialize_weights(n_features)
        self.bias = 0

        for _ in range(self.n_iterations):
            model = np.dot(X, self.weights) + self.bias
            predicted = self._sigmoid(model)

            cost = -1 / n_samples * np.sum(y * np.log(predicted) + (1 - y) * np.log(1 - predicted))
            self.cost_history.append(cost)

            dw = (1 / n_samples) * np.dot(X.T, (predicted - y))
            db = (1 / n_samples) * np.sum(predicted - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict_proba(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        predicted_proba = self._sigmoid(linear_model)
        return predicted_proba.reshape(-1, 1)

    def predict(self, X, threshold=0.5):
        probabilities = self.predict_proba(X)
        return [1 if i > threshold else 0 for i in probabilities]

def prepare_data(model_ranked_file):
    # read the txt file, considering the file format is separated by space
    # because the output_data_to_txt function writes the score directly (not in list form), we don't need to do string processing on the score here
    rank = pd.read_csv(model_ranked_file, sep=" ", header=None,
                       names=['qid', 'unused', 'pid', 'rank', 'score', 'model'], usecols=[0, 2, 4])

    # ensure the data type of 'pid' and 'score' columns are correct
    rank['pid'] = rank['pid'].astype(int)
    rank['score'] = rank['score'].astype(float)

    # read the validation dataset
    validation_data = pd.read_csv('validation_data.tsv', sep='\t', header=0)

    # merge rank data and validation dataset based on 'qid' and 'pid'
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
            precisions = [data_per_query['relevancy'].iloc[:i + 1].mean() for i in
                          range(min(len(data_per_query), cutoff))]
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
        DCG = sum([(2 ** rel - 1) / math.log2(idx + 2) for idx, rel in enumerate(data_per_query['relevancy'])])
        IDCG = sum([(2 ** rel - 1) / math.log2(idx + 2) for idx, rel in
                    enumerate(sorted(data_per_query['relevancy'], reverse=True))])
        NDCG = DCG / IDCG if IDCG > 0 else 0
        NDCG_list.append(NDCG)
    avg_NDCG = np.mean(NDCG_list)
    return avg_NDCG


def output_data_to_txt(relevancy_predict, qid_list, filename="LR.txt"):
    with open(filename, "w") as f:
        for qid in qid_list:
            # according to the score, sort the pid of each qid in descending order and get the top 100 (if any)
            pid_score = sorted(relevancy_predict[qid].items(), key=lambda x: x[1], reverse=True)[:100]

            # iterate over each pid and its score, generate each line in the file
            for rank, (pid, score) in enumerate(pid_score, start=1):
                # define the string format to ensure there are six columns, with at least one space separating each column
                line = f"{qid} A2 {pid} {rank} {score} LR\n"
                f.write(line)


def evaluate_model(validation_data, y_predict):
    relevancy_predict = {}
    for i, row in validation_data.iterrows():
        qid = row['qid']
        pid = row['pid']
        if qid not in relevancy_predict:
            relevancy_predict[qid] = {}
        relevancy_predict[qid][pid] = y_predict[i]

    qid_list = list(relevancy_predict.keys())
    output_data_to_txt(relevancy_predict, qid_list)
    rank_rel = prepare_data("LR.txt")
    mAP = compute_mAP(rank_rel, 100)
    NDCG = compute_NDCG(rank_rel, 100)

    return mAP, NDCG

def process_model_input_with_word2vec(df_train, df_validation, embedding_dict):
    X_train = []
    y_train = df_train['relevancy'].values
    X_val = []
    y_val = df_validation['relevancy'].values

    # construct the input features for training data
    for index, row in df_train.iterrows():
        query_vector = embedding_dict.get(row['qid'], np.zeros(300))
        passage_vector = embedding_dict.get(row['pid'], np.zeros(300))
        X_train.append(np.concatenate([query_vector, passage_vector]))

    # build the input features for validation data
    for index, row in df_validation.iterrows():
        query_vector = embedding_dict.get(row['qid'], np.zeros(300))
        passage_vector = embedding_dict.get(row['pid'], np.zeros(300))
        X_val.append(np.concatenate([query_vector, passage_vector]))

    return np.array(X_train), y_train, np.array(X_val), y_val
def main():
    # Step 1: Load training and validation datasets
    df_train, df_validation = load_dataset('train_data.tsv', 'validation_data.tsv')

    print("Loading embedding dictionary...")
    embedding_dict = load_embedding_pickle()

    k = 10
    df_train_sampled = negative_sampling(df_train, k)

    # Step 4: Prepare model input
    print("Preparing model input...")
    X_train, y_train, X_val, y_val = process_model_input_with_word2vec(df_train_sampled, df_validation, embedding_dict)

    # save train and validation data for task4 model
    np.save('x_train.npy', X_train)
    np.save('y_train.npy', y_train)

    np.save('x_valid.npy', X_val)
    np.save('y_valid.npy', y_val)

    # Define learning rates to experiment with
    learning_rates = [2, 1, 0.1, 0.01, 0.001, 0.0001]

    # Iterate over learning rates, train and evaluate a model for each
    for lr in learning_rates:
        print(f"\nTraining with learning rate: {lr}")
        model = LogisticRegression(learning_rate=lr, n_iterations=1000)
        model.fit(X_train, y_train)

        # Predict on validation set
        y_pred = model.predict_proba(X_val)

        relevancy_predict = {qid: {} for qid in df_validation['qid'].unique()}
        for i, row in df_validation.iterrows():
            qid, pid = row['qid'], row['pid']
            relevancy_predict[qid][pid] = y_pred[i][0]  # 取概率预测的正类部分

        # Output the data to a file
        output_data_to_txt(relevancy_predict, df_validation['qid'].unique(), "LR.txt")
        rank_rel = prepare_data("NN.txt")
        mAP = compute_mAP(rank_rel, 100)
        NDCG = compute_NDCG(rank_rel, 100)

        print(f"mAP: {mAP}", f"NDCG: {NDCG}")

        final_loss = model.cost_history[-1]

        print(f"Final training loss for learning rate {lr}: {final_loss}")

        plt.plot(model.cost_history, label=f'LR={lr}')

    plt.title('Training Loss vs. Number of Iterations')
    plt.xlabel('Number of Iterations')
    plt.ylabel('Training Loss')
    plt.legend()
    plt.show()

    model_filename = 'logistic_regression_model.pkl'
    with open(model_filename, 'wb') as file:
        pickle.dump(model, file)

if __name__ == "__main__":
    main()

