import math
import os

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# define the model input
inputs = Input(shape=(600,))  # modify the input shape to match the data

# model layers
x = Embedding(input_dim=10000, output_dim=128)(inputs)
x = Conv1D(filters=128, kernel_size=5, activation='relu')(x)
x = GlobalMaxPooling1D()(x)
outputs = Dense(1)(x)

# create the model
model = Model(inputs=inputs, outputs=outputs)
model.compile(loss='mean_squared_error', optimizer='adam')

# load and preprocess the data (assuming the data has been preprocessed into integer sequences)
x_train = np.load('x_train.npy')
y_train = np.load('y_train.npy')
x_valid = np.load('x_valid.npy')
y_valid = np.load('y_valid.npy')

# adjust the length of the input sequences to match the input shape of the model
x_train = pad_sequences(x_train, maxlen=600)
x_valid = pad_sequences(x_valid, maxlen=600)

# train the model
model.fit(x_train, y_train, epochs=5, batch_size=32, validation_data=(x_valid, y_valid))

# predict
y_predict = model.predict(x_valid)

def prepare_data(model_ranked_file):
    # read the txt file, considering the file format is separated by space
    # since the output_data_to_txt function writes the score directly (not in list form), we don't need to do string processing for the score here
    rank = pd.read_csv(model_ranked_file, sep=" ", header=None,
                       names=['qid', 'unused', 'pid', 'rank', 'score', 'model'], usecols=[0, 2, 4])

    # ensure the data type of 'pid' and 'score' columns are correct
    rank['pid'] = rank['pid'].astype(int)
    rank['score'] = rank['score'].astype(float)

    # read the validation data
    validation_data = pd.read_csv('validation_data.tsv', sep='\t', header=0)

    # merge rank and validation_data based on 'qid' and 'pid'
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

def output_data_to_txt(relevancy_predict, qid_list, filename="NN.txt"):
    with open(filename, "w") as f:
        for qid in qid_list:
            pid_score = sorted(relevancy_predict[qid].items(), key=lambda x: x[1], reverse=True)[:100]
            for rank, (pid, score) in enumerate(pid_score, start=1):
                if isinstance(score, np.ndarray):
                    score = score[0]
                line = f"{qid} A2 {pid} {rank} {score} NN\n"
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
    rank_rel = prepare_data("NN.txt")
    mAP = compute_mAP(rank_rel, 100)
    NDCG = compute_NDCG(rank_rel, 100)

    return mAP, NDCG


save_path = 'predictions'
if not os.path.exists(save_path):
    os.makedirs(save_path)
save_file = os.path.join(save_path, 'y_predict.npy')

# save the predicted scores
np.save(save_file, y_predict)

# load the predicted scores
y_predict_loaded = np.load(save_file)
# evaluate the model
validation_data = pd.read_csv('validation_data.tsv', sep='\t', header=0)
mAP, NDCG = evaluate_model(validation_data, y_predict)

print("Evaluation Result:")
print(f"The mAP value of the NN model is {mAP}.")
print(f"The NDCG value of the NN model is {NDCG}.")