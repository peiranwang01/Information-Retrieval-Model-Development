import math
import numpy as np
import xgboost as xgb
import pandas as pd

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

def output_data_to_txt(relevancy_predict, qid_list, filename="LM.txt"):
    with open(filename, "w") as f:
        for qid in qid_list:
            # sort the pid of each qid in descending order based on the score, and get the top 100 (if exists)
            pid_score = sorted(relevancy_predict[qid].items(), key=lambda x: x[1], reverse=True)[:100]

            # iterate through each pid and its score, generate each line in the file
            for rank, (pid, score) in enumerate(pid_score, start=1):
                #   if the score is a numpy array, convert it to a float
                line = f"{qid} A2 {pid} {rank} {score} LM\n"
                f.write(line)



def evaluate_model(validation_data, y_predict):
    # create a dictionary to store the predicted relevance score for each qid and pid
    relevancy_predict = {}
    for i, row in validation_data.iterrows():
        qid = row['qid']
        pid = row['pid']
        if qid not in relevancy_predict:
            relevancy_predict[qid] = {}
        relevancy_predict[qid][pid] = y_predict[i]

    # get all qid list
    qid_list = list(relevancy_predict.keys())

    output_data_to_txt(relevancy_predict, qid_list)

    # prepare data for evaluation
    rank_rel = prepare_data("LM.txt")
    mAP = compute_mAP(rank_rel, 100)
    NDCG = compute_NDCG(rank_rel, 100)

    return mAP, NDCG

def optimize_parameter(x_train, y_train):

    traning_data = xgb.DMatrix(x_train, label=y_train)

    tune_params = [
        (max_depth, min_child_weight) for max_depth in range(1, 10) for min_child_weight in range(1, 10)
    ]

    initial_params = {'max_depth': 1,
              'min_child_weight': 1,
              'objective': 'rank:pairwise',
              'eval_metric': ["ndcg", "map"]}

    highest_mean_average_precision = 0.0
    optimal_settings = initial_params.copy()

    for max_depth, min_child_weight in tune_params:

        # update parameters of the model
        initial_params['max_depth'] = max_depth
        initial_params['min_child_weight'] = min_child_weight

        print(f"Testing with max_depth ={max_depth}, min_child_weight={min_child_weight}")

        # run model by CV
        cv_results = xgb.cv(
            initial_params,
            traning_data,
        )
        # Extract the highest MAP value
        mean_average_precision = cv_results['test-map-mean'].max()

        print(f"\tMean Average Precision (MAP): {mean_average_precision}")

        # Update the best parameters if current MAP is higher
        if mean_average_precision > highest_mean_average_precision:
            highest_mean_average_precision = mean_average_precision
            optimal_settings.update({'max_depth': max_depth, 'min_child_weight': min_child_weight})

    print(
        f"Optimal parameters: Depth={optimal_settings['max_depth']}, Child Weight={optimal_settings['min_child_weight']}, MAP: {highest_mean_average_precision}")
    # train to get the best model using the best parameters
    optimal_model = xgb.train(
        optimal_settings,
        traning_data,
    )

    return optimal_model


if __name__ == "__main__":
    # prepare train data, validation data for XGBoost model

    x_train_LR = np.load('x_train.npy')
    y_train_LR = np.load('y_train.npy')

    x_valid = np.load('x_valid.npy')
    y_valid = np.load('y_valid.npy')

    # tune pramaters and get predicted result based on the best model

    best_model = optimize_parameter(x_train_LR, y_train_LR)
    dvalid = xgb.DMatrix(x_valid, label=y_valid)
    y_predict = best_model.predict(dvalid)

    # evaluate model performance using mAP and NDCG

    validation_data = pd.read_csv('validation_data.tsv', sep='\t', header=0)
    mAP, NDCG = evaluate_model(validation_data, y_predict)

    print("Evaluation Result: ")
    print(f"The mAP value of the LM model is {mAP}.")
    print(f"The NDCG value of the LM model is {NDCG}.")