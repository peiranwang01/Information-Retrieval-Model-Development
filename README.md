## RUN GUIDE

There are 5 python files to solve all the subtasks.

You can directly run task1.py with all given dataset ready.

For subtask 2, you need to download the dataset from https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?resourcekey=0-wjGZdNAUop6WykTtMip30g
And, you need to run task2_generate_embedding.py first, and then run the task2_model_evaluation.py to complete the task.

Task 3 and Task 4 may require you to install some libraries needed, it may take some time. Then you can run them directly with all given dataset ready.

## Project Description

This project is to develop an information retrieval model that solves the problem of passage retrieval, i.e., a model that can effectively and efficiently return a ranked list of short texts (i.e. passages) relevant to a given query. 

###### Data URL: https://drive.google.com/file/d/1eKDfmDZoVuDADcR_HGMHMnjNJHDrXUs9/view

###### Data Description:

- est-queries.tsv is a tab separated file containing the queries in the test set, where each row contains a query ID (qid) and the query (i.e., query text).
- candidate_passages_top1000.tsv is a tab separated file, containing initial rankings that contain 1000 passages for each of the given queries (as it was in the first part of the assignment) in file test-queries.tsv. The format of this file is <qid pid query passage>, where qid is the query ID, pid is the ID of the passage retrieved, query is the query text and passage is the passage text, all tab separated. Figure 1 shows some sample rows from the file.
- train_data.tsv and validation_data.tsv. These are the datasets you will be using for training and validation. You are expected to train your model on the training set and evaluate your models’ performance on the validation set. In these datasets, you are given additional relevance column indicating the relevance of the passage to the query, which you will need during training and validation.

###### Main Contribution:

1. Evaluating Retrieval Quality.  Implement methods to compute the average precision and NDCG metrics. Compute the performance of using BM25 as the retrieval model on the validation data (validation_data.tsv) using these metrics. 

2. Logistic Regression (LR).  Represent passages and query based on a word embedding method, (such as Word2Vec, GloVe, FastText, or ELMo). Compute query (/passage) embeddings by averaging embeddings of all the words in that query (/pas- sage). With these query and passage embeddings as input, implement a logistic re- gression model to assess relevance of a passage to a given query. Describe how you perform input processing & representation or features used. Using the metrics you have implemented in the previous part, report the performance of your model based on the validation data. Analyze the effect of the learning rate on the model training loss. 

3. LambdaMART Model (LM). Use the LambdaMART [1] learning to rank algo- rithm (a variant of LambdaRank we have learned in the class) from XGBoost gradient boosting library to learn a model that can re-rank passages. Command XG- Boost to use LambdaMART algorithm for ranking by setting the appropriate value to the objective parameter as described in the documentation. It is expected to carry out hyper-parameter tuning in this task and describe the methodology used in deriving the best performing model. Using the metrics that have implemented in the first part, report the performance of this model on the validation data.

4. Neural Network Model (NN).  Using the same training data representation from the previous question, build a neural network based model that can re-rank pas- sages. Use existing packages, namely Tensorflow or PyTorch in this subtask. It is allowed to use different types of neural network architectures (e.g. feed forward, convolutional, recurrent and/or transformer based neural networks) for this part. Using the metrics that have implemented in the first part, report the performance of this model on the validation data.

   