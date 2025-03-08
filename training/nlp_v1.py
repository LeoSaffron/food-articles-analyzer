# -*- coding: utf-8 -*-
"""
Created on Thu Dec 22 01:43:34 2022

@author: hor
"""

import pandas as pd  # for working with dataframes
import numpy as np  # for numerical computing

# for working with word embeddings

# for downloading and working with pre-trained word embeddings
import gensim.downloader as api

# for natural language processing tasks
import nltk
import re
import gensim

# for padding and preprocessing sequences
from keras.preprocessing.sequence import pad_sequences

# for data visualization
import seaborn as sns
import matplotlib.pyplot as plt

# for evaluating classification models
from sklearn.metrics import confusion_matrix, roc_curve, average_precision_score, precision_recall_curve
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score

# packages for models
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# preprocessing packages
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler

# for splitting the dataset into train and test sets

# Cross-validation
from sklearn.model_selection import KFold
import itertools


model = api.load("word2vec-google-news-300")
w2v_embedding = api.load("glove-wiki-gigaword-100")



# Load the sheet from an Excel file
df = pd.read_excel('ingredients_short_list_tagged.xlsx', sheet_name='Combined')

df = df[(df['label'] == 'v') | (df['label'] == 'x')].reset_index(drop=True)


def tokenize(text):
    # Tokenize the text using nltk's word_tokenize function
    tokens = nltk.word_tokenize(text)

    # Use a regular expression to remove punctuation
    # and make all characters lowercase
    tokens = [re.sub(r'[^\w\s]', '', token.lower()) for token in tokens]

    # Remove any remaining tokens that are just whitespace
    tokens = [token for token in tokens if token.strip() != '']

    # Find similar words and group them together
    grouped_tokens = []
    for token in tokens:
        token_new = token
        if not token_new in w2v_embedding:
            token_new = "one"
        grouped_tokens.append(w2v_embedding.word_vec(token_new))

    return grouped_tokens

# Apply the tokenize function to each cell in the 'text' column of the DataFrame
df['tokens'] = df['ingredient'].apply(tokenize)



def pad(data):
    # x, y = data
    x = data
    x = pad_sequences([x], padding="post", maxlen=10, dtype='float32')[0]
    # y = pad_sequences([y], padding="post", maxlen=10)[0]
    return x#, y

data = [pad(pair) for pair in df['tokens']]


def plot_evaluation(y_true, y_pred):
    """
    Plot evaluation metrics for a binary classification model.
    
    Parameters:
    - y_true: 1D array of true labels
    - y_pred: 1D array of predicted labels
    """
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Normalize confusion matrix
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create a heatmap
    sns.heatmap(cm, annot=True, fmt='.2f', xticklabels=['0', '1'], yticklabels=['0', '1'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()
    
    # Plot ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred)
    plt.plot(fpr, tpr, label=f'AUC = {auc:.2f}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()
    
    # Plot precision-recall curve
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    average_precision = average_precision_score(y_true, y_pred)
    plt.plot(recall, precision, label=f'Average Precision = {average_precision:.2f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.show()


X = np.array(data)
X = X.reshape((X.shape[0],X.shape[1]* X.shape[2]))
y = np.array((df['label'] == 'v').astype('int'))




####
## train test split
####

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)



# Create the over-sampler
over_sampler = RandomOverSampler()


def create_grid_df(grid_params):
  df_result = pd.DataFrame(itertools.product(*grid_params.values()), columns=grid_params.keys())
  return df_result



def my_grid_search(model_name, param_grid):
    # Create a KFold object with 5 folds
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    list_model_cv_results = []
    # Loop through the folds
    df_params_temp = create_grid_df(param_grid)
    for train_index, test_index in kf.split(X_train):
        # Split the data into train and test sets
        X_train_fold, X_test_fold = X_train[train_index], X_train[test_index]
        y_train_fold, y_test_fold = y_train[train_index], y_train[test_index]
        
        # Fit and transform the data
        X_resampled, y_resampled = over_sampler.fit_resample(X_train_fold, y_train_fold)
        for i in range(len(df_params_temp)):
            parameter_row = dict(df_params_temp.iloc[i])
            # Train a model on the training set with the current set of hyperparameters
            
            model = model_name(**parameter_row)
            # model.fit(X_train_fold, y_train_fold)
            model.fit(X_resampled, y_resampled)
    
            # Evaluate the model on the test set
            y_pred_fold = model.predict(X_test_fold)
            parameter_row['accuracy'] = accuracy_score(y_test_fold, y_pred_fold)
            parameter_row['model'] = model
            list_model_cv_results.append(parameter_row)
        # Loop through all combinations of hyperparameters
        
    
    model_results = pd.DataFrame(list_model_cv_results).sort_values(by='accuracy', ascending=False)
    return model_results

param_grid_svc = {
    'C': [0.1, 1, 10, 100, 1000],
    'gamma': [0.001, 0.01, 0.1, 1, 10]
}

results_svc = my_grid_search(SVC, param_grid_svc)
model_best_results_on_CV = results_svc['model'].iloc[0]
y_pred = model_best_results_on_CV.predict(X_test)
plot_evaluation(y_test, y_pred)

param_grid_random_forest = {
    'n_estimators': [10, 50, 100, 200],
    'max_depth': [5, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2'],
}

results_random_forest = my_grid_search(RandomForestClassifier, param_grid_random_forest)
model_best_results_on_CV = results_random_forest['model'].iloc[0]
y_pred = model_best_results_on_CV.predict(X_test)
plot_evaluation(y_test, y_pred)

