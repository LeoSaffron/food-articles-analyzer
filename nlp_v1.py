# -*- coding: utf-8 -*-
"""
Created on Thu Dec 22 01:43:34 2022

@author: hor
"""

import pandas as pd  # for working with dataframes
import numpy as np  # for numerical computing

# for working with word embeddings
# from vecsim.utils import download_model
# from vecsim.embedders import Word2VecEmbedder

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
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

# for training and evaluating a random forest classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

# for balancing classes in the dataset
from imblearn.over_sampling import RandomOverSampler

# for splitting the dataset into train and test sets
from sklearn.model_selection import train_test_split



# model_path = download_model("word2vec", "google-news-vectors-negative300")


model = api.load("word2vec-google-news-300")
w2v_embedding = api.load("glove-wiki-gigaword-100")




# Load the sheet from an Excel file
df = pd.read_excel('ingredients_short_list_tagged.xlsx', sheet_name='Combined')

# Print the DataFrame
print(df)


df = df[(df['label'] == 'v') | (df['label'] == 'x')].reset_index(drop=True)



# def tokenize(text):
#     # Tokenize the text using nltk's word_tokenize function
#     tokens = nltk.word_tokenize(text)

#     # Use a regular expression to remove punctuation
#     # and make all characters lowercase
#     tokens = [re.sub(r'[^\w\s]', '', token.lower()) for token in tokens]

#     # Remove any remaining tokens that are just whitespace
#     tokens = [token for token in tokens if token.strip() != '']

#     return tokens

# # Apply the tokenize function to each cell in the 'text' column of the DataFrame
# df['tokens'] = df['ingredient'].apply(tokenize)



# def tokenize1(text):
#     # Tokenize the text using nltk's word_tokenize function
#     tokens = nltk.word_tokenize(text)

#     # Use a regular expression to remove punctuation
#     # and make all characters lowercase
#     tokens = [re.sub(r'[^\w\s]', '', token.lower()) for token in tokens]

#     # Remove any remaining tokens that are just whitespace
#     tokens = [token for token in tokens if token.strip() != '']

#     # Find similar words and group them together
#     grouped_tokens = []
#     for token in tokens:
#         token_new = token
#         if not token_new in w2v_embedding:
#             token_new = "one"
#         similar_words = w2v_embedding.most_similar(token_new)
#         grouped_tokens.append([word[0] for word in similar_words])

#     return grouped_tokens


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
print(data)





counts = []
for i in range(len(df['tokens'])):
    counts.append(len(df['tokens'].iloc[i]))





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




###
## Rebalance
###


# # Create the over-sampler
over_sampler = RandomOverSampler()

# # Fit and transform the data
# X_resampled, y_resampled = over_sampler.fit_resample(X_train, y_train)



# # Create a random forest classifier
# clf = RandomForestClassifier(n_estimators=100, random_state=42)

# # Fit the model to the training data
# clf.fit(X_resampled, y_resampled)


# y_pred = clf.predict(X_test)


# accuracy = accuracy_score(y_test, y_pred)
# print(f'Accuracy: {accuracy:.2f}')


# precision = precision_score(y_test, y_pred)
# print(f'Precision: {precision:.2f}')




# recall = recall_score(y_test, y_pred)
# print(f'Recall: {recall:.2f}')


## Random search Random forect Cross validation

# Define the hyperparameter grid for the random forest classifier
param_grid = {
    'n_estimators': [10, 50, 100, 200],
    'max_depth': [None, 5, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2'],
}

# Create the random forest classifier
rfc = RandomForestClassifier(random_state=42)

# Create the random search object
random_search = RandomizedSearchCV(estimator=rfc, param_distributions=param_grid,
                                   n_iter=50, cv=5, random_state=42, verbose=2, n_jobs=-1)

# Fit the random search object to the training data
random_search.fit(X_train, y_train)

# Print the best parameters and score
print(f'Best parameters: {random_search.best_params_}')
print(f'Best score: {random_search.best_score_:.2f}')

y_pred = random_search.predict(X_test)

plot_evaluation(y_test, y_pred)




####
## train val test split
####




X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)








    

plot_evaluation(y_test, y_pred)









from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from imblearn.over_sampling import RandomOverSampler

# Define the hyperparameter grid for the random forest classifier
param_grid = {
    'classifier__n_estimators': [10, 50, 100, 200],
    'classifier__max_depth': [None, 5, 10, 20],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__min_samples_leaf': [1, 2, 4],
    'classifier__max_features': ['auto', 'sqrt', 'log2'],
}

# Create the pipeline
pipeline = Pipeline([
    ('oversampler', RandomOverSampler()),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Create the random search object
random_search = RandomizedSearchCV(estimator=pipeline, param_distributions=param_grid,
                                   n_iter=50, cv=5, random_state=42, verbose=2, n_jobs=-1)

# Fit the random search object to the training data
random_search.fit(X, y)

# Print the best parameters and score
print(f'Best parameters: {random_search.best_params_}')
print(f'Best score: {random_search.best_score_:.2f}')









from imblearn.over_sampling import RandomOverSampler
from sklearn.tree import DecisionTreeClassifier
from imblearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
import numpy as np

dtpass = DecisionTreeClassifier()
sampling=RandomOverSampler()


pipe1=make_pipeline(sampling,dtpass)
# pipe1 = Pipeline([('sampling', RandomOverSampler()), ('class', dtpass)])

parameters = {'class__max_depth': range(3,7), 
          'class__ccp_alpha': np.arange(0, 0.001, 0.00025), 
          'class__min_samples_leaf' : [50]
         }

dt2 = GridSearchCV(estimator = pipe1, 
               param_grid = parameters,
               n_jobs = 4,
              scoring = 'roc_auc'
)

dt2.fit(X_train, y_train)









from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from scipy.stats import randint
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report

# # Load the data and labels
# X = data
# y = labels

# Create a list of hyperparameters to tune
# param_grid = {
#     'C': randint(0.1, 1000),
#     'gamma': randint(0.001, 10)
# }

# # Create a KFold object with 5 folds
# kf = KFold(n_splits=5, shuffle=True, random_state=42)

# # Loop through the folds
# for train_index, test_index in kf.split(X_train):
#     # Split the data into train and test sets
#     X_train_fold, X_test_fold = X_train[train_index], X_train[test_index]
#     y_train_fold, y_test_fold = y_train[train_index], y_train[test_index]
    
    


#     # Fit and transform the data
#     X_resampled, y_resampled = over_sampler.fit_resample(X_train_fold, y_train_fold)

#     # Randomly search over the hyperparameters
#     clf = RandomizedSearchCV(SVC(), param_grid, random_state=42, n_iter=10, cv=3)
#     clf.fit(X_resampled, y_resampled)

#     # Print the best set of hyperparameters
#     print(clf.best_params_)

#     # Evaluate the model on the test set
#     y_pred_fold = clf.predict(X_test_fold)
#     print(classification_report(y_test_fold, y_pred_fold))







from sklearn.model_selection import KFold

# # # Load the data and labels
# # X = data
# # y = labels

# # Create a list of hyperparameters to tune
# param_grid = {
#     'C': [0.1, 1, 10, 100, 1000],
#     'gamma': [0.001, 0.01, 0.1, 1, 10]
# }

# # Create a KFold object with 5 folds
# kf = KFold(n_splits=5, shuffle=True, random_state=42)

# list_model_cv_results = []
# # Loop through the folds
# for train_index, test_index in kf.split(X_train):
#     # Split the data into train and test sets
#     X_train_fold, X_test_fold = X_train[train_index], X_train[test_index]
#     y_train_fold, y_test_fold = y_train[train_index], y_train[test_index]
    
#     # Fit and transform the data
#     X_resampled, y_resampled = over_sampler.fit_resample(X_train_fold, y_train_fold)


#     # Loop through all combinations of hyperparameters
#     for C in param_grid['C']:
#         for gamma in param_grid['gamma']:
#             # Train a model on the training set with the current set of hyperparameters
#             model = SVC(C=C, gamma=gamma)
#             model.fit(X_train_fold, y_train_fold)

#             # Evaluate the model on the test set
#             y_pred_fold = model.predict(X_test_fold)
#             score = accuracy_score(y_test_fold, y_pred_fold)
#             list_model_cv_results.append({'C' : C, 'gamma' : gamma, 'accuracy' : score, 'model' : model})
#             print(f'C={C}, gamma={gamma}: {score:.3f}')

# model_results = pd.DataFrame(list_model_cv_results).sort_values(by='accuracy', ascending=False)
# model_best_results_on_CV = model_results['model'].iloc[0]
# y_pred = model_best_results_on_CV.predict(X_test)
# plot_evaluation(y_test, y_pred)


import itertools
import pandas as pd

def convert_param_list_dict_to_dataframeof_combinations(param_grid):
    # Create the dataframe
    df = pd.DataFrame(
        param_grid
    )
    
    
    # Create a list of lists containing the values in each column
    values = [df[col].tolist() for col in df.columns]
    
    # Create a dataframe of every combination of values
    df_combinations = pd.DataFrame(list(itertools.product(*values)), columns=df.columns)
    
    # Print the dataframe
    return df_combinations



def create_combinations_df(lists):
  # Create a list of tuples, where each tuple contains one value from each list
  combinations = list(itertools.product(*lists))

  # Convert the list of tuples to a dataframe
  df_result = pd.DataFrame(combinations, columns=range(len(lists)))

  return df_result


def create_grid_df(grid_params):
  # Create a list of tuples, where each tuple contains one value from each list
  # combinations = list(itertools.product(*grid_params.values()))

  # # Convert the list of tuples to a dataframe
  # df_result = pd.DataFrame(combinations, columns=grid_params.keys())
  
  
  df_result = pd.DataFrame(itertools.product(*grid_params.values()), columns=grid_params.keys())

  # Convert the list of tuples to a dataframe
  # df_result = pd.DataFrame(combinations, columns=grid_params.keys())
  # for key in grid_params.keys():
  #     df_result[key] = df_result[key].astype(pd.Series(grid_params[key]).dtype)

  return df_result



param_grid = {
    'C': [0.1, 1, 10, 100, 1000],
    'gamma': [0.001, 0.01, 0.1, 1, 10]
}

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
        # df_params_temp = convert_param_list_dict_to_dataframeof_combinations(param_grid)
        for i in range(len(df_params_temp)):
            parameter_row = dict(df_params_temp.iloc[i])
            # Train a model on the training set with the current set of hyperparameters
            
            model = model_name(**parameter_row)
            # model.fit(X_train_fold, y_train_fold)
            model.fit(X_resampled, y_resampled)
    
            # Evaluate the model on the test set
            y_pred_fold = model.predict(X_test_fold)
            score = accuracy_score(y_test_fold, y_pred_fold)
            parameter_row['accuracy'] = accuracy_score(y_test_fold, y_pred_fold)
            parameter_row['model'] = model
            list_model_cv_results.append(parameter_row)
            # print(f'C={C}, gamma={gamma}: {score:.3f}')
        # Loop through all combinations of hyperparameters
        
    
    model_results = pd.DataFrame(list_model_cv_results).sort_values(by='accuracy', ascending=False)
    return model_results

results_svc = my_grid_search(SVC, param_grid)
model_best_results_on_CV = results_svc['model'].iloc[0]
y_pred = model_best_results_on_CV.predict(X_test)
plot_evaluation(y_test, y_pred)






param_grid = {
    'classifier__n_estimators': [10, 50, 100, 200],
    'classifier__max_depth': [None, 5, 10, 20],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__min_samples_leaf': [1, 2, 4],
    'classifier__max_features': ['auto', 'sqrt', 'log2'],
}




param_grid = {
    'n_estimators': [10, 50, 100, 200],
    'max_depth': [5, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2'],
}


# # Create the pipeline
# pipeline = Pipeline([
#     ('oversampler', RandomOverSampler()),
#     ('classifier', RandomForestClassifier(random_state=42))
# ])



results_random_forest = my_grid_search(RandomForestClassifier, param_grid)
model_best_results_on_CV = results_random_forest['model'].iloc[0]
y_pred = model_best_results_on_CV.predict(X_test)
plot_evaluation(y_test, y_pred)






# # import itertools
# # import pandas as pd
# # # Example usage
# # df_result = create_combinations_df([[1, 2, 3], [4, 5, 6], [7, 8, 9, 10]])
# # print(df_result)


# import itertools
# import pandas as pd

# def create_grid_df(grid_params):
#   # Create a list of tuples, where each tuple contains one value from each list
#   combinations = list(itertools.product(*grid_params.values()))

#   # Convert the list of tuples to a dataframe
#   df = pd.DataFrame(combinations, columns=grid_params.keys())

#   return df

# # Example usage
# grid_params = {'param1': [1, 2, 3], 'param2': [4, 5, 6], 'param3': [7, 8, 9, 10]}
# df = create_grid_df(grid_params)
# print(df)


