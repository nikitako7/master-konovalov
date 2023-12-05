#!pip install LibRecommender

#!pip install ctgan

#!pip install table-evaluator

#!pip install ctgan

import numpy as np
import pandas as pd
from google.colab import drive
import warnings
from sklearn.model_selection import train_test_split
import tensorflow as tf
from numpy import expand_dims
from numpy import zeros
from numpy import ones
from numpy.random import randn
from numpy.random import randint
from keras.datasets.fashion_mnist import load_data
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import Dropout
import seaborn as sns
from ctgan import CTGAN
from pathlib import Path
import keras
from IPython.display import SVG
from keras.optimizers import Adam
from keras.utils import model_to_dot
from table_evaluator import TableEvaluator
from collections import defaultdict
import matplotlib.pyplot as plt

from libreco.data import random_split, DatasetPure
from libreco.algorithms import NCF
from libreco.evaluation import evaluate
from load_ml_1m import load_ml_1m
from ncf import ncf_model
from get_correct_rating import correct_rating

from sklearn.metrics import mean_squared_error as MSE,mean_absolute_error
from tabulate import tabulate
from sklearn.model_selection import train_test_split
from metrics import get_precision_recall_f1_score
from metrics_plot import precision_recall_f1_curves_
from get_movie_rec_for_user import get_movie_recomendations_for_all_movies

warnings.filterwarnings("ignore")

drive.mount('/content/drive')

# Data Reading
movielens_data = load_ml_1m()

print(movielens_data)

movielens_data = movielens_data[['user', 'item', 'label', 'time']]

# Generative adversarial network (GAN)

real_data = movielens_data[['user', 'item', 'label']]

#Identifies all the discrete columns (conditions)
discrete_columns = [
    'user',
    'item',
]

# Initiates the CTGAN and call its fit method to pass in the table
ctgan = CTGAN(epochs=10)
ctgan.fit(real_data[:50000], discrete_columns)

#ctgan.save('/content/drive/MyDrive/konovalov_master/ctgan_model')

synthetic_data = ctgan.sample(10000)
print(synthetic_data.head(5))
print(synthetic_data.label.unique())

# GAN Evaluation
table_evaluator =  TableEvaluator(real_data[:50000], synthetic_data, cat_cols= discrete_columns)
table_evaluator.visual_evaluation()

# NCF
tf.compat.v1.enable_eager_execution()

movielens_data_selected = movielens_data[['user', 'item', 'label', 'title', 'genre1']]

movielens_data_selected['user_encoded'] = movielens_data_selected.user.astype('category').cat.codes.values
movielens_data_selected['movie_encoded'] = movielens_data_selected.item.astype('category').cat.codes.values
user_movie_cate_df = movielens_data_selected[['user','user_encoded','item','movie_encoded']]
movielens_data_selected.drop(['user','item'],axis=1,inplace=True)

train, test = train_test_split(movielens_data_selected[['user_encoded', 'movie_encoded', 'label']], test_size=0.2,random_state=32)

# True rating for test dataframe
y_true = test.label

n_users, n_movies = len(movielens_data_selected.user_encoded.unique()), len(movielens_data_selected.movie_encoded.unique())
print(train.head())

model, history_ep30_adam_lr0_005 = ncf_model(30, 'adam', 'mean_absolute_error', n_movies, n_users, train, test)

print(history_ep30_adam_lr0_005.history.keys())

def loss_curve(train_loss, val_loss):
  
  print('Loss (MAE): ', val_loss[-1])

  loss_values = pd.DataFrame({"train": train_loss,
                           "val": val_loss})
  loss_values = loss_values.reset_index(drop=False).melt(id_vars ="index")
  loss_values = loss_values.rename(columns={"index":"epoch", "value" : "mse", "variable" :"sample"})

  fig = plt.figure(figsize=(8, 7))
  sns.set(style='darkgrid')
  sns.lineplot(x="epoch", y="mse", hue="sample", data=loss_values)
  plt.title("Loss Curve per Epochs", fontsize=12)
  plt.show()

# Evaluation
loss_curve(history_ep30_adam_lr0_005.history['loss'], history_ep30_adam_lr0_005.history['val_loss'])
print(mean_absolute_error(y_true, model.predict([test.user_encoded, test.movie_encoded])))

genres = pd.DataFrame()
genres['movie_encoded'] = movielens_data_selected.movie_encoded.unique()
genres = pd.merge(genres, movielens_data_selected[['movie_encoded', 'genre1']], how='left', on = ['movie_encoded']).drop_duplicates().reset_index(drop=True)
genres = pd.merge(genres, movielens_data_selected[['movie_encoded', 'title']], how='left', on = ['movie_encoded']).drop_duplicates().reset_index(drop=True)

rec_user_214 = get_movie_recomendations_for_all_movies(214, model, genres, movielens_data_selected)

def testing(test, model):
  test_res = pd.DataFrame()
  test_res = pd.DataFrame({'user_encoded' : test.user_encoded,'movie_encoded': test.movie_encoded})
  test_res['predicted_rating'] = np.round(model.predict([test.user_encoded, test.movie_encoded]),0).flatten()

  result = pd.merge(test_res, test, how='left',on=['user_encoded','movie_encoded'])

  final_df = pd.merge(result,genres,how='inner',on=['movie_encoded'])

  return final_df

res = model.predict([test.user_encoded, test.movie_encoded])
print(res)

res_testing = testing(test, model)
print(res_testing)

user_pred_true = defaultdict(list)
for i in range(0, len(res_testing)):
    actual_rating = res_testing.loc[i,'label']
    predicted_rating = res_testing.loc[i,'predicted_rating']
    user = res_testing.loc[i,'user_encoded']
    user_pred_true[user].append((predicted_rating, actual_rating))

avg_precision_df, avg_recall_df, avg_f1_score_df = get_precision_recall_f1_score(10, 4, user_pred_true)

print(avg_precision_df)
print(avg_recall_df)
print(avg_f1_score_df)

#ep=30, opt=adam, lr=0.001
precision_recall_f1_curves_(avg_precision_df, avg_recall_df, avg_f1_score_df)

# Hybrid Model

synthetic_data_corrected = synthetic_data.copy()
synthetic_data_corrected['label_corrected'] = synthetic_data_corrected['label'].apply(correct_rating)
synthetic_data_corrected

res_testing['ctgan_predicted'] = 0
for i in range(len(res_testing)):
  for j in range(len(synthetic_data_corrected)):
    if (synthetic_data_corrected.loc[j,'user'] == res_testing.loc[i,'user_encoded']) & (synthetic_data_corrected.loc[j,'item'] == res_testing.loc[i,'movie_encoded']):
      res_testing.loc[i, 'ctgan_predicted'] = synthetic_data_corrected.loc[j,'label_corrected']
res_testing['hybrid'] = res_testing['predicted_rating']
res_testing['hybrid'][res_testing['ctgan_predicted'] != 0] =  0.7*res_testing['predicted_rating']+0.3*res_testing['ctgan_predicted']


