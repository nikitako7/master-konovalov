import pandas as pd
import numpy as np

def get_movie_recomendations_for_all_movies(user_id, model, genres, data):
  test_user = pd.DataFrame({'user_encoded' : [user_id] * len(data['movie_encoded'].unique()),'movie_encoded':list(data['movie_encoded'].unique())})
  test_user['Predicted_Ratings'] = np.round(model.predict([test_user.user_encoded, test_user.movie_encoded]),0)

  result = pd.merge(test_user,data[['user_encoded','movie_encoded', 'label']],how='left',on=['user_encoded','movie_encoded'])
  result.sort_values(by='Predicted_Ratings',ascending=False,inplace=True)

  final_df = pd.merge(result,genres,how='inner',on=['movie_encoded'])
  final_df = final_df.sort_values(by=['Predicted_Ratings'],ascending=False)

  return final_df