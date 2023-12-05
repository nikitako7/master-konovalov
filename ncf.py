import keras

def ncf_model(epochs, opt, loss, n_movies, n_users, train, test):
  n_latent_factors_user = 6
  n_latent_factors_movie = 10

  #movie embeddings
  movie_input = keras.layers.Input(shape=[1])
  movie_embedding = keras.layers.Embedding(n_movies + 1, n_latent_factors_movie)(movie_input)
  movie_vec = keras.layers.Flatten()(movie_embedding)

  #user embeddings
  user_input = keras.layers.Input(shape=[1])
  user_embedding = keras.layers.Embedding(n_users + 1, n_latent_factors_user)(user_input)
  user_vec = keras.layers.Flatten()(user_embedding)

  concat = keras.layers.Concatenate([movie_vec, user_vec])
  concat_dropout = keras.layers.Dropout(0.2)(concat)
  dense = keras.layers.Dense(128)(concat)
  dropout_1 = keras.layers.Dropout(0.2)(dense)
  dense_2 = keras.layers.Dense(64, activation='relu')(dense)

  result = keras.layers.Dense(1, activation='relu')(dense_2)
  model = keras.layers.Model([user_input, movie_input], result)
  model.compile(optimizer = opt, loss = loss)
  history = model.fit([train.user_encoded, train.movie_encoded], train.label, validation_data = ([test.user_encoded, test.movie_encoded], test.label), epochs=epochs, verbose=1)

  return model, history