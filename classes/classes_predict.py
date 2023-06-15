import pickle
import numpy as np
from gensim.models import Word2Vec, word2vec
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import LSTM
from keras.layers import Embedding
from keras.utils import pad_sequences

data_path = 'classes/models/'
w2v_path = 'classes/embedding/'

with open(data_path + 'label_encoder.pkl', 'rb') as file:
    label_encoder = pickle.load(file)

word2vec_model = word2vec.Word2Vec.load(w2v_path + "word2vec_300.w2v")

# Preprocessing
embedding_dim = 300
max_length = 20
trunc_type = 'post'
padding_type = 'post'

# Define the LSTM model
model = Sequential()
model.add(Embedding(len(word2vec_model.wv.key_to_index), embedding_dim, input_length=max_length, trainable=False))
model.add(Dropout(0.3))
model.add(LSTM(100, return_sequences=True, dropout=0.3, recurrent_dropout=0.2))
model.add(LSTM(100))
model.add(Dense(6, activation='sigmoid'))
print(model.summary())
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# load model weights
model.load_weights(data_path + 'lstm_model_weights.h5')


def process_text(text):
    # Tokenize the text into words
    words = text.split()

    # Convert words to word indices
    word_indices = []
    for word in words:
        if word in word2vec_model.wv.key_to_index:
            word_indices.append(word2vec_model.wv.key_to_index[word])

    # Pad the sequence to ensure consistent length
    padded_sequences = pad_sequences([word_indices], maxlen=max_length, padding=padding_type, truncating=trunc_type)

    return padded_sequences


def predict_domain(text):
    # Process the text data
    processed_text = process_text(text)

    # Make the prediction
    predictions = model.predict(processed_text)

    # prediction to label
    predictions = np.argmax(predictions, axis=1)
    predictions = label_encoder.inverse_transform(predictions)

    return predictions[0]


def predict_classes_df(df):
    print("=====================================")
    # print(df)
    # print(type(df))
    print(df['text'])
    print(type(df['text']))
    print("===================")
    # if not None
    if df['text'] is not None:
        df['classes'] = df['text'].apply(lambda x: predict_domain(x))

    # # save to csv
    # df.to_csv('final_classes_detect.csv', index=False)

    return df
