import pickle

import pandas as pd
from gensim.models import word2vec
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import LSTM
from keras.models import Sequential
from keras.utils import pad_sequences
from w2v import word2vec_model

data_path = 'hate/models/'
w2v_path = 'hate/embedding/'

with open(data_path + 'label_encoder.pkl', 'rb') as file:
    label_encoder = pickle.load(file)


# Preprocessing
embedding_dim = 300
max_length = 20
trunc_type = 'post'
padding_type = 'post'

# Define the LSTM model
model = Sequential()
model.add(Embedding(len(word2vec_model.wv.key_to_index), embedding_dim, input_length=max_length, trainable=False))
model.add(Dropout(0.2))
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# print(model.summary())

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


def predict_sentiment(text):
    # Process the text data
    processed_text = process_text(text)

    # Make the prediction
    predictions = model.predict(processed_text)

    return predictions[0][0]


def predict_hate_df(df):
    print("=====================================")
    # print(df)
    # print(type(df))
    print(df['text'])
    print(type(df['text']))
    print("===================")
    # if not None
    if df['text'] is not None:
        df['hate'] = df['text'].apply(lambda x: predict_sentiment(x))

    # # save to csv
    # df.to_csv('final_hate_detect.csv', index=False)

    return df


def main():
    text = "අනේ කොච්චර ආදර්ශමත් ද බලන්න ඒ තාත්තගෙයි පුතාගෙයි බැදීම"
    print("\nText:", text)
    sentiment = predict_sentiment(text)
    print("Predicted sentiment:", sentiment)

    text = "අනේ පල යන්න.තොපිට මේ වෙලාවේ තියෙන්නෙ ඕකද බොල.වචනෙකින් හරි රටට යහපතක් වෙන දෙයක් කියන එක නේද."
    print("\nText:", text)
    sentiment = predict_sentiment(text)
    print("Predicted sentiment:", sentiment)

    # text = "දරුව මගෙ කියලා හිතුන මම"
    # print("\nText:", text)
    # sentiment = predict_sentiment(text)
    # print("Predicted sentiment:", sentiment)

    file_path = r"D:\Projects\Youtube_Research\UI_old\runs\run_4\recognized_processed.csv"

    df = pd.read_csv(file_path)

    predict_hate_df(df)


if __name__ == '__main__':
    main()
    print("main")
