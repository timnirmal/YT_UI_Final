import pickle

import numpy as np
import pandas as pd
from gensim.models import word2vec
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import LSTM
from keras.models import Sequential
from keras.utils import pad_sequences
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# set seed for reproducibility
np.random.seed(42)

data = pd.read_csv('classes_dataset_cleaned.csv')  # Replace with your actual file path

# label encoder
label_encoder = LabelEncoder()
data['class'] = label_encoder.fit_transform(data['class'])

texts = data['filtered_sentence'].tolist()
labels = data['class'].tolist()

# Tokenize the text into sentences and words
sentences = [text.split() for text in texts]

# Preprocessing
embedding_dim = 300
max_length = 20
trunc_type = 'post'
padding_type = 'post'

# load word2vec model
word2vec_model = word2vec.Word2Vec.load("embedding/word2vec_300.w2v")

# Convert sentences to sequences of word indices
word_indices = []
for sentence in sentences:
    indices = []
    for word in sentence:
        if word in word2vec_model.wv.key_to_index:
            indices.append(word2vec_model.wv.key_to_index[word])

    word_indices.append(indices)

# Pad sequences to ensure consistent length
padded_sequences = pad_sequences(word_indices, maxlen=max_length, padding=padding_type, truncating=trunc_type)

# get dummy variables
encoded_labels = pd.get_dummies(labels).values
print(encoded_labels.shape)
print(encoded_labels.shape)
print(encoded_labels.shape)

X_train, X_test, y_train, y_test = train_test_split(padded_sequences, encoded_labels, test_size=0.2, random_state=42)

# save X_train, X_test, y_train, y_test
pickle.dump(X_train, open('models/X_train.pkl', 'wb'))
pickle.dump(X_test, open('models/X_test.pkl', 'wb'))
pickle.dump(y_train, open('models/y_train.pkl', 'wb'))
pickle.dump(y_test, open('models/y_test.pkl', 'wb'))

# labels = 0,1,2,3,4,5

# Define the LSTM model
model = Sequential()
model.add(Embedding(len(word2vec_model.wv.key_to_index), embedding_dim, input_length=max_length, trainable=False))
model.add(Dropout(0.3))
model.add(LSTM(100, return_sequences=True, dropout=0.3, recurrent_dropout=0.2))
model.add(LSTM(100))
model.add(Dense(6, activation='sigmoid'))
print(model.summary())
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# load
# X_train = pickle.load(open('models/X_train.pkl', 'rb'))
# X_test = pickle.load(open('models/X_test.pkl', 'rb'))
# y_train = pickle.load(open('models/y_train.pkl', 'rb'))
# y_test = pickle.load(open('models/y_test.pkl', 'rb'))

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Train the lstm model
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=64, verbose=1)
model.save_weights('models/lstm_model_weights.h5')
# model.save('models/lstm_model.h5')

model_acc = model.evaluate(X_test, y_test, verbose=1)[1]

# save the label encoder
pickle.dump(label_encoder, open('models/label_encoder.pkl', 'wb'))

# load the model
# model = load_model('lstm_model/lstm.h5')

# load the label encoder
# label_encoder = pickle.load(open('models/label_encoder.pkl', 'rb'))

# classification report
from sklearn.metrics import classification_report

y_pred = model.predict(X_test)
y_pred = np.round(y_pred).astype(int)
lstm_report = classification_report(y_test, y_pred)
print(lstm_report)

# training graph for lstm model
plt.figure(figsize=(10, 10))
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('LSTM Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('models/lstm_model_accuracy.png')
# plt.show()

# training graph for lstm model
plt.figure(figsize=(10, 10))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('LSTM Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('models/lstm_model_loss.png')
# plt.show()


# create log file
with open('models/log.txt', 'w') as f:
    f.write("Best model accuracy: " + str(model_acc) + "\n")
    f.write("Best model confusion matrix: " + "\n")
    f.write(lstm_report)
