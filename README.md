# Multimodal Hate Speech and Sentiment Detection

## Overview
This project is a multimodal machine learning approach that integrates audio and video data to predict hate speech and sentiment severity. It aims to accurately identify and classify the emotional content of multimedia inputs, providing a comprehensive understanding of the expressed sentiments and potential hate speech content. 

## Data Preprocessing

### Audio Data

#### Purpose: 
Predict hate speech and sentiment from audio files.
#### Preprocessing:
Convert audio data into a suitable format for analysis.
Use one-hot encoding to convert categorical data.
Remove null values to clean the dataset.
#### Model Training:
Split the data into training and testing subsets.
Train models using the prepared data, recording the process in log files.
Select the most accurate model by comparing predictions against test data.

### Video Data

#### Purpose: 
Extract age, gender, emotion classes features to augment the prediction model.
#### Preprocessing:
Perform feature extraction relevant to age classification.
Apply one-hot encoding and remove null values.
#### Model Training:
Conduct a train-test split.
Train models and document the process.
Choose the model with the highest accuracy based on test data comparison.

## class/domain prediction
classes_list = Sports, Religious, Political, Sexual, Education, Entertainment
for this a w2v embedding are taken for the text. Then these embedding are passed to Keras embedding layer with LSTM and few dense layers. With this we can categorize the text we have.

model = Sequential()
model.add(Embedding(len(word2vec_model.wv.key_to_index), embedding_dim, input_length=max_length, trainable=False))
model.add(Dropout(0.3))
model.add(LSTM(100, return_sequences=True, dropout=0.3, recurrent_dropout=0.2))
model.add(LSTM(100))
model.add(Dense(6, activation='sigmoid'))
print(model.summary())
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

## Age, Gender from Audio

For this first audio are separated into chunks. Then audio features like nobs, mean, skew, kurtosis, median, mode, std, low, peak, q25, q75, iqr are extracted. With the trained models for that age group and gender are predicted.





![WhatsApp Image 2024-03-12 at 15 38 39_3e680947](https://github.com/timnirmal/YT_UI_Final/assets/42657815/30681376-3aa1-4f45-993d-ff2ff9e80b42)

![WhatsApp Image 2024-03-12 at 15 38 41_9674670c](https://github.com/timnirmal/YT_UI_Final/assets/42657815/acdedfa9-ab83-4b5f-971e-e1b8b3cf1075)

![WhatsApp Image 2024-03-12 at 22 04 46_1fceff39](https://github.com/timnirmal/YT_UI_Final/assets/42657815/40df9224-1051-4aa5-a376-561e366bc10e)

![WhatsApp Image 2024-03-12 at 22 04 47_740641ca](https://github.com/timnirmal/YT_UI_Final/assets/42657815/69a5b755-e1c0-47fc-b08c-b9aa09370aec)


UI 
The process start with taking a youtube Url as input. 
Then that video is downloaded
then audio extracted with pydub and converted to mono chennel flac.
then audio is uploaded to google storage bucket and transcription is taken place with google cloud services. Then each work with saved into a csv file with start and end time of it.
Then we use preprocessing pipline to preprocess these text data chunk by chunk and then later merged into one again. But in this process some words are loss, some words are converted to another form or maybe another language. So we need some method to keep track of words with the original meta data and seqaunce they were appear. For that fuzzywuzzy is used.

Then classes are predicted
classes_list = Sports, Religious, Political, Sexual, Education, Entertainment

## data preprocessing pipeline
clean data - replace URL of a text, mentions, numbers, emojies, extra white spaces, trim whites spaces at beginig and end
translate to english to sinhala - google translator
singlish to sinhala converter - convert rest of the english to sinhala
tokanizing, stop word removal, detokanize
simplyfying sinhala letters


### singlish converter 

vowel_only_mapping = {
    'a': 'අ',
    'aa': 'ආ',
    'ae': 'ඇ',
    'aae': 'ඈ'
}

vowel_mapping = {
    'a': 'අ',
    'aa': 'ආ',
    'ae': 'ඇ',
    'aae': 'ඈ',
    'i': 'ඉ'
}

constant_mapping = {
    'k': 'ක්',
    'kh': 'ඛ්',
    'g': 'ග්',
    'gh': 'ඝ්'
}

depended_vowl_mapping = {
    'aa': 'ා',
    'ae': 'ැ',
    'aae': 'ෑ',
    'i': 'ි'
}

Each word is passed through a series of functions: convert_to_sinhala to get the basic Sinhala script representation, followed by fix_word and fix_first_letter_vowel to apply specific linguistic corrections.
fix_const_plus_a is applied last to address any occurrences of redundant virama plus the අ vowel. The processed words are then joined together to form the output text. Then sentimnet are predcited with the early method.


### simplifying letters

simplify_characters_dict = {
    # Consonant
    "ඛ": "ක",
    "ඝ": "ග",
    "ඟ": "ග",
    "ඡ": "ච"

    # Vowels
    "ආ": "අ",
    "ඈ": "ඇ",
    "ඊ": "ඉ"
}


### Fix sinhala vowel letters

sinhalese_chars = [
    "අ", "ආ", "ඇ", "ඈ", "ඉ", "ඊ",
    "උ", "ඌ", "ඍ", "ඎ", "ඏ", "ඐ",
    "එ", "ඒ", "ඓ", "ඔ", "ඕ", "ඖ",
    "ං", "ඃ",
    "ක", "ඛ", "ග", "ඝ", "ඞ", "ඟ",
    "ච", "ඡ", "ජ", "ඣ", "ඤ", "ඥ", "ඦ",
    "ට", "ඨ", "ඩ", "ඪ", "ණ", "ඬ",
    "ත", "ථ", "ද", "ධ", "න", "ඳ",
    "ප", "ඵ", "බ", "භ", "ම", "ඹ",
    "ය", "ර", "ල", "ව",
    "ශ", "ෂ", "ස", "හ", "ළ", "ෆ",
    "෴", "\u200d"
]

sinhalese_vowel_signs = ["්", "ා", "ැ", "ෑ", "ි", "ී", "ු", "ූ", "ෘ", "ෙ", "ේ", "ෛ", "ො", "ෝ",
                         "ෞ", "ෟ", "ෲ", "ෳ", "ර්‍"]

vowel_sign_fix_dict = {
    "ෑ": "ැ",
    "ෙ" + "්": "ේ",
    "්" + "ෙ": "ේ",

    "ෙ" + "ා": "ො",
    "ා" + "ෙ": "ො"
    
    # duplicating same symbol
    "ේ" + "්": "ේ",
    "ේ" + "ෙ": "ේ",

    # special cases - may be typing mistakes
    "ො" + "ෟ": "ෞ",
    "ෟ" + "ො": "ෞ",
}
