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
Extract age classes features to augment the prediction model.
#### Preprocessing:
Perform feature extraction relevant to age classification.
Apply one-hot encoding and remove null values.
#### Model Training:
Conduct a train-test split.
Train models and document the process.
Choose the model with the highest accuracy based on test data comparison.

![WhatsApp Image 2024-03-12 at 15 38 39_3e680947](https://github.com/timnirmal/YT_UI_Final/assets/42657815/30681376-3aa1-4f45-993d-ff2ff9e80b42)

![WhatsApp Image 2024-03-12 at 15 38 41_9674670c](https://github.com/timnirmal/YT_UI_Final/assets/42657815/acdedfa9-ab83-4b5f-971e-e1b8b3cf1075)

![WhatsApp Image 2024-03-12 at 22 04 46_1fceff39](https://github.com/timnirmal/YT_UI_Final/assets/42657815/40df9224-1051-4aa5-a376-561e366bc10e)

![WhatsApp Image 2024-03-12 at 22 04 47_740641ca](https://github.com/timnirmal/YT_UI_Final/assets/42657815/69a5b755-e1c0-47fc-b08c-b9aa09370aec)
