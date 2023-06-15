# Move to Main Folder for run

import pandas as pd

df = pd.read_csv("datasets/merged_dataset_2.csv")

# drop file_name,word,speaker,text,processed_text,
df = df.drop(['file_name', 'word', 'speaker', 'text', 'processed_text'], axis=1)


# classes,audio_age_range,audio_gen,hate,sentiment_word,age,gen,emotion,age_classes


def calculate_severity(domain, audio_age, audio_gender, hate, sentiment, video_age, video_gen, emotion, age_class):
    domain_severity = {'Political': 3, 'Religious': 3, 'Sexual': 2, 'Entertainment': 1, 'Education': 1, 'Sports': 1}
    audio_age_severity = {0: 1, 1: 3, 2: 2, 3: 0}
    audio_gender_severity = {0: 2, 1: 1}
    # hate_severity = {1: 5, 0: 0}
    sentiment_severity = {-1: 3, 0: 1, 1: 0}
    video_age_severity = {0: 1, 1: 3, 2: 2, 3: 0}
    video_gender_severity = {0: 2, 1: 1}
    emotion_severity = {'Angry': 4, 'Fear': 3, 'Happy': 2, 'Neutral': 1, 'Sad': 3, 'Surprise': 2}
    age_class_severity = {0: 1, 1: 3, 2: 2, 3: 0}

    severity = domain_severity.get(domain, 0)
    severity += audio_age_severity.get(audio_age, 0)
    severity += audio_gender_severity.get(audio_gender, 0)
    severity += (hate * 5)
    severity += sentiment_severity.get(sentiment, 0)
    severity += video_age_severity.get(video_age, 0)
    severity += video_gender_severity.get(video_gen, 0)
    severity += emotion_severity.get(emotion, 0)
    severity += age_class_severity.get(age_class, 0)

    return severity


# Calculate severity for each row
df['Severity'] = df.apply(
    lambda row: calculate_severity(row['classes'], row['audio_age_range'], row['audio_gen'], row['hate'],
                                   row['sentiment_word'], row['age'], row['gen'], row['emotion'], row['age_classes']),
    axis=1)

# Normalize severity values to a range of 1 to 5
min_severity = df['Severity'].min()
max_severity = df['Severity'].max()
if min_severity == 1 and max_severity == 6:
    # for df['Severity'] = 6 replace with 5
    df['Normalized Severity'] = df['Severity'].replace(6, 5)
else:
    # normalize severity values 1 to 6
    df['Normalized Severity'] = df['Severity'].apply(
        lambda x: round((x - min_severity) / (max_severity - min_severity) * 4 + 1))


# Display the DataFrame
print(df)

# Save the DataFrame to a CSV file
df.to_csv('datasets/merged_dataset_severity.csv', index=False)
