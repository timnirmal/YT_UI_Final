## Move this to main folder of UI

import glob

import pandas as pd

merged_df = pd.DataFrame()
audio_df = pd.DataFrame()
video_df = pd.DataFrame()

# for each merged_df.csv in D:\Projects\UI\datasets\Runs subfolders (more deep inside) print df
# get all merged_df.csv files
for file in glob.glob("datasets/Runs/**/**/merged_df.csv", recursive=True):
    print(file)
    df = pd.read_csv(file)
    # concat all df
    merged_df = pd.concat([merged_df, df], axis=0)

# get audio_df
for file in glob.glob("datasets/Runs/**/**/word_sentiment_df.csv", recursive=True):
    print(file)
    df = pd.read_csv(file)
    # concat all df
    audio_df = pd.concat([audio_df, df], axis=0)

# get video_df
for file in glob.glob("datasets/Runs/**/**/filtered_frames.csv", recursive=True):
    print(file)
    df = pd.read_csv(file)
    # concat all df
    video_df = pd.concat([video_df, df], axis=0)

# save all df
merged_df.to_csv("datasets/merged_df.csv", index=False)
audio_df.to_csv("datasets/audio_df.csv", index=False)
video_df.to_csv("datasets/video_df.csv", index=False)

from sentiment.sentiment_word import calculate_sentiment_df

# calculate sentiment for all words in audio_df
audio_df = calculate_sentiment_df(audio_df)
audio_df.to_csv("datasets/audio_dataset.csv", index=False)

# merged
merged_df = calculate_sentiment_df(merged_df)
merged_df.to_csv("datasets/merged_dataset.csv", index=False)

# save video_df as video dataset
video_df.to_csv("datasets/video_dataset.csv", index=False)

