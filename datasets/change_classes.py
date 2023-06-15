## Move this to main folder of UI

import glob
import pandas as pd

audio_df = pd.read_csv("audio_dataset.csv")
merged_df = pd.read_csv("merged_dataset.csv")

from classes.classes_predict import predict_classes_df

# predict classes for audio_df
audio_df = predict_classes_df(audio_df)
audio_df.to_csv("datasets/audio_dataset_2.csv", index=False)

# predict classes for merged_df
merged_df = predict_classes_df(merged_df)
merged_df.to_csv("datasets/merged_dataset_2.csv", index=False)


