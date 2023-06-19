import re
import numpy as np
import pandas as pd

from SubProcesses import video_severity, audio_severity, Merging, VideoPrediction, SentimentCalculation, HatePrediction, \
    AgeNGenderEstimation, DomainPrediction, TranscribingNProcessing, AudioConversion, merged_severity, create_folder

# load word2vec model


print("loaded")

# show all columns
pd.set_option('display.max_columns', None)


def main(video_file):
    print("Main Started...", video_file)
    # get file name spit by / or \
    file_name = re.split(r'[/\\]', video_file)[-1]

    # create a folder inside runs folder if not exists if not exists print "file exists"
    run_number = create_folder("runs/", file_name)
    # run_number = 3
    run_path = "runs/run_" + str(run_number) + "/"

    AudioConversion(file_name, run_path, video_file)

    audio_df = TranscribingNProcessing(file_name, run_number, run_path)
    audio_df = DomainPrediction(audio_df, run_path)
    audio_df = AgeNGenderEstimation(audio_df, run_path, file_name)
    hate_df = HatePrediction(audio_df, run_path)
    word_sentiment_df = SentimentCalculation(hate_df, run_path)

    video_df = VideoPrediction(run_path, video_file)

    print(audio_df.head())
    print(video_df.head())

    merged_df = Merging(file_name, run_path, video_df, word_sentiment_df)

    audio_severity(run_path, word_sentiment_df)

    video_severity(run_path, video_df)

    merged_severity(merged_df, run_path)

    ##################################### Finalizing #####################################

    audio_df = pd.read_csv(run_path + "audio_predicted.csv")[['hate', 'severity']]
    video_df = pd.read_csv(run_path + "video_predicted.csv")[['severity']]
    merged_df = pd.read_csv(run_path + "merged_predicted.csv")[['hate', 'severity']]

    print("Mean of the audio hate: ", np.mean(audio_df['hate']), " - ", np.round(np.mean(audio_df['hate']) * 100, 2),
          "%")
    print("Mean of the audio severity: ", np.mean(audio_df['severity']), " - ",
          np.round(np.mean(audio_df['severity']) * 25, 2), "%")
    print("Mean of the video severity: ", np.mean(video_df['severity']), " - ",
          np.round(np.mean(video_df['severity']) * 25, 2), "%")
    print("Mean of the merged severity: ", np.mean(merged_df['severity']), " - ",
          np.round(np.mean(merged_df['severity']) * 25, 2), "%")


file_path = "test_files/Malinga.mkv"

main(file_path)

# Audio conversion
# Transcribing and processing
# Domain prediction
# Age & Gender estimation
# Hate prediction
# Sentiment calculation
# Video prediction
# Merging
# Audio sentiment
# Video sentiment
# Merged sentiment
