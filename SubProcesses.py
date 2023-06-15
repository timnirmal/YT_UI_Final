import datetime
import os
import pickle
import re

import librosa
import pandas as pd

from audio.process_data import process_data_df
from audio.process_with_timestamp import process_with_timestamp
from audio.transcribe_data_gcloud import get_large_audio_transcription
from audio.video_to_audio import convert_video_to_audio_ffmpeg, wav_to_mono_flac
from classes.classes_predict import predict_classes_df
from hate.hate_lstm_predict import predict_hate_df
from lib.sync_av import sync_audio_and_video
from sentiment.sentiment_word import calculate_sentiment_df
from video.model import predict
from voice_classifier.age_predictor import voice_age_predict_df
from voice_classifier.featureExtractor import get_features_chunk
from voice_classifier.gen_predictor import voice_gen_predict_df


def video_sentiment(run_path, video_df):
    ##################################### video severity #####################################
    video_df = video_df[['age', 'gen', 'emotion', 'age_classes']]
    # onehot encoding on gen and emotion
    video_df = pd.get_dummies(video_df, columns=['gen', 'emotion'])
    classes_list = ['emotion_Angry', 'emotion_Fear', 'emotion_Happy', 'emotion_Neutral', 'emotion_Sad',
                    'emotion_Surprise', 'gen_F', 'gen_M']
    for class_name in classes_list:
        if class_name not in video_df.columns:
            video_df[class_name] = 0
    # arrange dataframe in order age,age_classes,severity,emotion_Happy,emotion_Neutral,emotion_Sad,emotion_Surprise,gen_F,gen_M,emotion_Angry,emotion_Fear
    video_df = video_df[['age', 'age_classes', 'emotion_Happy', 'emotion_Neutral', 'emotion_Sad', 'emotion_Surprise',
                         'gen_F', 'gen_M', 'emotion_Angry', 'emotion_Fear']]
    # remove null values
    video_df = video_df.dropna()
    # load video model
    video_severity_model = pickle.load(open("models/severity/severity_model_video.pkl", 'rb'))
    # predict severity
    video_df['severity'] = video_severity_model.predict(video_df)
    # save df
    video_df.to_csv(run_path + "video_predicted.csv", index=False)


def AudioSentiment(run_path, word_sentiment_df):
    ##################################### audio severity #####################################
    word_sentiment_df = word_sentiment_df[['classes', 'hate', 'sentiment_word']]
    # onehot encoding on classes, gen and emotion
    word_sentiment_df = pd.get_dummies(word_sentiment_df, columns=['classes'])
    classes_list = ['classes_Sports', 'classes_Religious', 'classes_Political', 'classes_Sexual', 'classes_Education',
                    'classes_Entertainment']
    for class_name in classes_list:
        if class_name not in word_sentiment_df.columns:
            word_sentiment_df[class_name] = 0
    # arrange dataset in order hate,sentiment_word,severity,classes_Education,classes_Entertainment,classes_Political,classes_Religious,classes_Sexual,classes_Sports
    word_sentiment_df = word_sentiment_df[['hate', 'sentiment_word', 'classes_Education', 'classes_Entertainment',
                                           'classes_Political', 'classes_Religious', 'classes_Sexual',
                                           'classes_Sports']]
    # remove null values
    word_sentiment_df = word_sentiment_df.dropna()
    # load audio model
    audio_severity_model = pickle.load(open("models/severity/audio_severity_model.pkl", 'rb'))
    # predict severity
    word_sentiment_df['severity'] = audio_severity_model.predict(word_sentiment_df)
    # save df
    word_sentiment_df.to_csv(run_path + "audio_predicted.csv", index=False)


def Merging(file_name, run_path, video_df, word_sentiment_df):
    merged_df = merge_audio_and_video(word_sentiment_df, video_df)
    # add file name to the merged_df as first column
    merged_df['file_name'] = file_name
    # as first column
    cols = merged_df.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    merged_df = merged_df[cols]
    merged_df.to_csv(run_path + "merged_df.csv", index=False)
    merged_df = pd.read_csv(run_path + "merged_df.csv")
    return merged_df


def VideoPrediction(run_path, video_file):
    predict(video_file, run_path)
    video_df = pd.read_csv(run_path + "filtered_frames.csv")
    return video_df


def SentimentCalculation(hate_df, run_path):
    # word sentiment
    word_sentiment_df = calculate_sentiment_df(hate_df)
    word_sentiment_df.to_csv(run_path + "word_sentiment_df.csv", index=False)
    word_sentiment_df = pd.read_csv(run_path + "word_sentiment_df.csv")
    return word_sentiment_df


def HatePrediction(audio_df, run_path):
    # hate
    hate_df = predict_hate_df(audio_df)
    hate_df.to_csv(run_path + "hate_df.csv", index=False)
    hate_df = pd.read_csv(run_path + "hate_df.csv")
    return hate_df


def AgeNGenderEstimation(audio_df, run_path, file_name):
    # voice data
    # chunk audio by start and end time
    def chunk_audio(audio_df, file_name):
        # get start and end time
        start_time = audio_df['start_time'].tolist()
        end_time = audio_df['end_time'].tolist()

        # get the audio file
        audio_file = run_path + file_name[:-4] + ".flac"

        # get the number of chunks
        number_of_chunks = len(start_time)

        # create a list of chunks
        chunks = []

        # loop through the start and end time
        for i in range(number_of_chunks):
            # get the start and end time
            start = start_time[i]  # 26.0 float in seconds
            end = end_time[i]  # 26.1 float in seconds

            # convert to seconds
            start_sec = int(start)
            end_sec = int(end)

            if start_sec == end_sec:
                end_sec = end_sec + 1

            # get the audio chunk
            chunk, sr = librosa.load(audio_file, offset=start_sec, duration=end_sec - start_sec)

            # append to the chunks list
            chunks.append(chunk)

            # # save the aduio chunk
            # sf.write(run_path + "chunk_" + str(i) + ".wav", chunk, sr)

        return chunks

    chunks = chunk_audio(audio_df, file_name)
    # get the features
    features_df = get_features_chunk(chunks)
    features_df.to_csv(run_path + "features.csv", index=False)
    features_df = pd.read_csv(run_path + "features.csv")
    # get the age
    age_df = voice_age_predict_df(features_df)
    gen_df = voice_gen_predict_df(features_df)
    # merge the features with the audio_df
    audio_df = pd.concat([audio_df, age_df["audio_age_range"], gen_df["audio_gen"]], axis=1)
    audio_df.to_csv(run_path + "recognized_processed_classes_features_age_gen.csv", index=False)
    audio_df = pd.read_csv(run_path + "recognized_processed_classes_features_age_gen.csv")
    return audio_df


def DomainPrediction(audio_df, run_path):
    # classes
    audio_df = predict_classes_df(audio_df)
    audio_df.to_csv(run_path + "recognized_processed_classes.csv", index=False)
    audio_df = pd.read_csv(run_path + "recognized_processed_classes.csv")
    return audio_df


def TranscribingNProcessing(file_name, run_number, run_path):
    processing_audio(run_path + file_name[:-4] + ".flac", run_path, run_number)
    audio_df = pd.read_csv(run_path + "recognized.csv")
    audio_df = process_data_df(audio_df, column_name='text')
    audio_df.to_csv(run_path + "recognized_processed.csv", index=False)
    audio_df = pd.read_csv(run_path + "recognized_processed.csv")
    # remove null
    audio_df = audio_df.dropna()
    return audio_df


def AudioConversion(file_name, run_path, video_file):
    convert_video_to_audio_ffmpeg(video_file, run_path, file_name[:-4])
    wav_to_mono_flac(run_path + file_name[:-4] + ".wav")


def MergedSentiment(merged_df, run_path):
    ##################################### mergeed severity #####################################
    merged_df = merged_df[
        ['classes', 'audio_age_range', 'audio_gen', 'hate', 'sentiment_word', 'age', 'gen', 'emotion', 'age_classes']]
    # onehot encoding on classes, gen and emotion
    merged_df = pd.get_dummies(merged_df, columns=['classes', 'gen', 'emotion'])
    classes_list = ['emotion_Angry', 'emotion_Fear', 'emotion_Happy', 'emotion_Neutral', 'emotion_Sad',
                    'emotion_Surprise', 'gen_F', 'gen_M', 'classes_Sports', 'classes_Religious', 'classes_Political',
                    'classes_Sexual', 'classes_Education', 'classes_Entertainment']
    for class_name in classes_list:
        if class_name not in merged_df.columns:
            merged_df[class_name] = 0
    # arrange dataframe in order audio_age_range,audio_gen,hate,sentiment_word,age,age_classes,severity,classes_Entertainment,classes_Political,classes_Religious,classes_Sexual,gen_F,gen_M,emotion_Neutral,emotion_Sad,emotion_Angry,emotion_Fear,emotion_Happy,emotion_Surprise,classes_Sports,classes_Education
    merged_df = merged_df[['audio_age_range', 'audio_gen', 'hate', 'sentiment_word', 'age', 'age_classes',
                           'classes_Entertainment', 'classes_Political', 'classes_Religious', 'classes_Sexual',
                           'gen_F', 'gen_M', 'emotion_Neutral', 'emotion_Sad', 'emotion_Angry', 'emotion_Fear',
                           'emotion_Happy', 'emotion_Surprise', 'classes_Sports', 'classes_Education']]
    # remove null values
    merged_df = merged_df.dropna()
    # load merged model
    merged_severity_model = pickle.load(open("models/severity/severity_model_merged.pkl", 'rb'))
    # predict severity
    merged_df['severity'] = merged_severity_model.predict(merged_df)
    # save df
    merged_df.to_csv(run_path + "merged_predicted.csv", index=False)
    print('Severity is predicted.')


def merge_audio_and_video(audio_df, video_df):
    # print audio_df columns
    print("\n\n\n\n\n\n\n\n\n\n\n")
    print(audio_df.columns)
    print(video_df.head(10))

    merged_df = sync_audio_and_video(audio_df, video_df)
    print(merged_df.head(10))

    print('Audio and Video are Merged.')

    return merged_df


def chunkify(df, num, n=10):
    new_df = pd.DataFrame()
    global chunk_df
    num_of_chunks = len(df) // n

    for i in range(num_of_chunks + 1):
        start = i * 10
        end = start + 10
        chunk_df = df[start:end]

        # join word in word column
        joined_text = " ".join(chunk_df['word'].tolist())

        # replace text column with joined text
        chunk_df['text'].replace(chunk_df['text'].tolist(), joined_text, inplace=True)

        # concat to new_df
        new_df = pd.concat([new_df, chunk_df])

    # save csv
    new_df.to_csv("runs/run_" + str(num) + "/recognized_chunked.csv", index=False)

    return new_df


def processing_audio(video_file, run_path, num):
    print('Processing Audio Started...')

    print("Transcribing audio")
    transcribed_text, df = get_large_audio_transcription(video_file, run_path)
    print(transcribed_text)
    print(df)

    # save csv
    df.to_csv(run_path + "recognized_gcloud.csv", index=False)

    print("Chunkify")
    df = chunkify(df, num)
    # df.to_csv(run_path + "/recognized_chunked.csv", index=False)
    # df = pd.read_csv(run_path + "/recognized_chunked.csv")

    print("Process")
    new_df = process_with_timestamp(df)
    print(new_df)
    new_df.to_csv(run_path + "recognized.csv", index=False)

    print('Processing Audio Finished...')

    return transcribed_text, new_df


def create_folder(location, file_name):
    global number

    if not os.path.exists(location + "/run_0"):
        os.makedirs(location + "/run_0")
        number = -1
    else:
        # then we need create it as run_0, run_1, ...
        # get all folders starting with run + "_"
        folders = [f for f in os.listdir(location) if re.match("run_", f)]
        # get the last folder
        last_folder = sorted(folders)[-1]
        # get the number
        number = int(last_folder.split("_")[-1])
        # create new folder
        os.mkdir(location + "/run" + "_" + str(number + 1))

    print(number + 1)
    # create log_gen.txt file inside the folder
    with open(location + "run" + "_" + str(number + 1) + "/log_gen.txt", "w", encoding="utf-8") as f:
        f.write("=======================================================\n")
        f.write("Run " + str(number + 1) + " started\n")
        f.write("Start Time: " + str(datetime.datetime.now()) + "\n")
        f.write("=======================================================\n\n")
        f.write("File Name: " + file_name + "\n")

    return number + 1
