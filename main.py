import datetime
import os
import re
import librosa
import pandas as pd
from audio.video_to_audio import convert_video_to_audio_ffmpeg, wav_to_mono_flac
from audio.transcribe_data_gcloud import get_large_audio_transcription
from audio.process_data import process_data_df
from audio.process_with_timestamp import process_with_timestamp
from Audio_FYP.classes.classes import processing_audio_classes
from voice_classifier.age_predictor import voice_age_predict_df
from voice_classifier.gen_predictor import voice_gen_predict_df
from voice_classifier.featureExtractor import get_features_chunk

# show all columns
pd.set_option('display.max_columns', None)

def processing_audio(video_file, run_path):
    print('Processing Audio Started...')

    print("Transcribing audio")
    transcribed_text, df = get_large_audio_transcription(video_file, run_path)
    print(transcribed_text)
    print(df)

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
    with open(location + "run" + "_" + str(number + 1) + "/log_gen.txt", "w") as f:
        f.write("=======================================================\n")
        f.write("Run " + str(number + 1) + " started\n")
        f.write("Start Time: " + str(datetime.datetime.now()) + "\n")
        f.write("=======================================================\n\n")
        f.write("File Name: " + file_name + "\n")

    return number + 1


if __name__ == '__main__':
    video_file = "test_files/Malinga.mkv"

    # get file name spit by / or \
    file_name = re.split(r'[/\\]', video_file)[-1]

    # create a folder inside runs folder if not exists if not exists print "file exists"
    run_number = create_folder("runs/", file_name)
    # run_number = 5
    run_path = "runs/run_" + str(run_number) + "/"

    convert_video_to_audio_ffmpeg(video_file, run_path, file_name[:-4])
    wav_to_mono_flac(run_path + file_name[:-4] + ".wav")
    processing_audio(run_path + file_name[:-4] + ".flac", run_path)
    audio_df = pd.read_csv(run_path + "recognized.csv")

    audio_df = process_data_df(audio_df, column_name='text')
    audio_df.to_csv(run_path + "recognized_processed.csv", index=False)
    audio_df = pd.read_csv(run_path + "recognized_processed.csv")

    # remove null
    audio_df = audio_df.dropna()

    # classes
    audio_df = processing_audio_classes(audio_df)
    audio_df.to_csv(run_path + "recognized_processed_classes.csv", index=False)
    audio_df = pd.read_csv(run_path + "recognized_processed_classes.csv")


    # voice data
    # chunk audio by start and end time
    def chunk_audio(audio_df):
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


    chunks = chunk_audio(audio_df)

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
