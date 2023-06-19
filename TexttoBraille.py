import codecs
import csv
import glob
import os
import subprocess
import tkinter as tk
from datetime import datetime
from time import sleep
from tkinter import *
from tkinter import filedialog

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageTk
from keras.utils import img_to_array
from pytube import YouTube
import yt_dlp
import threading

from SubProcesses import video_severity, audio_severity, Merging, VideoPrediction, SentimentCalculation, \
    HatePrediction, \
    AgeNGenderEstimation, DomainPrediction, TranscribingNProcessing, AudioConversion, merged_severity, create_folder

# from lib import predict

currentPath = os.getcwd()
openFilePath = os.path.join(currentPath, "test_images")
placeHolderImagePath = r"placeholder.jpg"
data = []
default_model = "model_3"
model_set = {
    "original": "model_1",
    "2 LSTM": "model_2",
    "1 LSTM": "model_3"
}


def pickImage():
    file = filedialog.askopenfilename(initialdir=openFilePath, title='Select File',
                                      filetypes=(('JPG', '*.jpg'), ('All Files', '*.*')))
    url_text.set(file)

    # if the image is less than 120x120, resize it
    image = cv2.imread(file)
    height, width, channels = image.shape

    global scaled_image

    """LOAD IMAGE"""
    # if the image is 256x256
    if height == 256 and width == 256:
        # show the image in the gui
        image1 = Image.open(file)
        image1 = image1.resize((256, 256), Image.ANTIALIAS)
        img = ImageTk.PhotoImage(image1)
        label = Label(middle, image=img, bg='red')
        label.image = img
        label.place(x=80, y=100)


def Download(link):
    # youtubeObject = YouTube(link, use_oauth=True, allow_oauth_cache=True)
    # youtubeObject = youtubeObject.streams.get_highest_resolution()

    ydl_opts = {
        'format': 'best',
        'outtmpl': '%(title)s.%(ext)s',
    }

    file_name = yt_dlp.YoutubeDL(ydl_opts) \
        .prepare_filename(yt_dlp.YoutubeDL(ydl_opts).extract_info(link, download=False))
    print(file_name)

    # create a folder inside runs folder if not exists if not exists print "file exists"
    run_number = create_folder("runs/", file_name)
    # run_number = 3
    run_path = "runs/run_" + str(run_number) + "/"

    try:
        # youtubeObject.download(output_path=run_path)
        ydl_opts = {
            'format': 'best',
            'outtmpl': run_path + '%(title)s.%(ext)s',
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([link])

        print("downloaded", link)
    except:
        print("An error has occurred")
    print("Download is completed successfully")


    # get the .mp4 file in the runs folder
    run_path_file = glob.glob(run_path + "*.mp4")[0]
    print("run_path_file: ", run_path_file)
    # get the file name
    run_path_file = run_path_file.split("\\")[-1]

    # compare the file name with the file name in the runs folder
    if run_path_file == file_name:
        print("file name is same")
    else:
        print("file name is not same")
        file_name = run_path_file


    return run_path, file_name, run_number


def toggle_single_item(index):
    step_status = steps[index][1]  # Get the status variable for the specified index
    lbl_step_status = lbl_step_statuses[index]  # Get the corresponding label

    step_status.set(not step_status.get())  # Toggle the status variable

    if step_status.get() == True:
        lbl_step_status.config(text="✅")  # Update the label text to display a tick mark
    else:
        lbl_step_status.config(text=" ")  # Update the label text to display an empty space


def showImage():
    url = url_text.get()
    # download the video
    if url.startswith("https://www.youtube.com/watch?v="):
        run_path, file_name, run_number = Download(url)
    else:
        print("Not a YouTube link")

    toggle_single_item(0)

    # # run_number = 7
    # # file_name = "People X Nainowale Ne  Chillout Mashup  @YashrajMukhateOfficial   MEHER.mp4"
    # # run_path = "runs/run_6/"
    #
    # print("run_path: ", run_path)
    # print("file_name: ", file_name)
    #
    # AudioConversion(file_name, run_path, file_name)
    # toggle_single_item(1)
    #
    # audio_df = TranscribingNProcessing(file_name, run_number, run_path)
    # toggle_single_item(2)
    #
    # sleep(50)
    # print("sleeping for 50 seconds")
    # print("sleeping for 50 seconds")
    # print("sleeping for 50 seconds")
    # print("sleeping for 50 seconds")
    # print("sleeping for 50 seconds")
    # print("sleeping for 50 seconds")
    # print("sleeping for 50 seconds")
    # print("sleeping for 50 seconds")
    # print("sleeping for 50 seconds")
    # print("sleeping for 50 seconds")
    #
    # audio_df = DomainPrediction(audio_df, run_path)
    # toggle_single_item(3)
    #
    # audio_df = AgeNGenderEstimation(audio_df, run_path, file_name)
    # toggle_single_item(4)
    #
    # hate_df = HatePrediction(audio_df, run_path)
    # toggle_single_item(5)
    #
    # word_sentiment_df = SentimentCalculation(hate_df, run_path)
    # toggle_single_item(6)

    print(run_path)
    print(file_name)
    video_df = VideoPrediction(run_path, file_name)

    # is run_path + file_name exists
    if os.path.exists(run_path + file_name):
        # show video in player
        video = os.startfile(run_path + file_name)
        # read the video
        cap = cv2.VideoCapture(run_path + file_name)
        # get the frame count
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print("frame_count: ", frame_count)
    else:
        print("file not exists")
        print("file not exists")
        print("file not exists")

    if not video_df.empty:
        # show image in left side of the content
        img = ImageTk.PhotoImage(Image.open(run_path + "Age vs Frames.png"))
        panel = tk.Label(left_subframe, image=img)
        panel.pack(side="bottom", fill="both", expand="yes")

        # show image in left side of the content
        img = ImageTk.PhotoImage(Image.open(run_path + "Age vs Frames.png"))
        panel = tk.Label(left_subframe, image=img)
        panel.pack(side="bottom", fill="both", expand="yes")

    toggle_single_item(7)

    # print(audio_df.head())
    # print(video_df.head())
    #
    # merged_df = Merging(file_name, run_path, video_df, word_sentiment_df)
    # toggle_single_item(8)
    #
    # if not word_sentiment_df.empty:
    #     audio_severity_done = audio_severity(run_path, word_sentiment_df)
    #     if audio_severity_done:
    #         toggle_single_item(9)
    # else:
    #     print("audio severity not done")
    #     audio_severity_done = False
    #
    #
    # if not video_df.empty:
    #     video_severity_done = video_severity(run_path, video_df)
    #     if video_severity_done:
    #         toggle_single_item(10)
    # else:
    #     print("video severity not done")
    #     video_severity_done = False
    #
    # if not merged_df.empty:
    #     merged_severity_done = merged_severity(run_path, merged_df)
    #     if merged_severity_done:
    #         toggle_single_item(11)
    # else:
    #     print("merged severity not done")
    #     merged_severity_done = False
    #
    #
    # ##################################### Finalizing #####################################
    # if audio_severity_done:
    #     audio_df = pd.read_csv(run_path + "audio_predicted.csv")[['hate', 'severity']]
    #     print("Mean of the audio hate: ", np.mean(audio_df['hate']), " - ", np.round(np.mean(audio_df['hate']) * 100, 2),
    #           "%")
    #     print("Mean of the audio severity: ", np.mean(audio_df['severity']), " - ",
    #           np.round(np.mean(audio_df['severity']) * 25, 2), "%")
    #     audio_hate_mean = np.round(np.mean(audio_df['hate']) * 100, 2)
    #     audio_severity_mean = np.round(np.mean(audio_df['severity']) * 25, 2)
    # else:
    #     audio_hate_mean = "NaN"
    #     audio_severity_mean = "NaN"
    #
    # if video_severity_done:
    #     video_df = pd.read_csv(run_path + "video_predicted.csv")[['severity']]
    #     print("Mean of the video severity: ", np.mean(video_df['severity']), " - ",
    #           np.round(np.mean(video_df['severity']) * 25, 2), "%")
    #     video_severity_mean = np.round(np.mean(video_df['severity']) * 25, 2)
    # else:
    #     video_severity_mean = "NaN"
    #
    #
    # if audio_severity_done and video_severity_done and merged_severity_done:
    #     merged_df = pd.read_csv(run_path + "merged_predicted.csv")[['hate', 'severity']]
    #     print("Mean of the merged severity: ", np.mean(merged_df['severity']), " - ",
    #           np.round(np.mean(merged_df['severity']) * 25, 2), "%")
    #     merged_severity_mean = np.round(np.mean(merged_df['severity']) * 25, 2)
    # else:
    #     merged_severity_mean = "NaN"
    #
    # # create new fram below content frame
    #
    #
    # # audio_hate_mean_label = Label(bottom_frame, text=audio_hate_mean)
    # # audio_hate_mean_label.grid(row=1, column=1)
    # #
    # # audio_severity_mean_label = Label(bottom_frame, text=audio_severity_mean)
    # # audio_severity_mean_label.grid(row=2, column=1)
    # #
    # # video_severity_mean_label = Label(bottom_frame, text=video_severity_mean)
    # # video_severity_mean_label.grid(row=3, column=1)
    # #
    # # merged_severity_mean_label = Label(bottom_frame, text=merged_severity_mean)
    # # merged_severity_mean_label.grid(row=4, column=1)
    #
    # bottom_frame_column_gap = 40
    # bottom_frame_title_font_size = 12
    # bottom_frame_title_font = ("Arial Bold", bottom_frame_title_font_size)
    # bottom_frame_title_bg = 'white'
    # bottom_frame_title_fg = 'black'
    # bottom_frame_value_font_size = 12
    # bottom_frame_value_font = ("Arial Bold", bottom_frame_value_font_size)
    # bottom_frame_value_bg = 'white'
    # bottom_frame_value_fg = 'black'
    #
    # audio_hate_mean_label = Label(bottom_frame, text="Audio Hate Mean", font=bottom_frame_title_font, bg=bottom_frame_title_bg, fg=bottom_frame_title_fg)
    # audio_hate_mean_label.grid(row=0, column=0, padx=bottom_frame_column_gap)
    # audio_hate_mean_title = Label(bottom_frame, text=audio_hate_mean, font=bottom_frame_value_font, bg=bottom_frame_value_bg, fg=bottom_frame_value_fg)
    # audio_hate_mean_title.grid(row=1, column=0, padx=bottom_frame_column_gap)
    #
    # audio_severity_mean_label_title = Label(bottom_frame, text="Audio Severity Mean", font=bottom_frame_title_font, bg=bottom_frame_title_bg, fg=bottom_frame_title_fg)
    # audio_severity_mean_label_title.grid(row=0, column=1, padx=bottom_frame_column_gap)
    # audio_severity_mean_label = Label(bottom_frame, text=audio_severity_mean, font=bottom_frame_value_font, bg=bottom_frame_value_bg, fg=bottom_frame_value_fg)
    # audio_severity_mean_label.grid(row=1, column=1, padx=bottom_frame_column_gap)
    #
    # video_hate_mean_label_title = Label(bottom_frame, text="Video Hate Mean", font=bottom_frame_title_font, bg=bottom_frame_title_bg, fg=bottom_frame_title_fg)
    # video_hate_mean_label_title.grid(row=0, column=2, padx=bottom_frame_column_gap)
    # video_hate_mean_label = Label(bottom_frame, text="NaN", font=bottom_frame_value_font, bg=bottom_frame_value_bg, fg=bottom_frame_value_fg)
    # video_hate_mean_label.grid(row=1, column=2, padx=bottom_frame_column_gap)
    #
    # video_severity_mean_label_title = Label(bottom_frame, text="Video Severity Mean", font=bottom_frame_title_font, bg=bottom_frame_title_bg, fg=bottom_frame_title_fg)
    # video_severity_mean_label_title.grid(row=0, column=3, padx=bottom_frame_column_gap)
    # merged_severity_mean_label = Label(bottom_frame, text=video_severity_mean, font=bottom_frame_value_font, bg=bottom_frame_value_bg, fg=bottom_frame_value_fg)
    # merged_severity_mean_label.grid(row=1, column=3, padx=bottom_frame_column_gap)
    #
    #
    # def download_audio_csv():
    #     # Open the CSV file with UTF-8 encoding
    #     csv_file_path = (run_path + "recognized_processed_classes_features_age_gen.csv")
    #     open_csv_files(csv_file_path)
    #
    # def download_video_csv():
    #     csv_file_path = (run_path + "filtered_frames.csv")
    #     open_csv_files(csv_file_path)
    #
    # def download_hate_sentiment_csv():
    #     csv_file_path = (run_path + "word_sentiment_df.csv")
    #     open_csv_files(csv_file_path)
    #
    # def download_merged_csv():
    #     csv_file_path = (run_path + "merged_df.csv")
    #     open_csv_files(csv_file_path)
    #
    #
    #
    # # add buttons to view the result in csv format (open csv supported external application) in a new frame
    # audio_download_button = tk.Button(download_frame, text="Download Audio CSV", command=download_audio_csv)
    # audio_download_button.grid(row=0, column=0, padx=bottom_frame_column_gap)
    #
    # video_download_button = tk.Button(download_frame, text="Download Video CSV", command=download_video_csv)
    # video_download_button.grid(row=0, column=1, padx=bottom_frame_column_gap)
    #
    # merged_download_button = tk.Button(download_frame, text="Download Merged CSV", command=download_merged_csv)
    # merged_download_button.grid(row=0, column=2, padx=bottom_frame_column_gap)
    #
    # hate_sentiment_download_button = tk.Button(download_frame, text="Download Hate Sentiment CSV", command=download_hate_sentiment_csv)
    # hate_sentiment_download_button.grid(row=0, column=3, padx=bottom_frame_column_gap)
    #






# Function to run showImage in a separate thread
def run_showImage():
    threading.Thread(target=showImage).start()

# Define a function to toggle the status variable and label
def toggle_status(status_var, label):
    status_var.set(not status_var.get())  # Toggle the status variable

    if status_var.get() == True:
        label.config(text="✅")  # Update the label text to display a tick mark
    else:
        label.config(text=" ")  # Update the label text to display an empty space





def open_csv_files(csv_file_path):
    # Open the CSV file with UTF-8 encoding
    with codecs.open(csv_file_path, "r", encoding="utf-8") as file:
        # Read the CSV data
        csv_data = csv.reader(file)

        # Create a temporary file with UTF-16 encoding to save the data
        temp_file_path = "temp_file.csv"
        with open(temp_file_path, "w", encoding="utf-16") as temp_file:
            # Write the non-empty rows to the temporary file with UTF-16 encoding
            csv.writer(temp_file, delimiter="\t").writerows(
                row for row in csv_data if any(field.strip() for field in row))
    # Open the temporary file with the default associated application
    os.startfile(temp_file_path)


def tksleep(t):
    ms = int(t)
    root = tk._get_default_root()
    var = tk.IntVar(root)
    root.after(ms, lambda: var.set(1))
    root.wait_variable(var)


if __name__ == '__main__':
    # Create Object and setup root
    root = Tk()
    root.title("Youtube Hate Detection")

    # Create Frames
    top = Frame(root, width=1200, height=100, bg='white')
    top.pack(side=TOP)
    middle = Frame(root, width=1200, height=100, bg='white')
    middle.pack(side=TOP)
    # create 2 horizontal frames in the content frame left and right
    content = Frame(root, width=1200, height=420, bg='white')
    content.pack(side=TOP)
    bottom_frame = Frame(root, bg="white", width=1000, height=100)
    bottom_frame.pack(side=TOP)

    # string variable with default value "model_1"
    model_choice = StringVar(value=default_model)
    # file name variable
    url_text = StringVar()
    result_text_var = StringVar()
    predicted_var = StringVar()

    # Audio conversion
    # Transcribing and processing
    # Domain prediction
    # Age & Gender estimation
    # Hate prediction
    # Sentiment calculation
    # Video prediction
    # Merging
    # Audio severity
    # Video severity
    # Merged severity

    # True or False for each step default False
    downloading_status = BooleanVar(value=False)
    audio_conversion_status = BooleanVar(value=False)
    transcribing_n_processing_status = BooleanVar(value=False)
    domain_prediction_status = BooleanVar(value=False)
    age_and_gender_estimation_status = BooleanVar(value=False)
    hate_prediction_status = BooleanVar(value=False)
    sentiment_calculation_status = BooleanVar(value=False)
    video_prediction_status = BooleanVar(value=False)
    merging_status = BooleanVar(value=False)
    audio_severity_status = BooleanVar(value=False)
    video_severity_status = BooleanVar(value=False)
    merged_severity_status = BooleanVar(value=False)

    # Create Widgets

    """Top Section"""
    # create a label as title in center
    lbl = tk.Label(top, text="Youtube Hate Detection", font=("Arial Bold", 36), fg='black')
    lbl.grid()

    """Image Section"""
    # On right side of the image show a text input
    lbl = tk.Label(middle, text="Input", font=("Arial Bold", 15), bg='blue')
    lbl.place(x=80, y=40)

    # show input text box to enter youtube url right side of the label
    input_text = tk.Entry(middle, width=50, textvariable=url_text, font=("Arial Bold", 15), bg='white')
    input_text.place(x=150, y=40)

    # Button right to text input to call main()
    btn = tk.Button(middle, text="Predict", command=run_showImage, font=("Arial Bold", 15), bg='white')
    btn.place(x=800, y=40)

    # # show textvariable value in label
    # input_text = tk.Label(middle, textvariable=file_name, width=50, bg='white')
    # input_text.place(x=80, y=100)

    """Content Section"""

    # Create the left subframe for "Some Text"
    left_subframe = tk.Frame(content, width=600, bg='white')
    left_subframe.pack(side=tk.LEFT)
    lbl_some_text = tk.Label(left_subframe, text="                                                        ",
                             font=("Arial Bold", 15), bg='white', anchor='w', justify='left')
    lbl_some_text.pack(padx=80, pady=0)

    # Create the right subframe for code segment 1
    right_subframe = tk.Frame(content, width=600, bg='white')
    right_subframe.pack(side=tk.LEFT)

    timeline_font_size = 12
    timeline_font = ("Arial Bold", timeline_font_size)
    timeline_tick_font_size = 12
    timeline_tick_font = ("Arial Bold", timeline_font_size)
    timeline_bg = 'white'
    timeline_fg = 'black'
    timeline_anchor = 'w'
    timeline_justify = 'left'
    timeline_padx = 80
    timeline_pady = 0
    timeline_sticky = 'w'

    # Define a list of steps or tasks
    steps = [
        ("Downloading", downloading_status),
        ("Audio Prediction", audio_conversion_status),
        ("Transcription", transcribing_n_processing_status),
        ("Domain Prediction", domain_prediction_status),
        ("Age and Gender Estimation", age_and_gender_estimation_status),
        ("Hate Prediction", hate_prediction_status),
        ("Sentiment Calculation", sentiment_calculation_status),
        ("Video Prediction", video_prediction_status),
        ("Merging", merging_status),
        ("Audio Severity", audio_severity_status),
        ("Video Severity", video_severity_status),
        ("Merged Severity", merged_severity_status)
    ]

    # Create labels dynamically for each step
    for i, (step_text, step_status) in enumerate(steps):
        lbl_step = tk.Label(right_subframe, text=step_text, font=timeline_font, bg=timeline_bg, fg=timeline_fg,
                            anchor=timeline_anchor, justify=timeline_justify)
        lbl_step.grid(row=i + 2, column=0, padx=timeline_padx, pady=timeline_pady, sticky=timeline_sticky)

        if step_status.get() == True:
            lbl_step_status = tk.Label(right_subframe, text="✅", font=timeline_tick_font, bg=timeline_bg,
                                       fg=timeline_fg)
        else:
            lbl_step_status = tk.Label(right_subframe, text=" ", font=timeline_tick_font, bg=timeline_bg,
                                       fg=timeline_fg)

        lbl_step_status.grid(row=i + 2, column=1, padx=timeline_padx, pady=timeline_pady, sticky=timeline_sticky)


    # Create labels dynamically for each step
    lbl_step_statuses = []  # List to store the label references
    for i, (step_text, step_status) in enumerate(steps):
        lbl_step = tk.Label(right_subframe, text=step_text, font=timeline_font, bg=timeline_bg, fg=timeline_fg,
                            anchor=timeline_anchor, justify=timeline_justify)
        lbl_step.grid(row=i + 2, column=0, padx=timeline_padx, pady=timeline_pady, sticky=timeline_sticky)

        lbl_step_status = tk.Label(right_subframe, text=" ", font=timeline_tick_font, bg=timeline_bg, fg=timeline_fg)
        lbl_step_status.grid(row=i + 2, column=1, padx=timeline_padx, pady=timeline_pady, sticky=timeline_sticky)

        lbl_step_statuses.append(lbl_step_status)  # Add the label reference to the list

    # # Example: Button to toggle a specific step
    # btn_toggle = tk.Button(right_subframe, text="Toggle Step 2",
    #                        command=lambda: toggle_single_item(1))  # Specify the index of the step to toggle
    # btn_toggle.grid(row=2, column=2, padx=timeline_padx, pady=timeline_pady, sticky=timeline_sticky)


    content.pack()

    # # show image in left side of the content
    # img = ImageTk.PhotoImage(Image.open(run_path + "Age vs Frames.png"))
    # panel = tk.Label(left_subframe, image=img)
    # panel.pack(side="bottom", fill="both", expand="yes")
    #
    # # show image in left side of the content
    # img = ImageTk.PhotoImage(Image.open(run_path + "Age vs Frames.png"))
    # panel = tk.Label(left_subframe, image=img)
    # panel.pack(side="bottom", fill="both", expand="yes")


    """Bottom Section"""
    bottom_frame = tk.Frame(root, width=1200, bg='white')
    bottom_frame.pack(side=tk.TOP)

    # add buttons to view the result in csv format (open csv supported external application) in a new frame
    download_frame = tk.Frame(root, width=1200, bg='white')
    download_frame.pack(side=tk.TOP)






    root.mainloop()

# https://www.youtube.com/watch?v=DuDRAZNIjdU
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
