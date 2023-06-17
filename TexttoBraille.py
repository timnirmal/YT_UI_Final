import os
import tkinter as tk
from datetime import datetime
from time import sleep
from tkinter import *
from tkinter import filedialog

import cv2
import numpy as np
from PIL import Image, ImageTk
from keras.utils import img_to_array
from pytube import YouTube

from SubProcesses import video_sentiment, AudioSentiment, Merging, VideoPrediction, SentimentCalculation, \
    HatePrediction, \
    AgeNGenderEstimation, DomainPrediction, TranscribingNProcessing, AudioConversion, MergedSentiment, create_folder

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
    youtubeObject = YouTube(link, use_oauth=True, allow_oauth_cache=True)
    youtubeObject = youtubeObject.streams.get_highest_resolution()

    # create a folder inside runs folder if not exists if not exists print "file exists"
    run_number = create_folder("runs/", youtubeObject.default_filename)
    # run_number = 3
    run_path = "runs/run_" + str(run_number) + "/"

    try:
        youtubeObject.download(output_path=run_path)
    except:
        print("An error has occurred")
    print("Download is completed successfully")

    return run_path, youtubeObject.default_filename, run_number


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
        print("Not a youtube link")

    # run_number = 7
    # file_name = "People X Nainowale Ne  Chillout Mashup  @YashrajMukhateOfficial   MEHER.mp4"
    # run_path = "runs/run_6/"

    print("run_path: ", run_path)
    print("file_name: ", file_name)

    AudioConversion(file_name, run_path, file_name)

    audio_df = TranscribingNProcessing(file_name, run_number, run_path)
    #
    #
    # # if the image is less than 120x120, resize it
    # image = cv2.imread(file)
    # height, width, channels = image.shape
    #
    # global scaled_image
    #
    # """LOAD IMAGE"""
    # # if the image is 256x256
    # if height == 256 and width == 256:
    #     # show the image in the gui
    #     image1 = Image.open(file)
    #     image1 = image1.resize((256, 256), Image.ANTIALIAS)
    #     img = ImageTk.PhotoImage(image1)
    #     label = Label(middle, image=img, bg='red')
    #     label.image = img
    #     label.place(x=80, y=100)
    #
    #     image_read = cv2.imread(file)
    #     image_read = cv2.resize(image_read, (256, 256))
    #     image_read = img_to_array(image_read)
    #     image_read = np.expand_dims(image_read, axis=0)
    #     image_read = image_read / 255.0
    #
    # predict_result, result_text = predict.predict(image_read, model=model_choice.get())
    #
    # # get list of images in result folder
    # result_folder = os.path.join(currentPath, "result")
    # result_images = os.listdir(result_folder)
    #
    # # show the result_images in the gui one by one with a 1 second delay
    # for i in range(len(result_images)):
    #     # show the image in the gui
    #     image1 = Image.open(os.path.join(result_folder, result_images[i]))
    #     # resize keeping aspect ratio
    #     image_sizes = image1.size
    #     SIZE = 400
    #     if image_sizes[0] > SIZE:
    #         image1 = image1.resize((SIZE, int(image_sizes[1] * SIZE / image_sizes[0])), Image.ANTIALIAS)
    #     img = ImageTk.PhotoImage(image1)
    #     label = Label(middle, image=img, bg='red')
    #     label.image = img
    #     label.place(x=400, y=100)
    #
    #     # sleep for 500 millisecond
    #     tksleep(500)
    #
    # result_text_var.set(result_text)
    # predicted_var.set(predict_result)
    #
    # # show the result in the gui after prediction
    # result = Label(middle, text=result_text_var.get(), font=("Arial", 10), bg='white')
    # result.place(x=1000, y=100)
    #
    # # text for prediction
    # predict_val = predicted_var.get()
    # # remove [ and ] from the string by replacing them with empty string
    # predict_val = predict_val.replace("[", "")
    # predict_val = predict_val.replace("]", "")
    # predicted = Label(middle, text=predict_val, font=("Arial", 10), bg='white')
    # predicted.place(x=1000, y=150)
    #
    # # text for prediction
    # # find model name by value
    # model = Label(middle, text=list(model_set.keys())[list(model_set.values()).index(model_choice.get())],
    #               font=("Arial", 10), bg='white')
    # model.place(x=1000, y=200)


# Define a function to toggle the status variable and label
def toggle_status(status_var, label):
    status_var.set(not status_var.get())  # Toggle the status variable

    if status_var.get() == True:
        label.config(text="✅")  # Update the label text to display a tick mark
    else:
        label.config(text=" ")  # Update the label text to display an empty space


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
    btn = tk.Button(middle, text="Predict", command=showImage, font=("Arial Bold", 15), bg='white')
    btn.place(x=800, y=40)

    # # show textvariable value in label
    # input_text = tk.Label(middle, textvariable=file_name, width=50, bg='white')
    # input_text.place(x=80, y=100)

    """Content Section"""

    # # Create a label as Audio Conversion with ✅
    # lbl_audio_conversion = tk.Label(content, text="Audio Conversion", font=("Arial Bold", 15), bg='white')
    # x_dis = 80
    # y_dis = 0
    # # grid with column 0 and row 0 and dis as x and y
    # lbl_audio_conversion.grid(column=0, row=0, padx=x_dis, pady=y_dis)
    # if audio_conversion_status.get() == True:
    #     lbl_audio_conversion_status = tk.Label(content, text="✅", font=("Arial Bold", 15), bg='white')
    # else:
    #     lbl_audio_conversion_status = tk.Label(content, text="", font=("Arial Bold", 15), bg='white')
    # lbl_audio_conversion_status.grid(column=1, row=0, padx=x_dis, pady=y_dis)
    #
    # # Create a label as Transcribing and Processing with ✅
    # lbl_transcribing_n_processing = tk.Label(content, text="Transcribing and Processing", font=("Arial Bold", 15), bg='white')
    # # grid with column 0 and row 1 and dis as x and y
    # lbl_transcribing_n_processing.grid(column=0, row=1, padx=x_dis, pady=y_dis)
    # if transcribing_n_processing_status.get() == True:
    #     lbl_transcribing_n_processing_status = tk.Label(content, text="✅", font=("Arial Bold", 15), bg='white')
    # else:
    #     lbl_transcribing_n_processing_status = tk.Label(content, text="", font=("Arial Bold", 15), bg='white')
    # lbl_transcribing_n_processing_status.grid(column=1, row=1, padx=x_dis, pady=y_dis)

    # Create the third frame with two columns
    # third_frame = tk.Frame(root, width=1200, bg='blue')  # Set the width to 1200

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
    #
    #
    # lbl_audio_conversion = tk.Label(right_subframe, text="Audio Conversion", font=timeline_font, bg=timeline_bg, fg=timeline_fg, anchor=timeline_anchor, justify=timeline_justify)
    # lbl_audio_conversion.grid(row=0, column=0, padx=timeline_padx, pady=timeline_pady, sticky=timeline_sticky)
    # if audio_conversion_status.get() == True:
    #     lbl_audio_conversion_status = tk.Label(right_subframe, text="✅", font=timeline_tick_font, bg=timeline_bg, fg=timeline_fg)
    # else:
    #     lbl_audio_conversion_status = tk.Label(right_subframe, text=" ", font=timeline_tick_font, bg=timeline_bg, fg=timeline_fg)
    # lbl_audio_conversion_status.grid(row=0, column=1, padx=timeline_padx, pady=timeline_pady, sticky=timeline_sticky)
    #
    #
    # lbl_transcribing_n_processing = tk.Label(right_subframe, text="Transcribing and Processing", font=timeline_font, bg=timeline_bg, fg=timeline_fg, anchor=timeline_anchor, justify=timeline_justify)
    # lbl_transcribing_n_processing.grid(row=1, column=0, padx=timeline_padx, pady=timeline_pady, sticky=timeline_sticky)
    # if transcribing_n_processing_status.get() == True:
    #     lbl_transcribing_n_processing_status = tk.Label(right_subframe, text="✅", font=timeline_tick_font, bg=timeline_bg, fg=timeline_fg)
    # else:
    #     lbl_transcribing_n_processing_status = tk.Label(right_subframe, text=" ", font=timeline_tick_font, bg=timeline_bg, fg=timeline_fg)
    # lbl_transcribing_n_processing_status.grid(row=1, column=1, padx=timeline_padx, pady=timeline_pady, sticky=timeline_sticky)
    #
    # # Create a label as domain prediction with ✅
    # lbl_domain_prediction = tk.Label(right_subframe, text="Domain Prediction", font=timeline_font, bg=timeline_bg, fg=timeline_fg, anchor=timeline_anchor, justify=timeline_justify)
    # lbl_domain_prediction.grid(row=2, column=0, padx=timeline_padx, pady=timeline_pady, sticky=timeline_sticky)
    # if domain_prediction_status.get() == True:
    #     lbl_domain_prediction_status = tk.Label(right_subframe, text="✅", font=timeline_tick_font, bg=timeline_bg, fg=timeline_fg)
    # else:
    #     lbl_domain_prediction_status = tk.Label(right_subframe, text=" ", font=timeline_tick_font, bg=timeline_bg, fg=timeline_fg)
    # lbl_domain_prediction_status.grid(row=2, column=1, padx=timeline_padx, pady=timeline_pady, sticky=timeline_sticky)
    #
    # # Create a label for age and gender estimation with ✅
    # lbl_age_and_gender_estimation = tk.Label(right_subframe, text="Age and Gender Estimation", font=timeline_font, bg=timeline_bg, fg=timeline_fg, anchor=timeline_anchor, justify=timeline_justify)
    # lbl_age_and_gender_estimation.grid(row=3, column=0, padx=timeline_padx, pady=timeline_pady, sticky=timeline_sticky)
    # if age_and_gender_estimation_status.get() == True:
    #     lbl_age_and_gender_estimation_status = tk.Label(right_subframe, text="✅", font=timeline_tick_font, bg=timeline_bg, fg=timeline_fg)
    # else:
    #     lbl_age_and_gender_estimation_status = tk.Label(right_subframe, text=" ", font=timeline_tick_font, bg=timeline_bg, fg=timeline_fg)
    # lbl_age_and_gender_estimation_status.grid(row=3, column=1, padx=timeline_padx, pady=timeline_pady, sticky=timeline_sticky)
    #
    # # Create a label for hate prediction with ✅
    # lbl_hate_prediction = tk.Label(right_subframe, text="Hate Prediction", font=timeline_font, bg=timeline_bg, fg=timeline_fg, anchor=timeline_anchor, justify=timeline_justify)
    # lbl_hate_prediction.grid(row=4, column=0, padx=timeline_padx, pady=timeline_pady, sticky=timeline_sticky)
    # if hate_prediction_status.get() == True:
    #     lbl_hate_prediction_status = tk.Label(right_subframe, text="✅", font=timeline_tick_font, bg=timeline_bg, fg=timeline_fg)
    # else:
    #     lbl_hate_prediction_status = tk.Label(right_subframe, text=" ", font=timeline_tick_font, bg=timeline_bg, fg=timeline_fg)
    # lbl_hate_prediction_status.grid(row=4, column=1, padx=timeline_padx, pady=timeline_pady, sticky=timeline_sticky)
    #
    # # Create a label for sentiment calculation with ✅
    # lbl_sentiment_calculation = tk.Label(right_subframe, text="Sentiment Calculation", font=timeline_font, bg=timeline_bg, fg=timeline_fg, anchor=timeline_anchor, justify=timeline_justify)
    # lbl_sentiment_calculation.grid(row=5, column=0, padx=timeline_padx, pady=timeline_pady, sticky=timeline_sticky)
    # if sentiment_calculation_status.get() == True:
    #     lbl_sentiment_calculation_status = tk.Label(right_subframe, text="✅", font=timeline_tick_font, bg=timeline_bg, fg=timeline_fg)
    # else:
    #     lbl_sentiment_calculation_status = tk.Label(right_subframe, text=" ", font=timeline_tick_font, bg=timeline_bg, fg=timeline_fg)
    # lbl_sentiment_calculation_status.grid(row=5, column=1, padx=timeline_padx, pady=timeline_pady, sticky=timeline_sticky)
    #
    # # Create a label for video prediction with ✅
    # lbl_video_prediction = tk.Label(right_subframe, text="Video Prediction", font=timeline_font, bg=timeline_bg, fg=timeline_fg, anchor=timeline_anchor, justify=timeline_justify)
    # lbl_video_prediction.grid(row=6, column=0, padx=timeline_padx, pady=timeline_pady, sticky=timeline_sticky)
    # if video_prediction_status.get() == True:
    #     lbl_video_prediction_status = tk.Label(right_subframe, text="✅", font=timeline_tick_font, bg=timeline_bg, fg=timeline_fg)
    # else:
    #     lbl_video_prediction_status = tk.Label(right_subframe, text=" ", font=timeline_tick_font, bg=timeline_bg, fg=timeline_fg)
    # lbl_video_prediction_status.grid(row=6, column=1, padx=timeline_padx, pady=timeline_pady, sticky=timeline_sticky)
    #
    # # Create a label for merging with ✅
    # lbl_merging = tk.Label(right_subframe, text="Merging", font=timeline_font, bg=timeline_bg, fg=timeline_fg, anchor=timeline_anchor, justify=timeline_justify)
    # lbl_merging.grid(row=7, column=0, padx=timeline_padx, pady=timeline_pady, sticky=timeline_sticky)
    # if merging_status.get() == True:
    #     lbl_merging_status = tk.Label(right_subframe, text="✅", font=timeline_tick_font, bg=timeline_bg, fg=timeline_fg)
    # else:
    #     lbl_merging_status = tk.Label(right_subframe, text=" ", font=timeline_tick_font, bg=timeline_bg, fg=timeline_fg)
    # lbl_merging_status.grid(row=7, column=1, padx=timeline_padx, pady=timeline_pady, sticky=timeline_sticky)
    #
    # # Create a label for audio severity with ✅
    # lbl_audio_severity = tk.Label(right_subframe, text="Audio Severity", font=timeline_font, bg=timeline_bg, fg=timeline_fg, anchor=timeline_anchor, justify=timeline_justify)
    # lbl_audio_severity.grid(row=8, column=0, padx=timeline_padx, pady=timeline_pady, sticky=timeline_sticky)
    # if audio_severity_status.get() == True:
    #     lbl_audio_severity_status = tk.Label(right_subframe, text="✅", font=timeline_tick_font, bg=timeline_bg, fg=timeline_fg)
    # else:
    #     lbl_audio_severity_status = tk.Label(right_subframe, text=" ", font=timeline_tick_font, bg=timeline_bg, fg=timeline_fg)
    # lbl_audio_severity_status.grid(row=8, column=1, padx=timeline_padx, pady=timeline_pady, sticky=timeline_sticky)
    #
    # # Create a label for video severity with ✅
    # lbl_video_severity = tk.Label(right_subframe, text="Video Severity", font=timeline_font, bg=timeline_bg, fg=timeline_fg, anchor=timeline_anchor, justify=timeline_justify)
    # lbl_video_severity.grid(row=9, column=0, padx=timeline_padx, pady=timeline_pady, sticky=timeline_sticky)
    # if video_severity_status.get() == True:
    #     lbl_video_severity_status = tk.Label(right_subframe, text="✅", font=timeline_tick_font, bg=timeline_bg, fg=timeline_fg)
    # else:
    #     lbl_video_severity_status = tk.Label(right_subframe, text=" ", font=timeline_tick_font, bg=timeline_bg, fg=timeline_fg)
    # lbl_video_severity_status.grid(row=9, column=1, padx=timeline_padx, pady=timeline_pady, sticky=timeline_sticky)
    #
    # # Create a label for merged severity with ✅
    # lbl_merged_severity = tk.Label(right_subframe, text="Merged Severity", font=timeline_font, bg=timeline_bg, fg=timeline_fg, anchor=timeline_anchor, justify=timeline_justify)
    # lbl_merged_severity.grid(row=10, column=0, padx=timeline_padx, pady=timeline_pady, sticky=timeline_sticky)
    # if merged_severity_status.get() == True:
    #     lbl_merged_severity_status = tk.Label(right_subframe, text="✅", font=timeline_tick_font, bg=timeline_bg, fg=timeline_fg)
    # else:
    #     lbl_merged_severity_status = tk.Label(right_subframe, text=" ", font=timeline_tick_font, bg=timeline_bg, fg=timeline_fg)
    # lbl_merged_severity_status.grid(row=10, column=1, padx=timeline_padx, pady=timeline_pady, sticky=timeline_sticky)
    #

    # Define a list of steps or tasks
    steps = [
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

    # # Create labels dynamically for each step
    # for i, (step_text, step_status) in enumerate(steps):
    #     lbl_step = tk.Label(right_subframe, text=step_text, font=timeline_font, bg=timeline_bg, fg=timeline_fg, anchor=timeline_anchor, justify=timeline_justify)
    #     lbl_step.grid(row=i+2, column=0, padx=timeline_padx, pady=timeline_pady, sticky=timeline_sticky)
    #
    #     lbl_step_status = tk.Label(right_subframe, text=" ", font=timeline_tick_font, bg=timeline_bg, fg=timeline_fg)
    #     lbl_step_status.grid(row=i+2, column=1, padx=timeline_padx, pady=timeline_pady, sticky=timeline_sticky)
    #
    #     toggle_status(step_status, lbl_step_status)  # Toggle the status and update the label
    #
    #     # Example: Button to toggle the status and update the label
    #     btn_toggle = tk.Button(right_subframe, text="Toggle", command=lambda sv=step_status, lbl=lbl_step_status: toggle_status(sv, lbl))
    #     btn_toggle.grid(row=i+2, column=2, padx=timeline_padx, pady=timeline_pady, sticky=timeline_sticky)

    # Define the tick_tick function
    # def tick_tick():
    #     counter = 1
    #     for i, (step_text, step_status) in enumerate(steps):
    #         step_status.set(True)  # Set the status to True
    #         lbl_step_status = lbl_step_statuses[i]  # Get the corresponding label
    #
    #         tick = "✅"
    #         lbl_step_status.config(text=tick)  # Update the label with tick marks
    #
    #         counter += 1
    #         print(counter)
    #         # print time now in seconds
    #         print(datetime.now().strftime('%S'))
    #
    #         sleep(1)  # Sleep for 1 second

    # Create labels dynamically for each step
    lbl_step_statuses = []  # List to store the label references
    for i, (step_text, step_status) in enumerate(steps):
        lbl_step = tk.Label(right_subframe, text=step_text, font=timeline_font, bg=timeline_bg, fg=timeline_fg,
                            anchor=timeline_anchor, justify=timeline_justify)
        lbl_step.grid(row=i + 2, column=0, padx=timeline_padx, pady=timeline_pady, sticky=timeline_sticky)

        lbl_step_status = tk.Label(right_subframe, text=" ", font=timeline_tick_font, bg=timeline_bg, fg=timeline_fg)
        lbl_step_status.grid(row=i + 2, column=1, padx=timeline_padx, pady=timeline_pady, sticky=timeline_sticky)

        lbl_step_statuses.append(lbl_step_status)  # Add the label reference to the list

    # # Create a button to call the tick_tick function
    # btn_tick = tk.Button(right_subframe, text="Tick", command=tick_tick)
    # btn_tick.grid(row=len(steps)+2, column=0, columnspan=2, padx=timeline_padx, pady=timeline_pady, sticky=timeline_sticky)

    # Example: Button to toggle a specific step
    btn_toggle = tk.Button(right_subframe, text="Toggle Step 2",
                           command=lambda: toggle_single_item(1))  # Specify the index of the step to toggle
    btn_toggle.grid(row=2, column=2, padx=timeline_padx, pady=timeline_pady, sticky=timeline_sticky)

    content.pack()

    # # Function to toggle the status between True and False
    # def toggle_status():
    #     transcribing_n_processing_status.set(not transcribing_n_processing_status.get())
    #     update_button_text()
    #
    # # Function to update the button text based on the status
    # def update_button_text():
    #     if transcribing_n_processing_status.get() == True:
    #         lbl_transcribing_n_processing_status.configure(text="✅")
    #     else:
    #         lbl_transcribing_n_processing_status.configure(text="")
    #
    # # Create a button to toggle the status
    # btn_toggle = tk.Button(right_subframe, text="Toggle", command=toggle_status, font=("Arial Bold", 15), bg='white')
    # btn_toggle.grid(row=2, column=0, padx=80, pady=0)
    #
    # # Create a button to update the button text
    # btn_update = tk.Button(right_subframe, text="Update", command=update_button_text, font=("Arial Bold", 15), bg='white')
    # btn_update.grid(row=2, column=1, padx=10, pady=0, sticky='w')

    #
    #
    #
    # # on right side of the image show text "Image"
    # lbl = tk.Label(middle, text="Input", font=("Arial Bold", 20), bg='white')
    # lbl.place(x=80, y=40)
    #
    # # # show the image in the gui and resize it to 256x256
    # # placeHolderImageOpen = Image.open(placeHolderImagePath)
    # # placeHolderImageOpen = placeHolderImageOpen.resize((256, 256), Image.ANTIALIAS)
    # # placeHolderImage = ImageTk.PhotoImage(placeHolderImageOpen)
    # # label = Label(middle, image=placeHolderImage, bg='red', width=256, height=256)
    # # label.place(x=80, y=100)
    #
    # # Button below the image to pick image and command= pickImage
    # pick_image_btn = tk.Button(middle, text="Pick Image", font=('arial', 15), width=10, height=1, bg='green',
    #                            fg='white',
    #                            command=pickImage)
    # pick_image_btn.place(x=80, y=420)
    #
    # Radiobutton(middle, text=list(model_set.keys())[0], variable=model_choice,
    #             value=model_set[list(model_set.keys())[0]], bg="white").place(x=90, y=380)
    # Radiobutton(middle, text=list(model_set.keys())[1], variable=model_choice,
    #             value=model_set[list(model_set.keys())[1]], bg="white").place(x=170, y=380)
    # Radiobutton(middle, text=list(model_set.keys())[2], variable=model_choice,
    #             value=model_set[list(model_set.keys())[2]], bg="white").place(x=250, y=380)
    #
    # # button to Predict
    # get_prediction_btn = tk.Button(middle, text="Predict", font=('arial', 15), width=10, height=1, bg='green',
    #                                fg='white',
    #                                command=showImage)
    # get_prediction_btn.place(x=220, y=420)
    #
    # """Intermediate Section"""
    # # on right side of the image show text "Image"
    # lbl = tk.Label(middle, text="Intermediate", font=("Arial Bold", 20), bg='white')
    # lbl.place(x=400, y=40)
    #
    # """Result Section"""
    # # on right side of the image show text "Image"
    # lbl = tk.Label(middle, text="Result", font=("Arial Bold", 20), bg='white')
    # lbl.place(x=870, y=40)
    #
    # # preidction score
    # prediction_score = Label(middle, text="Prediction: ", font=("Arial", 10), bg='white')
    # prediction_score.place(x=870, y=100)
    #
    # # preidction score
    # prediction_score = Label(middle, text="Prediction Score: ", font=("Arial", 10), bg='white')
    # prediction_score.place(x=870, y=150)
    #
    # # used model
    # used_model = Label(middle, text="Used Model: ", font=("Arial", 10), bg='white')
    # used_model.place(x=870, y=200)

    root.mainloop()

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
