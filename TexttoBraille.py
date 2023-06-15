import os
import tkinter as tk
from tkinter import *
from tkinter import filedialog

import cv2
import numpy as np
from PIL import Image, ImageTk
from keras.utils import img_to_array

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
    file_name.set(file)

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


# def clean

def showImage():
    file = file_name.get()

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

        image_read = cv2.imread(file)
        image_read = cv2.resize(image_read, (256, 256))
        image_read = img_to_array(image_read)
        image_read = np.expand_dims(image_read, axis=0)
        image_read = image_read / 255.0

    predict_result, result_text = predict.predict(image_read, model=model_choice.get())

    # get list of images in result folder
    result_folder = os.path.join(currentPath, "result")
    result_images = os.listdir(result_folder)

    # show the result_images in the gui one by one with a 1 second delay
    for i in range(len(result_images)):
        # show the image in the gui
        image1 = Image.open(os.path.join(result_folder, result_images[i]))
        # resize keeping aspect ratio
        image_sizes = image1.size
        SIZE = 400
        if image_sizes[0] > SIZE:
            image1 = image1.resize((SIZE, int(image_sizes[1] * SIZE / image_sizes[0])), Image.ANTIALIAS)
        img = ImageTk.PhotoImage(image1)
        label = Label(middle, image=img, bg='red')
        label.image = img
        label.place(x=400, y=100)

        # sleep for 500 millisecond
        tksleep(500)

    result_text_var.set(result_text)
    predicted_var.set(predict_result)

    # show the result in the gui after prediction
    result = Label(middle, text=result_text_var.get(), font=("Arial", 10), bg='white')
    result.place(x=1000, y=100)

    # text for prediction
    predict_val = predicted_var.get()
    # remove [ and ] from the string by replacing them with empty string
    predict_val = predict_val.replace("[", "")
    predict_val = predict_val.replace("]", "")
    predicted = Label(middle, text=predict_val, font=("Arial", 10), bg='white')
    predicted.place(x=1000, y=150)

    # text for prediction
    # find model name by value
    model = Label(middle, text=list(model_set.keys())[list(model_set.values()).index(model_choice.get())],
                  font=("Arial", 10), bg='white')
    model.place(x=1000, y=200)


def tksleep(t):
    ms = int(t)
    root = tk._get_default_root()
    var = tk.IntVar(root)
    root.after(ms, lambda: var.set(1))
    root.wait_variable(var)


if __name__ == '__main__':
    # Create Object and setup root
    root = Tk()
    root.title("Plant Disease Detection")

    # Create Frames
    top = Frame(root, width=1200, height=100, bg='white')
    top.pack(side=TOP)
    middle = Frame(root, width=1200, height=520, bg='white')
    middle.pack(side=TOP)

    # string variable with default value "model_1"
    model_choice = StringVar(value=default_model)
    # file name variable
    file_name = StringVar()
    result_text_var = StringVar()
    predicted_var = StringVar()

    # Create Widgets

    """Top Section"""
    # create a label as title in center
    lbl = tk.Label(top, text="Plant Disease Detection", font=("Arial Bold", 36), fg='black')
    lbl.grid()

    """Image Section"""
    # on right side of the image show text "Image"
    lbl = tk.Label(middle, text="Input", font=("Arial Bold", 20), bg='white')
    lbl.place(x=80, y=40)

    # show the image in the gui and resize it to 256x256
    placeHolderImageOpen = Image.open(placeHolderImagePath)
    placeHolderImageOpen = placeHolderImageOpen.resize((256, 256), Image.ANTIALIAS)
    placeHolderImage = ImageTk.PhotoImage(placeHolderImageOpen)
    label = Label(middle, image=placeHolderImage, bg='red', width=256, height=256)
    label.place(x=80, y=100)

    # Button below the image to pick image and command= pickImage
    pick_image_btn = tk.Button(middle, text="Pick Image", font=('arial', 15), width=10, height=1, bg='green',
                               fg='white',
                               command=pickImage)
    pick_image_btn.place(x=80, y=420)

    Radiobutton(middle, text=list(model_set.keys())[0], variable=model_choice,
                value=model_set[list(model_set.keys())[0]], bg="white").place(x=90, y=380)
    Radiobutton(middle, text=list(model_set.keys())[1], variable=model_choice,
                value=model_set[list(model_set.keys())[1]], bg="white").place(x=170, y=380)
    Radiobutton(middle, text=list(model_set.keys())[2], variable=model_choice,
                value=model_set[list(model_set.keys())[2]], bg="white").place(x=250, y=380)

    # button to Predict
    get_prediction_btn = tk.Button(middle, text="Predict", font=('arial', 15), width=10, height=1, bg='green',
                                   fg='white',
                                   command=showImage)
    get_prediction_btn.place(x=220, y=420)

    """Intermediate Section"""
    # on right side of the image show text "Image"
    lbl = tk.Label(middle, text="Intermediate", font=("Arial Bold", 20), bg='white')
    lbl.place(x=400, y=40)

    """Result Section"""
    # on right side of the image show text "Image"
    lbl = tk.Label(middle, text="Result", font=("Arial Bold", 20), bg='white')
    lbl.place(x=870, y=40)

    # preidction score
    prediction_score = Label(middle, text="Prediction: ", font=("Arial", 10), bg='white')
    prediction_score.place(x=870, y=100)

    # preidction score
    prediction_score = Label(middle, text="Prediction Score: ", font=("Arial", 10), bg='white')
    prediction_score.place(x=870, y=150)

    # used model
    used_model = Label(middle, text="Used Model: ", font=("Arial", 10), bg='white')
    used_model.place(x=870, y=200)

    root.mainloop()
