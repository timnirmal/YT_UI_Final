from pathlib import Path
import cv2
import dlib
import sys
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from lib.wide_resnet import WideResNet
from keras.utils.data_utils import get_file
from keras.models import load_model
from tensorflow.keras.utils import img_to_array
import json

from concurrent import futures
from contextlib import contextmanager
from threading import Lock
import time

classifier = load_model("models/video/emotion_little_vgg_2.h5")
pretrained_model = "https://github.com/yu4u/age-gender-estimation/releases/download/v0.5/weights.28-3.73.hdf5"

modhash = 'fbe63257a054c1c5466cfd7bf14646d6'
emotion_classes = {0: 'Angry', 1: 'Fear', 2: 'Happy', 3: 'Neutral', 4: 'Sad', 5: 'Surprise'}

# Define our model parameters
depth = 16
k = 8
weight_file = None
margin = 0.4
image_dir = None

# Get our weight file
if not weight_file:
    weight_file = get_file("weights.28-3.73.hdf5", pretrained_model, cache_subdir="models/video/",
                           file_hash=modhash, cache_dir=Path(sys.argv[0]).resolve().parent)

# Load model and weights
img_size = 64
model = WideResNet(img_size, depth=depth, k=k)()
model.load_weights(weight_file)

detector = dlib.get_frontal_face_detector()

frame_no = 0  # Initialize frame_no
frames_dict = {}
toJSON = {}
current_time_min = 0
f = open("frame_data.json", 'w+')
lock = Lock()


def draw_label(image, point, label, font=cv2.FONT_HERSHEY_SIMPLEX,
               font_scale=0.8, thickness=1):
    size = cv2.getTextSize(label, font, font_scale, thickness)[0]
    x, y = point
    cv2.rectangle(image, (x, y - size[1]), (x + size[0], y), (255, 0, 0), cv2.FILLED)
    cv2.putText(image, label, point, font, font_scale, (255, 255, 255), thickness, lineType=cv2.LINE_AA)


@contextmanager
def video_capture(*args, **kwargs):
    cap = cv2.VideoCapture(*args, **kwargs)
    try:
        yield cap
    finally:
        cap.release()


def process_frame(frame, index, cap):
    current_time_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
    current_time_sec = current_time_msec / 1000
    current_time_min = current_time_sec

    value = []
    preprocessed_faces_emo = []

    input_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_h, img_w, _ = np.shape(input_img)
    detected = detector(frame, 1)
    faces = np.empty((len(detected), img_size, img_size, 3))

    preprocessed_faces_emo = []
    if len(detected) > 0:
        for i, d in enumerate(detected):
            x1, y1, x2, y2, w, h = d.left(), d.top(), d.right() + 1, d.bottom() + 1, d.width(), d.height()
            xw1 = max(int(x1 - margin * w), 0)
            yw1 = max(int(y1 - margin * h), 0)
            xw2 = min(int(x2 + margin * w), img_w - 1)
            yw2 = min(int(y2 + margin * h), img_h - 1)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            faces[i, :, :, :] = cv2.resize(frame[yw1:yw2 + 1, xw1:xw2 + 1, :], (img_size, img_size))
            face = frame[yw1:yw2 + 1, xw1:xw2 + 1, :]
            face_gray_emo = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            face_gray_emo = cv2.resize(face_gray_emo, (48, 48), interpolation=cv2.INTER_AREA)
            face_gray_emo = face_gray_emo.astype("float") / 255.0
            face_gray_emo = img_to_array(face_gray_emo)
            face_gray_emo = np.expand_dims(face_gray_emo, axis=0)
            preprocessed_faces_emo.append(face_gray_emo)

        # Make a prediction for Age and Gender
        results = model.predict(np.array(faces))
        predicted_genders = results[0]
        ages = np.arange(0, 101).reshape(101, 1)
        predicted_ages = results[1].dot(ages).flatten()

        # Make a prediction for Emotion
        emo_labels = []
        for i, d in enumerate(detected):
            preds = classifier.predict(preprocessed_faces_emo[i])[0]
            emo_labels.append(emotion_classes[preds.argmax()])

        # Draw results
        for i, d in enumerate(detected):
            label = "{}, {}, {}".format(int(predicted_ages[i]),
                                        "F" if predicted_genders[i][0] > 0.4 else "M", emo_labels[i])
            draw_label(frame, (d.left(), d.top()), label)
            value.append(label.split(', '))
            cur_frame = [index]
            cur_frame.extend(label.split(', '))
            # add start and end time
            cur_frame.append(current_time_min - 1)
            cur_frame.append(current_time_min)

            with lock:
                frames_dict[index] = cur_frame

    with lock:
        key = f"Frame:{index}"
        toJSON[key] = value


def process_video_frames(cap):
    global frame_no  # Declare frame_no as global
    global current_time_min
    start_time = time.time()
    executor = futures.ThreadPoolExecutor()  # Create a ThreadPoolExecutor

    print(frame_no, "/", cap.get(cv2.CAP_PROP_FRAME_COUNT), end='\t')
    while True:
        ret, frame = cap.read()

        if not ret:
            print("End of Video")
            break

        executor.submit(process_frame, frame, frame_no, cap)  # Submit frame processing to executor

        if cv2.waitKey(1) == 13:  # 13 is the Enter Key
            break

        frame_no += 1

    executor.shutdown()  # Wait for all tasks to complete
    json_str = json.dumps(toJSON, indent=4) + '\n'
    f.write(json_str)
    f.close()
    fps_count = cap.get(cv2.CAP_PROP_FPS)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Elapsed Time:", elapsed_time)

    df = pd.DataFrame.from_dict(frames_dict, orient='index', columns=['frame', 'age', 'gen', 'emotion', 'start_time', 'end_time'])
    sorted_df = df.sort_values('frame', ascending=True)

    sorted_df.to_csv('sorted_data.csv', index=False)

    return df


def predict(file_name, run_path):
    with video_capture(run_path + file_name) as cap:
        df = process_video_frames(cap)

    print("End Of Program")

    print(df.head(10))

    # df['age'] to int
    df['age'] = df['age'].astype(int)
    df['age_classes'] = df['age'].apply(lambda x: 0 if x < 20 else (1 if x < 30 else (2 if x < 40 else 3)))

    # frame,start_time,end_time,age,gen,emotion,age_classes
    df = df[["frame","start_time","end_time","age","gen","emotion","age_classes"]]

    # save to csv
    df.to_csv(run_path + "filtered_frames.csv", index=False)

    if not df.empty:
        df['age'] = df['age'].astype(int)
        df.plot(y='age', figsize=(25, 10), title='Age vs Frames', ylabel='age')
        # save to png
        plt.savefig(run_path + "Age vs Frames.png")

        dict_emo = df.set_index('frame').to_dict()['emotion']
        x = np.array(list(zip(*dict_emo.items())))
        u, ind = np.unique(x[1, :], return_inverse=True)
        x[1, :] = ind
        x = x.astype(int).T

        plt.figure(figsize=(20, 5))
        # plot the two columns of the array
        plt.plot(x[:, 0], x[:, 1])
        # set the labels accordinly
        plt.gca().set_yticks(range(len(u)))
        plt.title("Variation of Emotions in Frames")
        plt.xlabel("frame")
        plt.ylabel("emotion")
        # plt.gca().set_yticklabels(['Angry', 'Fear', 'Happy', 'Neutral', 'Sad'])
        plt.tick_params(labelsize=10)
        # plt.show()
        plt.savefig(run_path + "Variation of Emotions in Frames.png")
    else:
        print("No faces detected in the video")

    return frames_dict
