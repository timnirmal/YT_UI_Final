# from pytube import YouTube
#
#
# # def Download(link):
# #     # youtubeObject = YouTube(link, use_oauth=True, allow_oauth_cache=True)
# #     youtubeObject = YouTube(link, use_oauth=True, allow_oauth_cache=True)
# #     youtubeObject = youtubeObject.streams.get_highest_resolution()
# #     try:
# #         youtubeObject.download()
# #     except:
# #         print("An error has occurred")
# #     print("Download is completed successfully")
# #
# #
# # link = "https://www.youtube.com/watch?v=n2srSTrV1UQ"
# # Download(link)
# #
# # exit()
#
# import yt_dlp
#
# link = "https://www.youtube.com/watch?v=n2srSTrV1UQ"
#
# ydl_opts = {
#     'format': 'best',
#     'outtmpl': '%(title)s.%(ext)s',
# }
#
# file_name = yt_dlp.YoutubeDL(ydl_opts)\
#     .prepare_filename(yt_dlp.YoutubeDL(ydl_opts).extract_info(link, download=False))
# print(file_name)
#
# exit()
# with yt_dlp.YoutubeDL(ydl_opts) as ydl:
#     ydl.download([link])
#
#
# print("downloaded", link)
#
# exit()
#
# import os
# import subprocess
#
# video_file = r"D:\Projects\UI\runs\run_6\People X Nainowale Ne  Chillout Mashup  @YashrajMukhateOfficial   MEHER.mp4"
#
# af = video_file[:-4]
# print(af)
#
# # convert mp4 to wav
# # os.system('ffmpeg -i {} -acodec pcm_s16le -ar 16000 {}.wav'.format(video_file, af))
# # os.system('ffmpeg -i {} -acodec pcm_s16le -ar 16000 {}/{}.wav'.format(video_file, af, af))
#
# subprocess.call(
#     [
#         "ffmpeg",
#         "-i",
#         video_file,
#         "-ab",
#         "160k",
#         "-ac",
#         "2",
#         "-ar",
#         "44100",
#         "-vn",
#         f"{af}.wav",
#     ]
# )
#
# from os import path
# # from pydub import AudioSegment
# #
# # # files
# # src = video_file
# # dst = af + ".wav"
# #
# # # convert mp4 to wav
# # sound = AudioSegment.from_file(src)
# # sound.export(dst, format="wav")
#
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
#
# from video.model import predict
#
# # video_file = r"D:\Projects\UI\datasets\3\Sri lankan Ticklish person (Kunuharapa).mkv"
# video_file = r"Sri lankan Ticklish person (Kunuharapa).mp4"
# run_path = r"D:\Projects\UI\runs\run_7"
#
# video_df = predict(video_file, run_path)
#
# df = pd.read_csv(r"filtered_frames.csv")
#
# df['age'] = df['age'].astype(int)
# df.plot(y='age', figsize=(25,10), title='Age vs Frames', ylabel='age')
# # save to png
# plt.savefig("Age vs Frames.png")
#
#
# dict_emo = df.set_index('frame').to_dict()['emotion']
# x = np.array(list(zip(*dict_emo.items())))
# u, ind = np.unique(x[1,:], return_inverse=True)
# x[1,:] = ind
# x = x.astype(int).T
#
# plt.figure(figsize=(20,5))
# # plot the two columns of the array
# plt.plot(x[:,0], x[:,1])
# #set the labels accordinly
# plt.gca().set_yticks(range(len(u)))
# plt.title("Variation of Emotions in Frames")
# plt.xlabel("frame")
# plt.ylabel("emotion")
# plt.gca().set_yticklabels(['Angry', 'Fear', 'Happy', 'Neutral', 'Sad'])
# plt.tick_params(labelsize=10)
# # plt.show()
# plt.savefig("Variation of Emotions in Frames.png")
import pandas as pd

from SubProcesses import merged_severity

run_path = "runs/run_6"
merged_df = pd.read_csv("runs/run_6/merged_df.csv")

# remove first row
merged_df = merged_df.iloc[1:]

merged_severity(merged_df,run_path)
