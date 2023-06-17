from pytube import YouTube

# def Download(link):
#     youtubeObject = YouTube(link, use_oauth=True, allow_oauth_cache=True)
#     youtubeObject = youtubeObject.streams.get_highest_resolution()
#     try:
#         youtubeObject.download()
#     except:
#         print("An error has occurred")
#     print("Download is completed successfully")
#
#
# link = "https://www.youtube.com/watch?v=n2srSTrV1UQ"
# Download(link)

import os
import subprocess

video_file = r"D:\Projects\UI\runs\run_6\People X Nainowale Ne  Chillout Mashup  @YashrajMukhateOfficial   MEHER.mp4"

af = video_file[:-4]
print(af)

# convert mp4 to wav
# os.system('ffmpeg -i {} -acodec pcm_s16le -ar 16000 {}.wav'.format(video_file, af))
# os.system('ffmpeg -i {} -acodec pcm_s16le -ar 16000 {}/{}.wav'.format(video_file, af, af))

subprocess.call(
    [
        "ffmpeg",
        "-i",
        video_file,
        "-ab",
        "160k",
        "-ac",
        "2",
        "-ar",
        "44100",
        "-vn",
        f"{af}.wav",
    ]
)

from os import path
# from pydub import AudioSegment
#
# # files
# src = video_file
# dst = af + ".wav"
#
# # convert mp4 to wav
# sound = AudioSegment.from_file(src)
# sound.export(dst, format="wav")