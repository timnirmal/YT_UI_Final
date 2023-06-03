import os
import subprocess

from pydub import AudioSegment


def convert_video_to_audio_ffmpeg(video_file, run_folder, file_name, output_ext="wav"):
    """Converts video to audio directly using `ffmpeg` command
    with the help of subprocess module"""

    print("Converting video to audio")
    # save file in run_folder
    subprocess.call(["ffmpeg", "-y", "-i", video_file, f"{run_folder}{file_name}.{output_ext}"],
                    stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

    # check if file exists
    if not os.path.exists(f"{run_folder}{file_name}.{output_ext}"):
        print(f"Error converting {video_file} to {run_folder}{file_name}.{output_ext}")
        return None

    print(f"Converted {video_file} to {run_folder}{file_name}.{output_ext}")

    return f"{run_folder}{file_name}.{output_ext}"


def wav_to_mono_flac(audio_file):
    """Converts audio to mono channel and flac format"""

    print("Converting to mono channel and flac format")

    mono_trac = AudioSegment.from_wav(audio_file)
    # save as audio_file excluded the extension
    filename, ext = os.path.splitext(audio_file)
    mono_trac = mono_trac.set_channels(1)
    mono_trac.export(f"{filename}.flac", format="flac")

    print(f"Converted {audio_file} to {filename}.flac")

    return f"{filename}.flac"
