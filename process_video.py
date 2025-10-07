import os
import subprocess

video_list=os.listdir("video assets")

for video in video_list:
    tutorial_no= video.split("_")[1].split(".mp4")[0]
    video_title=video.split("_")[0].strip("Tutorial")
    print(tutorial_no, video_title)
    subprocess.run(["ffmpeg","-i",f"video assets/{video}",f"audio assets/{tutorial_no}_{video_title}.mp3"])