import whisper
import json
import os

model=whisper.load_model("large-v2")
audio_list=os.listdir("audio assets/")


for audio in audio_list:
    # print(audio)
    if "_" in audio:
        audio_num=audio.split("_")[0]
        audio_title=audio.split("_")[1][:-5]
        result=model.transcribe(audio=f"audio assets/{audio}", language="hi", task="translate", word_timestamps=False)
        # result=model.transcribe(audio=f"audio assets/sample.mp3", language="hi", task="translate", word_timestamps=False)
        chunks=[]
        for segment in result["segments"]:
            chunks.append({"number":audio_num,"title":audio_title,"start":segment["start"],"end":segment["end"], "text":segment["text"]})
        # print(chunks)
        chunks_with_metadata={"chunks":chunks, "text":result["text"]}      
        with open(f"json2_files/{audio}.json", "w") as f:
            json.dump(chunks_with_metadata,f)