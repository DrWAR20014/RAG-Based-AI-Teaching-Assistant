# Steps to use RAG based AI teaching assistant

## Step 0: Collect video assets
Download the tutorial videos and organize them in a folder

## Step 1: Convert mp4 or any format videos into mp3 audios
Use the file process_video.py to convert videos into audios

## Step 2: Convert the audios into text
Use the python file speech_to_text.py to convert mp3 speech into text and they will be stored as json files containing all the metadata

## Step 3: Vectorizing the texts
Use the python file read_chunks.py to convert the texts into vectors/embeddings which will be stored in a .joblib file

## Step 4: Prompt generation, asking queries and feeding to LLM
Use the python file chunk_loading.py to read/load the joblib file which was made previously, create a relevant prompt and feed it to the LLM

# ⚠️ Disclaimer
This project uses whisper generated transcripts and embeddings derived from publicly available educational videos. All original content belongs to [Code With Harry]. No video or audio files are redistributed. This assistant is intended for educational enhancement and fair use.
