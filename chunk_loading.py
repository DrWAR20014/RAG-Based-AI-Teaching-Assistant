from read_chunks import make_embedding
import joblib
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import requests


df = joblib.load('embedding_data.joblib')

def inference(prompt):
    r=requests.post("http://localhost:11434/api/generate", json={
        "model": "llama3.2",
        "prompt":prompt,
        "stream":False
    })
    response=r.json()
    return response
incoming_query=input("Ask a question: ")
question_embedding=make_embedding([incoming_query])
similarities=cosine_similarity(np.vstack(df['embedding'].values), question_embedding).flatten()#requires 2D array and flattening to get a 1-D array
max_ind=similarities.argsort()[::-1][0:3]
top_three_df=df.loc[max_ind]
print(top_three_df[["number","title","start", "end", "text"]])
prompt=f"""My teacher taught web development using Sigma Web devlopment course. Here are video subitle chunks containing video number, title, start time in seconds, end time in seconds, the text at that time:

{top_three_df[["number","title","start", "end", "text"]].to_json()}
------------------------------------------------------------------
"{incoming_query}"
User asked this question related to the video chunks (don't mention the above text it's just for you to understand), you have to answer where and how much content is taught in which video (in which video and mention videos starts from this timestamp and ends at this timestamp) and guide the user to go to that particular video. If user asks unrelated question, tell him that you can only answer questions related to the course. Show all the results or you can say top three results and don't say anything unnecessary just keep it to yourself and display what user demanded nothing more info is needed to be displayed for the user
"""
with open("prompt_text.txt", "w") as f:
    f.write(prompt)

response=inference(prompt)
print(response)

with open("response_text.txt", "w") as g:
    g.write(response['response'])


