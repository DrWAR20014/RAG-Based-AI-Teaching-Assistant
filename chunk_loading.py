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
Task:
First of all don't write any code or explanation of solving the problem or how can it be solved, it doesn't matter to the user, just provide what was asked and consider the following also:-
Formatting rules:
1. Use the video number from the "number" object's value (convert to integer if needed).
2. Show seconds with two decimal places (e.g., 84.16s). Convert seconds to mm:ss for the parenthesis (round seconds to two decimals).
3. Compute duration = end - start and show with two decimals.
4. Sort output by video number (ascending) and then by start time (ascending).
5. By default show all matching segments. If there are more than 3 matching segments and the user did NOT explicitly ask "show all", return the top 3 segments by longest individual duration.
6. Do NOT print chunk IDs, the original JSON, explanations, or extra text—only the list lines.
7. If the user asks something unrelated to the course, reply exactly: "I can only answer questions related to the course."

Example output line (must match formatting):
- Video 9 — start 0.00s (0:00) — end 3.08s (0:03.08) — duration 3.08s
IMPORTANT: The "number" field is a mapping of chunk_id -> video_number (e.g. "5023":"9"). 
**Always use the VALUE from the "number" mapping as the video number. Do NOT use the chunk_id (the mapping KEY).**
"""
with open("prompt_text.txt", "w") as f:
    f.write(prompt)

response=inference(prompt)
print(response)

with open("response_text.txt", "w") as g:
    g.write(response['response'])


