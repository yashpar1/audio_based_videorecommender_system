import pandas as pd
import numpy as np
import json
from ast import literal_eval

with open('data/audioset_data.json', 'r') as file:
    file_read = json.loads(file.read())
    data = literal_eval(file_read)

# a function to normalize the data now so that cosine similarity runs faster
def norm(x):
    return list(x/np.linalg.norm(x))

# converts id's from bytes to strings, and normalizes the audio embeddings
df = pd.DataFrame(data)
df['video_id'] = df['video_id'].str.decode('utf-8')
df['audio'] = df['audio'].apply(norm)

df.to_csv('data/audioset_dataframe.csv')