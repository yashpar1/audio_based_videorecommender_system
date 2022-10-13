import pandas as pd
from sklearn.metrics.pairwise import linear_kernel
import numpy as np

df = pd.read_csv('data/audioset_dataframe.csv', converters={'audio': pd.eval})

def recommendations(dataframe, video_id):
    df_rec = dataframe
    # puts audio embeddings into numpy arrays to allow cosine similarity calucation
    aud_embeds = pd.DataFrame(df['audio'].tolist())
    aud_embeds = aud_embeds.fillna(0)

    # because the data has already been normalized, cosine similarity can be calculated by simply using a dot product
    cos_sim = pd.DataFrame(linear_kernel(aud_embeds, aud_embeds))

    # matches the indices of the cos_sim array to the video ids
    indices = pd.Series(aud_embeds.index, index=df_rec['video_id']).drop_duplicates()

    # finds indices of video id
    idx = indices[video_id]
    # finds the highest cosine similarities
    similarity_scores = list(enumerate(cos_sim[idx]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    top_five = similarity_scores[1:6]
    video_indices = [i[0] for i in top_five]
    top_five_ids = df_rec['video_id'].iloc[video_indices]
    top_five_ids = 'youtube.com/watch?v=' + top_five_ids.astype(str)
    print(top_five_ids)

recommendations(df, 'cQiEI7HLGJg')