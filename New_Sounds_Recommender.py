import numpy as np
import pandas as pd
import tensorflow as tf
import os
from sklearn.metrics.pairwise import linear_kernel

def norm(x):
    return list(x/np.linalg.norm(x))

def AudioRecommender(path_to_new_sounds):
    """A function that takes a path to the audioset_dataframe.csv and where new sounds recommendations are wanted for
    and returns video recommendations. Comments are minimal as the code is largely reused from other files.
    """
    df_trained = pd.read_csv('data/audioset_dataframe.csv', converters={'audio': pd.eval})

    # formats new audio input correctly
    data_init = []
    for file in os.listdir(path_to_new_sounds):
        if file.endswith('.tfrecord'): # filters out the audio files themselves, bringing in only the embeddings
            data_init.append(f'{path_to_new_sounds}{file}')

    data_tf = tf.data.TFRecordDataset(data_init)
    features = []
    cutoff = 10
    bits = 128
    new_vid_id = 1

    for tf_file in data_tf:
        base = tf.train.SequenceExample()
        base.ParseFromString(tf_file.numpy())
        audio = base.feature_lists.feature_list['audio_embedding'].feature
        audio_by_second = sum([list(second.bytes_list.value[0]) for second in audio],[])[:cutoff*bits]
        if len(audio_by_second) < cutoff*bits:
            continue
        feats = {
            'video_id': 'new_video_' + str(new_vid_id),
            'audio': audio_by_second
        }

        features.append(feats)
        new_vid_id += 1

    df_new = pd.DataFrame.from_records(features)
    df_new['audio'] = df_new['audio'].apply(norm)

    aud_embeds_trained = pd.DataFrame(df_trained['audio'].tolist())
    aud_embeds_trained = aud_embeds_trained.fillna(0)
    aud_embeds_trained = aud_embeds_trained.to_numpy()

    aud_embeds_new = pd.DataFrame(df_new['audio'].tolist())
    aud_embeds_new = aud_embeds_new.fillna(0)
    aud_embeds_new = aud_embeds_new.to_numpy()

    i = 0
    for new_audio in df_new:

        cos_sim = pd.DataFrame(linear_kernel(aud_embeds_new[i,:], aud_embeds_trained))

        indices = pd.Series(aud_embeds_new.index, index=df_new['video_id']).drop_duplicates()

        # finds indices of video id
        idx = indices[new_audio['video_id']]
        # finds the highest cosine similarities
        similarity_scores = list(enumerate(cos_sim[idx]))
        similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
        top_five = similarity_scores[1:6]
        video_indices = [i[0] for i in top_five]
        top_five_ids = df_trained['video_id'].iloc[video_indices]
        top_five_ids = 'youtube.com/watch?v=' + top_five_ids.astype(str)
        print(data_init[i])
        print(top_five_ids)
        i += 1

AudioRecommender('data/new_sounds/')