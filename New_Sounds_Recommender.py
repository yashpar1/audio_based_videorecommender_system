import numpy as np
import pandas as pd
import tensorflow as tf
import os
from sklearn.metrics.pairwise import linear_kernel

def norm(x):
    return list(x/np.linalg.norm(x))

def AudioRecommender(path_to_new_sounds):
    """A function that takes a path to the audioset_dataframe.csv and where new sounds recommendations are wanted for
    and returns video recommendations.
    """

    df_trained = pd.read_csv('data/audioset_dataframe.csv', converters={'audio': pd.eval})

    # formats new audio input correctly
    data_init = []
    for file in os.listdir(path_to_new_sounds):
        # filters out the audio files themselves, bringing in only the embeddings. Could be avoided with the use of folders.
        if file.endswith('.tfrecord'):
            data_init.append(f'{path_to_new_sounds}{file}')

    data_tf = tf.data.TFRecordDataset(data_init)
    features = []
    cutoff = 10
    bits = 128
    # for providing generic, easy-to-find names for the new videos
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

    # puts the new audio into the same dataframe as the ones downloaded from AudioSet
    df_new = pd.DataFrame.from_records(features)
    df_new['audio'] = df_new['audio'].apply(norm)
    df_total = pd.concat([df_trained, df_new]).reset_index()

    aud_embeds = pd.DataFrame(df_total['audio'].tolist())
    aud_embeds = aud_embeds.fillna(0)

    # because the new audio is at the end, we can iterate backwards to analyze it all
    i = -1
    for new_audio in df_new:

        cos_sim = pd.DataFrame(linear_kernel(np.array(aud_embeds), np.array(aud_embeds)[i].reshape(1, -1)), columns=['cos_sim'])

        # creates a temporary dataframe that contains both similarity scores and video links
        temp = df_total.join(cos_sim)
        # removes the new audio from the dataset, as we don't want to find similar audio files but instead videos
        temp = temp[~temp['video_id'].str.contains('new_video')]
        # finds the five most similar videos
        temp = temp.sort_values(by=['cos_sim'], ascending=False)
        top_five = temp['video_id'].iloc[1:6]
        top_five = 'youtube.com/watch?v=' + top_five.astype(str)
        print(data_init[i])
        print(top_five)
        i -= 1

AudioRecommender('data/new_sounds/')