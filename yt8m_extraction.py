import numpy as np
import pandas as pd
import tensorflow as tf
import json
import os

# def AudioExtractor(data_path):

data_path = 'data/audioset/'
data_init = []
for file in os.listdir(data_path):
    if file.endswith('.tfrecord'): #This line is needed to skip trying to add the frame_train_download_plan to the dataset
        data_init.append(f'{data_path}{file}')

data_tf = tf.data.TFRecordDataset(data_init)
features = []

for tf_file in data_tf:
    base = tf.train.SequenceExample()
    base.ParseFromString(tf_file.numpy())
    vid_id = base.context.feature['id'].bytes_list.value[0]
    labels = base.context.feature['labels'].int64_list.value
#    video = base.feature_lists.feature_list['rgb'].feature
#    video_per_second = [list(second.bytes_list.value[0]) for second in video]
#    video_clip = [item for sublist in video_per_second[:10] for item in sublist]
    audio = base.feature_lists.feature_list['audio'].feature
    audio_per_second = sum([list(second.bytes_list.value[0]) for second in audio],[])[:cutoff*bits]
    if len(audio_by_second) < cutoff*bits:
        continue
    feats = {
        'video_id': vid_id,
#        'labels': labels,
        'audio': audio_clip
#        'video': video_per_second
    }
    features.append(feats)

with open('data/data.json', 'w') as file:
    json.dump(repr(features), file)