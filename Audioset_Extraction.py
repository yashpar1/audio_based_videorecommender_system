import numpy as np
import pandas as pd
import tensorflow as tf
import json
import os

def AudioExtractor(path_to_audioset_data, where_to_save_json):
    """A function that takes a path to where the AudioSet data is stored, and another to where the output
    .json file should go, and outputs said json file containing video ID's and the audio embeddings
    """
    data_path = path_to_audioset_data
    json_location = where_to_save_json

    # puts tensorflow records into a list
    data_init = []
    for file in os.listdir(data_path):
        data_init.append(f'{data_path}{file}')

    # the following line limits the number of tensorflow records parsed because of memory limits; it can be increased or removed
    # depending on the amount of memory available
    data_init = data_init[:1500]

    # converts the list of tensorflow records into a tensorflow dataset
    data_tf = tf.data.TFRecordDataset(data_init)

    # creates an empty list to store records in
    features = []

    # creates a 10-second cutoff for each clip because 10 seconds is the AudioSet default clip length; can be shortened
    # to save memory if necessary
    cutoff = 10
    bits = 128

    # parses through each tensorflow file and pulls info on each
    for tf_file in data_tf:
        # these next two lines convert the tensorflow file into a json-ish format: for more details, run this notebook
        # https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/load_data/tfrecord.ipynb
        # and jump to "TFRecord files in Python", and/or see this code
        # https://github.com/ajhalthor/spotify-recommender/blob/main/AudioSet%20Processing.ipynb
        base = tf.train.SequenceExample()
        base.ParseFromString(tf_file.numpy())

        # from here, we can pull the features we want from the tf file, starting with labels; as a note, even if there
        # is only one value for a call it is generally output in a list, so calling [0] at the end is helpful for those
        labels = base.context.feature['labels'].int64_list.value

        # for the purposes of this project, I needed animal videos without music; 72 is the label for animal videos, 137 is 
        # the label for videos with music; can be changed (or removed) depending on your needs; values of labels/values
        # are available from AudioSet
        if 72 in labels and 137 not in labels:
            vid_id = base.context.feature['video_id'].bytes_list.value[0]
            audio = base.feature_lists.feature_list['audio_embedding'].feature

            # running sum(list, []) takes a multidimensional list and flattents it to one-d, which is important for cosine
            # similarity-based recommendation
            audio_by_second = sum([list(second.bytes_list.value[0]) for second in audio],[])[:cutoff*bits]

            # drops any audio files that are shorter than the cutoff length
            if len(audio_by_second) < cutoff*bits:
                continue
            feats = {
                'video_id': vid_id,
                'audio': audio_by_second
            }
            features.append(feats)
        else:
            continue

    with open(json_location, 'w') as file:
        json.dump(repr(features), file)

# if you have data saved elsewhere, or are planning on saving the output elsewhere, just change these values
AudioExtractor('data/audioset/unbal_train/', 'data/audioset_data.json')