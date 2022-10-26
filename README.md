# Audio-Based Video Recommendation System
A recommender system that recommends YouTube videos based on input audio

### Project Intro

This project takes audio inputs and embeddings and uses them to output video recommendations using [](https://research.google.com/audioset/ "AudioSet") data, while providing the code for some feature extraction for [](https://research.google.com/youtube8m/ "YouTube-8M") data. The reasoning is fairly simple; consumers are watching movies and TV with phones in-hand increasingly often, increasing the relative importance of audio. Additionally, as genres become less meaningful (BoJack Horseman and Parks and Recreation are both comedies, for example), using other features to recommend media content becomes increasingly relevant. Lastly, audio itself contains a fair amount of information, including tone/mood and production quality.

### Methods
TensorFlow Data Extraction  
Audio Embedding  
Cosine Similarity/Recommender Systems  

### Requirements
[](https://www.python.org/downloads/ "Python")
[](https://www.tensorflow.org/install "TensorFlow")
[](https://github.com/tensorflow/models/tree/master/research/audioset/vggish "VGG-ish")
[](https://numpy.org/ "Numpy")
[](https://pandas.pydata.org/ "Pandas")
[](https://scikit-learn.org/stable/ "scikit-learn")

### The Process
#### A Brief Summary of the files and the order in which they should be run

##### AudioSet_Extraction.py
This file is dedicated to extracting audio data from Google's Audioset database. The database contains over 2 million YouTube clips, labeled by genre and stored in TensorFlow Records. I decided to exclude videos with music in them, as music would dominate the audio embeddings. From there, I chose to use Animal videos as my category. AudioSet provides a csv file showing the numerical values for all their labels, so changing which categories to include or not include is fairly straightforward. This would be ideally downloaded into the AudioSet folder within the data folder.

###### yt8m_extraction.py
A very similar file to AudioSet_Extraction.py, the main difference being that it is less organized and uses Google's follow-up to AudioSet, YouTube-8M. There are significantly more available videos in this set (over two terabytes worth), so if you have the computational capabilities this file can be used to extract features from the dataset. YouTube-8M also provides embeddings of visual features.

##### DataFrame_Creation.py
Simply takes the read data and applies some formatting and normalization while putting it into a Pandas DataFrame.

##### Cosine_Similarity_Recommender.py
This file contains the code to create a recommender based on cosine similarity. Specifically, this code takes a Video_ID of one of the videos in the AudioSet data and outputs other, similar videos.

##### New_Sounds_Recommender.py
This file also contains a cosine similarity-based recommender, but instead of comparing videos in the dataset to each other, it takes .wav files that were run through VGG-ish's embedding and finds similar related videos. The code to download VGG-ish is provided above; of note is that it appears a recent numpy update has led to their test file failing (it now falls ever-so-slightly outside of their set bounds on expected returns), but outside of that the code is functional. Running:
$ python vggish_inference_demo.py --wav_file /path/to/a/wav/file \
                                    --tfrecord_file /path/to/tfrecord/file \
                                    --checkpoint /path/to/model/checkpoint \
                                    --pca_params /path/to/pca/params
will create embeddings of wav files that can then be run through the recommender.

### Future Improvements
Potential future improvements include integrating the process of embedding wav files directly into the code, as well as running a similar model with more data from the YouTube-8M dataset, and potentially incorporating visual features in the recommender.
