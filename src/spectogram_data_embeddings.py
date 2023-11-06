import librosa
import numpy as np
import scipy
import math
import pydub

DIMENSIONS = 2000 

def generate_features(y, sr):
    # Compute spectrogram
    D = np.abs(librosa.stft(y))

    # Extract various statistics from the spectrogram
    means = np.mean(D, axis=1)
    std_devs = np.std(D, axis=1)
    skewness = scipy.stats.skew(D, axis=1)

    # Compute MFCCs and its derivatives
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    delta_mfccs = librosa.feature.delta(mfccs)
    delta2_mfccs = librosa.feature.delta(mfccs, order=2)

    # Extract statistics from MFCCs
    mfcc_means = np.mean(mfccs, axis=1)
    mfcc_deltas_means = np.mean(delta_mfccs, axis=1)
    mfcc_delta2_means = np.mean(delta2_mfccs, axis=1)

    # Concatenate features
    features = np.concatenate([means, std_devs, skewness, 
                               mfcc_means, mfcc_deltas_means, mfcc_delta2_means])

    # If the feature vector is too long, truncate or sample from it; if too short, pad with zeros.
    if len(features) > DIMENSIONS:
        features = features[:DIMENSIONS]
    elif len(features) < DIMENSIONS:
        features = np.concatenate([features, np.zeros(DIMENSIONS - len(features))])

    return features

def process_filepath_to_vector(filepath, sr=None):
    # get file length
    dur = pydub.utils.mediainfo(filepath)["duration"]

    # Load the audio file
    y, sr = librosa.load(filepath, duration=math.floor(float(dur)), sr=None)
    return generate_features(y, sr)

def get_embedding(filepath):
    return process_filepath_to_vector(filepath)
