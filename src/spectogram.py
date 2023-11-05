import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
from firestore import add_song
import subprocess
import tempfile
import re
import pydub
import math
import gc

def sanitize_filename(filename):
    """
    Sanitize the filename by removing or replacing characters that are invalid for file paths.
    """
    # Replace reserved characters with an underscore
    return re.sub(r'[\\/:"*?<>|]+', '_', filename)

def generate_spectrogram(artist, title, filepath, freq_limit=None):
    """
    Generate a spectrogram for the given audio file.

    Args:
    - filepath (str): Path to the audio file.

    Returns:
    - str: Path where the spectrogram image is saved.
    """
    try:
        print("Creating spectrogram for " + title + " by " + artist + " from file " + filepath + "\n")

        # get file length
        dur = pydub.utils.mediainfo(filepath)["duration"]

        # Load the audio file
        y, sr = librosa.load(filepath, duration=math.floor(float(dur)), sr=None)
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)

        # Plotting
        plt.figure(figsize=(10, 4))
        ax = plt.gca()
        librosa.display.specshow(D, sr=sr, x_axis=None, y_axis=None, ax=ax)

        # Limiting the frequency range
        if freq_limit:
            ax.set_ylim(librosa.hz_to_mel(freq_limit))

        # Removing all axes, labels, whitespace, and legends
        ax.set_axis_off()
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        ax.xaxis.set_major_locator(plt.NullLocator())
        ax.yaxis.set_major_locator(plt.NullLocator())

        # Save the spectrogram
        output_directory = "data/spectrograms"
        os.makedirs(output_directory, exist_ok=True)
        base_filename = artist + " - " + title + " - spectrogram.png"
        filename = sanitize_filename(base_filename)
        output_path = os.path.join(output_directory, filename)
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)

        plt.cla()
        plt.clf()
        plt.close('all')
        del ax

        return output_path
    except Exception as e:
        if plt:
            plt.cla()
            plt.clf()
            plt.close('all')
        gc.collect()
        print("Error generating spectrogram for audio file: " + filepath)
        print(e)
        return None
