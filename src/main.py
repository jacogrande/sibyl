import sys
import firestore
import clip
import multiprocessing
import spectogram
import typesense
from embeddings import typesense 
import glob
import os
import mp3data
import gc

def create_spectrogram(artist, title, audio_filepath):
    # Limit the frequency range to 20kHz
    return spectogram.generate_spectrogram(artist, title, audio_filepath, freq_limit=(0, 20000)) 


def save_to_firestore(title, artist, audio_filepath):
    print("Saving song data to Firestore")
    docId = firestore.add_song(title, artist, audio_filepath)
    return docId

def generate_embeddings(spectrogram_image_path):
    print("Generating embeddings for spectrogram: " + spectrogram_image_path)
    return clip.get_embeddings(spectrogram_image_path)

def save_embeddings_to_typesense(song_id, embeddings):
    print("Saving embeddings to Typesense")
    document = {
        'song_id': song_id,
        'embedding': embeddings
    }
    typesense.client.collections['songs'].documents.create(document)

def get_song_paths(root_dir="data/songs"):
    """
    Retrieves a list of all song paths within the given root directory.

    :param root_dir: The root directory to search within.
    :return: A list of file paths to all the songs found.
    """
    # Read the data/songs directory and get the list of files (nested in subdirectories)
    song_paths = glob.glob(os.path.join(root_dir, '**', '*.mp3'), recursive=True)
    return song_paths
    
def process_song(song_path):
    print("\n=========================================\n")
    artist, title = mp3data.get_id3_tag_info(song_path)

    # Verify the song hasn't already been added

    song_exists, song_id = firestore.check_song_exists(title, artist)
    if song_exists: 
        print(f"Song {title} by {artist} already exists in Firestore.")
        return

    # embed song
    spectrogram_image_path = create_spectrogram(artist, title, song_path)
    if spectrogram_image_path is None:
        return
    docId = save_to_firestore(title, artist, song_path)
    embeddings = generate_embeddings(spectrogram_image_path)
    save_embeddings_to_typesense(docId, embeddings)
    gc.collect()

# first attempt: Command line callable with arguments for artist, title, and audiofile path
if __name__ == "__main__":
    # Read the data/songs directory and get the list of files (nested in subdirectories)
    # For each file, create a spectrogram, generate embeddings, and save to Firestore and Typesense
    song_paths = get_song_paths()
    print(f"Found {len(song_paths)} songs.")

    """
    Processes a list of song paths sequentially in separate processes.
    We need this in order to clear memory leaked by the librosa load function.
    """
    for song_path in song_paths:
        # Create a separate process for each song
        process_song(song_path)

    
