import sys
import firestore
import clip
import spectogram
from embeddings import typesense 

def create_spectrogram(artist, title, audio_filepath):
    print(f"Creating spectrogram for audio file: {audio_filepath}")
    return spectogram.generate_spectrogram(artist, title, audio_filepath, freq_limit=(0, 20000))

def save_to_firestore(title, artist, audio_filepath):
    print("Saving song data to Firestore")
    return firestore.add_song(title, artist, audio_filepath)

def generate_embeddings(spectrogram_image_path):
    print(f"Generating embeddings for spectrogram: {spectrogram_image_path}")
    return clip.get_embeddings(spectrogram_image_path)

def save_embeddings_to_typesense(song_id, embeddings):
    print("Saving embeddings to Typesense")
    document = {
        'song_id': song_id,
        'embedding': embeddings
    }
    typesense.client.collections['songs'].documents.create(document)

def search_song_in_typesense(embeddings):
    print("Searching song in Typesense")
    # Assuming you have a function to search based on embeddings
    return typesense.search_song_by_embedding(embeddings)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: search.py <artist> <title> <audio_filepath>")
        sys.exit(1)

    artist = sys.argv[1]
    title = sys.argv[2]
    audio_filepath = sys.argv[3]

    # Check if the song already exists in Firestore
    song_exists, song_id = firestore.check_song_exists(title, artist)

    if not song_exists:
        # The song does not exist, so go through the steps to create and save it
        spectrogram_image_path = create_spectrogram(artist, title, audio_filepath)
        embeddings = generate_embeddings(spectrogram_image_path)
        docId = save_to_firestore(title, artist, audio_filepath)
        save_embeddings_to_typesense(docId, embeddings)
        search_results = search_song_in_typesense(embeddings)
    else:
        # The song exists, fetch the embeddings from Typesense
        embeddings = typesense.fetch_embeddings(song_id)
        search_results = search_song_in_typesense(embeddings)

    print(f"Search results: {search_results}")
