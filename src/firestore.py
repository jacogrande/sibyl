# firestore.py
from firebase import db
import datetime
import warnings
warnings.filterwarnings('ignore')

def add_song(title, artist, filepath):
    """Adds a new song to the 'songs' collection with 'title' and 'artist'."""
    try:
        # Create a new song in the "songs" collection
        song_ref = db.collection('songs').add({
            'title': title,
            'artist': artist,
            'filepath': filepath,
            'created': datetime.datetime.now()
        })
        return song_ref[1].id
    except Exception as e:
        print(f"An error occurred: {e}")

def check_song_exists(title, artist):
    """Checks if a song with 'title' and 'artist' exists in the 'songs' collection."""
    try:
        # Query the "songs" collection for the song
        song_ref = db.collection('songs').where('title', '==', title).where('artist', '==', artist).get()
        if len(song_ref) > 0:
            return True, song_ref[0].id
        else:
            return False, None
    except Exception as e:
        print(f"An error occurred: {e}")
        return False, None
