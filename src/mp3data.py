import eyed3

def get_id3_tag_info(file_path):
    # Load an MP3 file
    audiofile = eyed3.load(file_path)
    
    # Access the tags
    if audiofile.tag is not None:
        artist = audiofile.tag.artist
        title = audiofile.tag.title
        return artist, title
    else:
        # Handle case where there are no ID3 tags
        print(f"No ID3 tag found for {file_path}")
        return None, None

# Usage
# mp3_file_path = 'path/to/your/musicfile.mp3'
# artist, title = get_id3_tag_info(mp3_file_path)
# if artist and title:
    # print(f"Artist: {artist}, Title: {title}")
