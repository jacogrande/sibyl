import sys
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import os
# Disable parallelism to run each model asynchonously 
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def get_clip_embedder(model_name="openai/clip-vit-base-patch16"):
    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name)
    model.eval()
    return processor, model

def embed_image(processor, model, image_path, text="a photo of a spectrogram"):
    image = Image.open(image_path)
    inputs = processor(text=text, images=image, return_tensors="pt", padding=True)
    outputs = model(**inputs)        
    return outputs.image_embeds[0]

def get_embeddings(image_path):
    processor, model = get_clip_embedder()
    embedding = embed_image(processor, model, image_path)
    
    print(f"Embedding Dimension: {embedding.shape[0]}")
    return embedding.tolist()
