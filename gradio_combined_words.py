import gradio as gr
from openai import OpenAI
import requests
from PIL import Image
from io import BytesIO
import os

prompts = []

def add_text(new_text):
    prompts.append(new_text)
    prompt_n = f"prompt_{len(prompts)}"
    combined = " ".join(prompts)
    
    # Generate image using DALL-E
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("Please set OPENAI_API_KEY in your .env file")
    client = OpenAI(api_key=api_key)
    
    # Generate image for new text
    response1 = client.images.generate(
        model="dall-e-2",
        prompt=new_text,
        size="512x512",
        quality="standard",
        n=1,
    )
    
    # Get image URL and download image for new text
    image_url1 = response1.data[0].url
    response_img1 = requests.get(image_url1)
    image1 = Image.open(BytesIO(response_img1.content))
    
    # Generate image for combined text
    response2 = client.images.generate(
        model="dall-e-2",
        prompt=combined,
        size="512x512",
        quality="standard",
        n=1,
    )
    
    # Get image URL and download image for combined text
    image_url2 = response2.data[0].url
    response_img2 = requests.get(image_url2)
    image2 = Image.open(BytesIO(response_img2.content))
    
    # Return values directly matching the output components
    return f"{prompt_n}: {combined}", image1, image2

interface = gr.Interface(
    fn=add_text,
    inputs=gr.Textbox(label="Enter text"),
    outputs=[
        gr.Textbox(label="Combined text"),
        gr.Image(label="Current prompt image"),
        gr.Image(label="Combined prompts image")
    ],
    title="Text and Image Generator", 
    description="Enter text to generate images using DALL-E"
)
interface.launch()
