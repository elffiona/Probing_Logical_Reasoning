import torch
from torch import nn
from transformers import BitsAndBytesConfig
from diffusers import StableDiffusion3Pipeline
import json
from tqdm import tqdm
import os

# Load the categories I want to generate
with open("PATH/category.json", "r") as f:
    categories = json.load(f)

pipe = StableDiffusion3Pipeline.from_pretrained(
    "stabilityai/stable-diffusion-3-medium-diffusers", 
    torch_dtype=torch.float16,
    token = 'YOUR_TOKEN')
pipe = pipe.to("cuda")

for c in tqdm(categories):
    cat_name = c["name"]
    prompt = f"A image of a {cat_name} with white background."
    # print(f"Generating {cat_name} images...")

    for i in range(10):
        image_name = f"{cat_name}_{i}.png"
        # Check if the image already exists
        if os.path.exists(f"PATH/sd3/{image_name}"):
            print(f"Image {image_name} already exists, skipping...")
            continue
        else:
            # Generate the image
            image = pipe(
                prompt=prompt,
                negative_prompt="",
                num_inference_steps=28,
                guidance_scale=7.0,
            ).images[0]
            image.save(f"PATH/sd3/{image_name}")