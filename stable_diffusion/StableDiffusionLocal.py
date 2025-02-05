#pip install diffusers transformers accelerate torch
import os
import json
import zipfile
import torch
from diffusers import StableDiffusionPipeline, LMSDiscreteScheduler

def main():

    json_filename = "captions.json"
    if not os.path.exists(json_filename):
        print(f"Error: {json_filename} not found.")
        return

    with open(json_filename, "r") as f:
        data = json.load(f)

    caption_groups = data.get("image_captions", [])
    print(f"Found {len(caption_groups)} caption groups.")


    output_folder = "generated_images"
    os.makedirs(output_folder, exist_ok=True)
    print(f"Images will be saved in folder: {output_folder}")


    model_id = "runwayml/stable-diffusion-v1-5"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe = pipe.to(device)

    pipe.scheduler = LMSDiscreteScheduler.from_config(pipe.scheduler.config)

    num_inference_steps = 75   
    guidance_scale = 9.0       
    width, height = 512, 512   
    seed = 42                  
    generator = torch.Generator(device).manual_seed(seed)
    negative_prompt = "blurry, oversaturated, low resolution, deformed"

    for idx, group in enumerate(caption_groups):
        
        if not isinstance(group, list):
            print(f"Skipping index {idx} as it is not a list of captions.")
            continue

        
        prompt = ", ".join(group[:5])
        print(f"Generating image {idx} with prompt: {prompt}")

        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            width=width,
            height=height
        )
        image = result.images[0]

        local_filename = os.path.join(output_folder, f"generated_image_{idx}.png")
        image.save(local_filename)
        print(f"Saved image {idx} as {local_filename}")

    zip_filename = "generated_images_json.zip"
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(output_folder):
            for file in files:
                file_path = os.path.join(root, file)
                zipf.write(file_path, arcname=file)
    print(f"Created zip file: {zip_filename}")

if __name__ == "__main__":
    main()
