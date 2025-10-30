import torch
from diffusers.pipelines.auto_pipeline import AutoPipelineForText2Image

# --- Configuration ---
# 1. The base model
base_model_path = "stabilityai/stable-diffusion-xl-base-1.0"

# 2. Path to your NEW, better adapter
# (This assumes your run was named 'sdxl-qlora-run-2')
adapter_path = "outputs/sdxl-qlora-run-2/unet_lora" 

# --- Load the Pipeline ---
print("Loading pipeline...")
pipe = AutoPipelineForText2Image.from_pretrained(
    base_model_path,
    torch_dtype=torch.bfloat16,
    variant="fp16"
).to("cuda")

# --- Load and Attach Your Adapter ---
print(f"Loading adapter from: {adapter_path}")
# Note: The folder *inside* checkpoint-1000 is often 'unet_lora'
# If that path doesn't work, try "outputs/sdxl-qlora-run-2/checkpoint-1000"
pipe.load_lora_weights(adapter_path) 
print("Adapter loaded successfully.")

# --- Generate an Image ---
prompt = """
Generate figure 1 for a US utility patent in CPC F24F. Figure 1 should be a block diagram illustrating the configuration of an air conditioning system. The air conditioning system comprises a plurality of air conditioners (labels 21, 22, and 23), and a central controller (label 10) which centrally and individually controls air conditioning behavior thereof.
"""
negative_prompt = "photograph, 3d render, blurry, grainy, text, watermark"

print(f"Generating image for prompt: {prompt}")
image = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_inference_steps=30,
    guidance_scale=7.5
).images[0]

# --- Save the Result ---
output_image_path = "test_generation_run2.png"
image.save(output_image_path)
print(f"Image saved to {output_image_path}")