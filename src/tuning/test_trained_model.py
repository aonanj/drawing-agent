import torch
from diffusers.pipelines.auto_pipeline import AutoPipelineForText2Image

# --- Configuration ---
# 1. Base model trained on
base_model_path = "stabilityai/stable-diffusion-xl-base-1.0"

# 2. Path to saved QLoRA adapter
adapter_path = "outputs/sdxl-qlora/checkpoint-1000"

# --- Load the Pipeline ---
print("Loading pipeline...")
# bf16 for consistency with training config
pipe = AutoPipelineForText2Image.from_pretrained(
    base_model_path,
    torch_dtype=torch.bfloat16,
    variant="fp16"
).to("cuda")

# --- Load and Attach Adapter ---
print(f"Loading adapter from: {adapter_path}")
pipe.load_lora_weights(adapter_path)
print("Adapter loaded successfully.")

# --- Generate an Image ---
prompt = "a patent drawing of a mechanical gear assembly with callout number 12"
negative_prompt = "photograph, 3d render, blurry, grainy, text, watermark"

print(f"Generating image for prompt: {prompt}")
image = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_inference_steps=30,
    guidance_scale=7.5
).images[0]

output_image_path = "test_generation.png"
image.save(output_image_path)
print(f"Image saved to {output_image_path}")