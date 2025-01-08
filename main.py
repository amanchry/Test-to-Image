from huggingface_hub import InferenceClient

# Initialize the client
client = InferenceClient("black-forest-labs/FLUX.1-dev", token="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")

# Generate image from text
print("Generating image...")
image = client.text_to_image("Astronaut riding a horse")

# Save the image to a file
output_image_path = "generated_image.png"
if image:
    # Convert the PIL.Image object to bytes and save
    image.save(output_image_path, format="PNG")
    print(f"Image generated and saved as '{output_image_path}'")
else:
    print("Failed to generate image.")

