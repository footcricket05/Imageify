from flask import Flask, render_template, request
from diffusers import StableDiffusionPipeline
import torch
import os
import uuid

app = Flask(__name__)

# Function to generate an image from a text prompt
def generate_image(prompt):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Generate a unique identifier for this request
    request_id = str(uuid.uuid4())[:8]  # Generate an 8-character unique ID
    print(f"Request ID: {request_id}")

    # Initialize the pipeline with environment variable for the API token
    hf_token = os.getenv("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")
    print(f"Hugging Face Token: {hf_token}")
    pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", use_auth_token=hf_token).to(device)

    with torch.no_grad():
        image = pipe(prompt).images[0]

    # Saving in static folder to serve it later with a unique filename
    output_path = f"generated_image_{request_id}.png"  # Removed 'static/' from the path
    image.save(os.path.join('static', output_path))  # Saving directly to the static folder

    # Return both the image path and the request ID
    return output_path, request_id

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        prompt = request.form.get('prompt')
        if prompt:
            image_path, request_id = generate_image(prompt)
            # Pass both the image path and the request ID to the template
            return render_template('show_image.html', image_path=image_path, request_id=request_id)
    return render_template('index.html')

if __name__ == "__main__":
    app.run()
