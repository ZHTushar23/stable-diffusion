# make sure you're logged in with `huggingface-cli login`
from torch import autocast
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained(
	"CompVis/stable-diffusion-v1-1", 
	use_auth_token=False
).to("cuda")

prompt = "a photo of a model on red carpet"
with autocast("cuda"):
	image = pipe(prompt)["images"][0]
	print(type(image))
    # image = pipe(prompt)["sample"][0]  
    
image.save("model.png")
