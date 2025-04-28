import os

from PIL import Image

image_paths = [f"Images/image{_}.png" for _ in range(1,523)]
output_path = "random_agent.gif"
frames = [Image.open(image_path) for image_path in image_paths]
frames[0].save(output_path, save_all=True, append_images=frames[1:], loop=0)