import os

from PIL import Image

image_paths = [f"Agent02_Trained/image{_}.png" for _ in range(1,594)]
output_path = "agent02_trained.gif"
frames = [Image.open(image_path) for image_path in image_paths]
frames[0].save(output_path, save_all=True, append_images=frames[1:], loop=0)