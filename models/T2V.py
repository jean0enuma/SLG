import torch
import cv2
import torch
from diffusers import DiffusionPipeline
import torch
from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained("Wan-AI/Wan2.1-T2V-1.3B", dtype=torch.bfloat16, device_map="cuda")

prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
image = pipe(prompt).images[0]
# 画像を表示
cv2.imshow("Generated Image", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
cv2.waitKey(0)
cv2.destroyAllWindows()
