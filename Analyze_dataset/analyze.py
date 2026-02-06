from datasets import load_from_disk
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import io
print("Loading dataset...")

ds = load_from_disk("data/mjsynth_300k")
print(f"✓ Loaded from disk: {len(ds)} samples")

for i in range(3):
    sample = ds[i]
    plt.imshow(sample["image"])
    plt.title(sample["label"])
    plt.axis("off")
    # plt.show()

## Analyze data structure:
sample = ds[0]
print("\nFirst sample keys:", sample.keys())
print("\nSample structure:")
for key, value in sample.items():
    print(f"  {key}: {type(value)}")
    
##Analyze image structure:
image_sizes = []
aspect_ratios = []
for i in range(min(1000, len(ds))):
    img = ds[i]['image']
    if isinstance(img, Image.Image):
        width, height = img.size
        image_sizes.append((width, height))
        aspect_ratios.append(width / height)

widths = [s[0] for s in image_sizes]
heights = [s[1] for s in image_sizes]

print(f"\nImage dimensions (from {len(image_sizes)} samples):")
print(f"  Width:  min={min(widths)}, max={max(widths)}, mean={np.mean(widths):.1f}")
print(f"  Height: min={min(heights)}, max={max(heights)}, mean={np.mean(heights):.1f}")
print(f"  Aspect ratio: min={min(aspect_ratios):.2f}, max={max(aspect_ratios):.2f}, mean={np.mean(aspect_ratios):.2f}")