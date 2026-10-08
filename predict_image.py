import sys
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from importdata import forward_prop, load_mapping

# Usage: python predict_image.py my_character.png
path = sys.argv[1] if len(sys.argv) > 1 else 'char.png'

#Load trained model
params = np.load('emnist_params.npz')
W1, B1, W2, B2 = params['W1'], params['B1'], params['W2'], params['B2']
chars = load_mapping('data/emnist-balanced-mapping.txt')


def center_and_resize(pixels, size=28, margin=2):
    """Crop to the character, pad to a square, shrink, and center in a size x size frame."""
    ys, xs = np.where(pixels > 0.2 * pixels.max())
    if len(ys) == 0:
        return np.zeros((size, size))
    crop = pixels[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = crop.shape
    side = max(h, w)
    square = np.zeros((side, side))
    top, left = (side - h) // 2, (side - w) // 2
    square[top:top + h, left:left + w] = crop
    inner = size - 2 * margin
    small = Image.fromarray(square.astype(np.uint8)).resize((inner, inner), Image.LANCZOS)
    out = np.zeros((size, size))
    out[margin:size - margin, margin:size - margin] = np.array(small)
    return out


#Load image to pixel values
img = Image.open(path)

# Flatten transparency onto a white background
if img.mode in ('RGBA', 'LA') or 'transparency' in img.info:
    img = img.convert('RGBA')
    bg = Image.new('RGBA', img.size, (255, 255, 255, 255))
    img = Image.alpha_composite(bg, img)

img = img.convert('L')  # grayscale, 0-255
pixels = np.array(img, dtype=np.float64)

# EMNIST is a white character on a black background. If yours is dark-on-light, invert.
if pixels.mean() > 127:
    print("Looks like dark-on-light, inverting to match EMNIST")
    pixels = 255 - pixels

pixels = center_and_resize(pixels)
X = (pixels / 255.0).reshape(784, 1)  # same scaling and layout as training

#predict
_, _, _, A2 = forward_prop(W1, B1, W2, B2, X)   # importdata's version returns 4 values
probs = A2[:, 0]
top = np.argsort(probs)[::-1][:5]
pred = int(top[0])
print(f"Predicted: {chars[pred]} ({probs[pred]*100:.1f}% confident)")
for k in top:
    print(f"  {chars[int(k)]}: {probs[k]*100:5.1f}%")

#display
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))
ax1.imshow(pixels, cmap='gray')
ax1.set_title(f"Model input -> predicted {chars[pred]}")
ax1.axis('off')
ax2.bar([chars[int(k)] for k in top], probs[top])
ax2.set_xlabel("Character")
ax2.set_ylabel("Probability (top 5)")
plt.tight_layout()
plt.show()