import numpy as np
import matplotlib.pyplot as plt

folder = '/home/jack/Code/VITON/Deep-Learning-Project-S24/data/zalando-hd-resized/train'

# Read images
img_agnostic = plt.imread(f'{folder}/agnostic-v3.2/00000_00.jpg')
image = plt.imread(f'{folder}/image/00000_00.jpg')
pose = plt.imread(f'{folder}/openpose_img/00000_00_rendered.png')
cloth = plt.imread(f'{folder}/cloth/00000_00.jpg')

# Concatenate the images
cat_input = np.concatenate((pose, img_agnostic, cloth), axis=2)

# Visualize the concatenated image
#plt.imshow(cat_input)

# Function to extract patches
def extract_patches(image, patch_size):
    patches = []
    for i in range(0, image.shape[0], patch_size):
        for j in range(0, image.shape[1], patch_size):
            patch = image[i:i+patch_size, j:j+patch_size, :]
            if patch.shape[0] == patch_size and patch.shape[1] == patch_size:
                patches.append(patch)
    return patches

# Extract 64x64 patches
patch_size = 64
patches = extract_patches(image, patch_size)

patches2 = extract_patches(cloth, patch_size)
patches3 = extract_patches(pose, patch_size)

# Visualizing the first 10 patches
fig, axes = plt.subplots(1, 30, figsize=(20, 4))
for i, ax in enumerate(axes):
    if i < 10:
        ax.imshow(patches2[i+10])
        ax.axis('off')
    elif i < 20:
        ax.imshow(patches[i+10])
        ax.axis('off')
    else:
        ax.imshow(patches3[i+5])
        ax.axis('off')

plt.show()