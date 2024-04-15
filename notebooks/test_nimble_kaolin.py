# %%
import os
os.chdir(os.path.join(os.path.dirname(__file__), '..'))
print(os.getcwd())

#import pyredner # pyredner will be the main Python module we import for redner.
import kaolin
import torch # We also import PyTorch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from utils.utils import load_ho_meta, apply_transform_to_mesh
from utils.mano import ManoLayer


bg_img = np.array(Image.open('data/HO3D_v3/train/ABF10/rgb/0000.jpg'), dtype=np.float32)/255.0
anno = load_ho_meta('data/HO3D_v3/train/ABF10/meta/0000.pkl')
mano_layer = ManoLayer()
mano_layer.load_nimble()
mano = mano_layer(anno)

resolution = bg_img.shape[:2]

# fig, ax = plt.subplots()
# ax.imshow(bg_img)
# ax.scatter(mano.vertices[0, :, 0] * resolution[1], (1-mano.vertices[0, :, 1]) * resolution[0], s=0.1)

from utils.kaolinRenderer import KaolinRenderer

renderer = KaolinRenderer(mano.vertices[0], mano_layer.faces, mano_layer.uv, mano_layer.tex_diff_mean)
