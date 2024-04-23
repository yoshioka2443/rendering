# %%

from pathlib import Path
import pickle
import cv2
import numpy as np

nimble_path = '../data/NIMBLE'
pklfiles = Path(nimble_path).glob("*.pkl")
nimble_data = {}
for fp in pklfiles:
    with open(fp, "rb") as f:
        data = pickle.load(f, encoding="latin1")
    nimble_data[fp.stem] = data

for texname in ['diffuse', 'spec', 'normal']:
    tex_diff_mean  = nimble_data['NIMBLE_TEX_DICT'][texname]['mean'].reshape(1024,1024,3)
    tex_diff_basis  = nimble_data['NIMBLE_TEX_DICT'][texname]['basis'].reshape(1024,1024,3, 10)
    tex_diff_std  = nimble_data['NIMBLE_TEX_DICT'][texname]['std']
    
    for i in range(10):
        diff = tex_diff_basis[..., i] * tex_diff_std[i] + tex_diff_mean
        diff = np.clip(diff, 0.0, 1.0)
        cv2.imwrite(f"../data/NIMBLE/tex/tex_{texname}_basis_{i:02d}.png", diff*255)

# for tex_diff_dim in tex_diff_basis:
# %%
