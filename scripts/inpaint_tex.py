# %%
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
os.chdir(os.path.join(os.path.dirname(__file__), '..'))
print(os.getcwd())


tex_paths = sorted(Path('data/NIMBLE/tex_mano_inpaint').glob('*.png'))

for tex_path in tqdm(tex_paths):
    tex = cv2.imread(str(tex_path), cv2.IMREAD_UNCHANGED)
    rgb = tex[..., :3]
    alpha = tex[..., 3] 
    mask = 255 - alpha
    dst = cv2.inpaint(rgb, mask, 3, cv2.INPAINT_TELEA)
    
    cv2.imwrite(str(tex_path), dst)
    
    # fig, axs = plt.subplots(1, 2)
    # axs[0].imshow(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
    # axs[1].imshow(cv2.cvtColor(dst, cv2.COLOR_BGR2RGB))
    # plt.show()
    # break
    # tex = cv2.cvtColor(tex, cv2.COLOR_BGR2RGB)
    # tex = tex / 255.0
    # tex = tex.astype(np.float32)
    # tex = tex[::-1, :, :]
    # tex = np.flip(tex, 1)
    # tex = tex.copy()
    # cv2.imwrite(str(tex_path), tex)


# %%
