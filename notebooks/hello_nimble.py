#%%
import pickle
import numpy as np
import matplotlib.pyplot as plt
import cv2
import imageio

nimble_tex_dict_fp = "/Users/shugo/workspace/rendering/data/NIMBLE/NIMBLE_TEX_DICT.pkl"
with open(nimble_tex_dict_fp, 'rb') as f:
    nimble_tex_dict = pickle.load(f, encoding='latin1')

# print(nimbple_tex_dict)
for k, v in nimble_tex_dict.items():
    print(k, len(v))
    for kk, vv in v.items():
        print("   -", kk, np.array(vv).shape)

for k, v in nimble_tex_dict.items():
    tex_mean = np.array(nimble_tex_dict[k]['mean']).reshape(1024, 1024, 3)
    # if k == 'diffuse':
    tex_mean = cv2.cvtColor(tex_mean, cv2.COLOR_BGR2RGB)
    if k == 'normal':
        print(tex_mean.min(), tex_mean.max())

    plt.imshow(tex_mean)
    imageio.imwrite(f"../data/NIMBLE/rand_0_{k}.png", (tex_mean*255).astype(np.uint8))