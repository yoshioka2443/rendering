# %%
import os
os.chdir(os.path.join(os.path.dirname(__file__), '..'))

import trimesh
import numpy as np
import imageio
import cv2
import matplotlib.pyplot as plt


# tex_diffuse = imageio.imread('./data/NIMBLE/tex/tex_diffuse_basis_00.png') / 255.0
# tex_spec = imageio.imread('./data/NIMBLE/tex/tex_spec_basis_00.png', ) / 255.0
tex_diffuse = cv2.imread('./data/NIMBLE/tex_mano/tex_normal_mean.png', cv2.IMREAD_UNCHANGED)#.astype(np.float32) #/ 255.0
print(tex_diffuse.shape)
# tex_diffuse = cv2.cvtColor(tex_diffuse, cv2.COLOR_BGR2RGBA)

# print(tex_diffuse[..., 0].max(), tex_diffuse[..., 1].max(), tex_diffuse[..., 2].max(), )
# plt.imshow(tex_diffuse)

mesh = trimesh.load_mesh(
    'data/mano_v1_2/MANO_UV_right.obj', 
    process=False, maintain_order=False)
uv = mesh.visual.uv
mesh.visual.material.image = tex_diffuse
# mesh.visual.material.diffuse = tex_diffuse

material = trimesh.visual.material.SimpleMaterial(
    image=tex_diffuse*255,
    # diffuse=tex_diffuse,
    # specular=tex_spec,
    )
color_visuals = trimesh.visual.texture.TextureVisuals(
    uv=uv, material=material, )
mesh=trimesh.Trimesh(
    vertices=mesh.vertices, faces=mesh.faces, visual=color_visuals, validate=False, process=True, smooth=True)

mesh.show()

# %%
