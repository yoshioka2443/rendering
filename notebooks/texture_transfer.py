# %%
import os
os.chdir(os.path.join(os.path.dirname(__file__), '..'))
print(os.getcwd())

import pyredner # pyredner will be the main Python module we import for redner.
import torch # We also import PyTorch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from utils.utils import load_ho_meta, apply_transform_to_mesh, calc_vertex_normals
from utils.mano import ManoLayer
from pathlib import Path
import math
from jupyterplot import ProgressPlot


# name = 'GPMF12'
# frame = 250
name = 'ABF10'
frame = 0
bg_img = np.array(Image.open(f'data/HO3D_v3/train/{name}/rgb/{frame:04d}.jpg'), dtype=np.float32)/255.0
anno = load_ho_meta(f'data/HO3D_v3/train/{name}/meta/{frame:04d}.pkl')
mano_layer = ManoLayer()
mano_layer.load_textures()
mano = mano_layer(anno)

resolution = bg_img.shape[:2]
uvs = torch.stack([mano_layer.uv[..., 0], 1 - mano_layer.uv[..., 1]], -1)

mano_redner = pyredner.Object(
    vertices = mano.vertices[0], 
    indices = mano_layer.faces.to(torch.int32), 
    uvs = torch.tensor(uvs, dtype=torch.float32),
    uv_indices = torch.tensor(mano_layer.face_uvs, dtype=torch.int32),
    normals = pyredner.compute_vertex_normal(mano.vertices[0], mano_layer.faces),
    material=pyredner.Material(
        diffuse_reflectance = mano_layer.tex_diffuse_mean.to(pyredner.get_device()),
        specular_reflectance = mano_layer.tex_spec_mean.to(pyredner.get_device()),
        # normal_map = mano_layer.tex_normal_mean.to(pyredner.get_device()),
    )
)

world2cam = torch.eye(4)
R = torch.diag(torch.tensor([-1.,1.,-1.]))
world2cam[:3,:3] = R
cam2world = world2cam.inverse()
K = torch.tensor(anno['camMat'], dtype=torch.float32)
fx, fy = K.diagonal()[:2]
px, py = K[:2,2]
intrinsic_mat = torch.tensor([
        [fx / resolution[1] * 2, 0.0000, px/resolution[1]-0.5],
        [0.0000, fy / resolution[1] * 2, py/resolution[0]-0.5],
        [0.0000, 0.0000, 1.0000]]
        )

camera = pyredner.Camera(
    intrinsic_mat=intrinsic_mat,
    position = torch.tensor([0, 0, 0.], dtype=torch.float32),
    look_at = torch.tensor([0, 0, -1.], dtype=torch.float32),
    up = torch.tensor([0, 1., 0], dtype=torch.float32),
    resolution=resolution,
)

scene = pyredner.Scene(camera = camera, objects = [mano_redner])
img = pyredner.render_albedo(scene)

plt.imshow(img.cpu().numpy())
plt.show()
# %%

''' ho3d camera matrix '''
world2cam = torch.eye(4)
''' convert opencv to opengl '''
cv2gl = torch.diag(torch.tensor([1., -1., -1., 1]))
world2cam = world2cam @ cv2gl

vertices_cam = mano.vertices[0] @ world2cam[:3,:3].T + world2cam[:3,3].T
vertices_ndc = vertices_cam @ K.T
vertices_screen = vertices_ndc[..., :-1] / vertices_ndc[..., -1:]

plt.imshow(bg_img)
plt.scatter(vertices_screen[:, 0], vertices_screen[:, 1])
plt.show()

#%%
vertices_uv = vertices_screen / torch.tensor([resolution[0], resolution[1]], dtype=torch.float32)
vertices_uv = torch.stack([vertices_uv[..., 0], 1-vertices_uv[..., 1]], -1)
print(vertices_uv.max(), vertices_uv.min())
mano_uv_redner = pyredner.Object(
    uvs = vertices_uv, 
    uv_indices = mano_layer.faces.to(torch.int32), 
    vertices = torch.cat([torch.tensor(uvs, dtype=torch.float32), torch.ones(uvs.shape[:-1] + (1,))], -1) * 2-1,
    # vertices = torch.cat([mano_layer.uv, torch.ones(mano_layer.uv.shape[:-1] + (1,))], -1) * 2 - 1,
    indices = torch.tensor(mano_layer.face_uvs, dtype=torch.int32),
    material=pyredner.Material(
        diffuse_reflectance = torch.tensor(bg_img).to(pyredner.get_device()),
    )
)
plt.scatter(uvs[:, 0], uvs[:, 1])
plt.show()

camera_uv = pyredner.Camera(
    camera_type = pyredner.camera_type.orthographic,
    position = torch.tensor([0, 0, 0.], dtype=torch.float32),
    look_at = torch.tensor([0, 0, 1.], dtype=torch.float32),
    up = torch.tensor([0, 1., 0], dtype=torch.float32),
    resolution=mano_layer.tex_diffuse_mean.shape[:2],
    # resolution=(512, 512)
)
    
transfered_texture = pyredner.render_albedo(pyredner.Scene(camera = camera_uv, objects = [mano_uv_redner]))
plt.imshow(transfered_texture.cpu().numpy())
plt.show()
# %%
