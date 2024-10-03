# %%
import os
os.chdir(os.path.join(os.path.dirname(__file__), '..'))
print(os.getcwd())

import pyredner # pyredner will be the main Python module we import for redner.
import torch # We also import PyTorch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from utils.utils import load_ho_meta, apply_transform_to_mesh
from utils.mano import ManoLayer


bg_img = np.array(Image.open('data/HO3D_v3/train/ABF10/rgb/0000.jpg'), dtype=np.float32)/255.0
anno = load_ho_meta('data/HO3D_v3/train/ABF10/meta/0000.pkl')
mano_layer = ManoLayer()
mano_layer.load_textures()
mano = mano_layer(anno)

resolution = bg_img.shape[:2]

def add_triangles2D(ax, uv, faces, **kwargs):
    from matplotlib import patches
    from matplotlib.collections import PatchCollection
    print('uv.shape', uv.shape)
    uv_image = np.stack([uv[..., 0], 1-uv[..., 1]], -1) * 1024
    print(uv_image)
    patch_list = [
        patches.Polygon(f) for f in uv_image[faces]
    ]
    colors = plt.cm.viridis(np.linspace(0, 1, len(patch_list)))
    pc = PatchCollection(patch_list, alpha=1, facecolor='None', edgecolor='white', **kwargs)
    ax.add_collection(pc)

x = np.linspace(0, 1, 1024)
y = np.linspace(0, 1, 1024)
X, Y = np.meshgrid(x, y)
uv_tex = torch.tensor(np.stack([X, Y, np.zeros_like(X)], -1), dtype=torch.float32)

uvs = torch.stack([mano_layer.uv[..., 0], 1-mano_layer.uv[..., 1]], -1)


mano_redner = pyredner.Object(
    vertices = mano.vertices[0], 
    indices = mano_layer.faces.to(torch.int32), 
    uvs = torch.tensor(uvs, dtype=torch.float32),
    uv_indices = torch.tensor(mano_layer.face_uvs, dtype=torch.int32),
    material=pyredner.Material(
        diffuse_reflectance = mano_layer.tex_diffuse_mean.to(pyredner.get_device()),
        specular_reflectance = mano_layer.tex_spec_mean.to(pyredner.get_device()),
        normal_map = mano_layer.tex_normal_mean.to(pyredner.get_device()),
    )
)


world2cam = torch.eye(4)
R = torch.diag(torch.tensor([-1.,1.,-1.]))
world2cam[:3,:3] = R
cam2world = world2cam.inverse()
K = torch.tensor(anno['camMat'], dtype=torch.float32)
fx, fy = K.diagonal()[:2]
px, py = K[:2,2]
print(K)
intrinsic_mat = torch.tensor([
        [fx / resolution[1] * 2, 0.0000, px/resolution[1]-0.5],
        [0.0000, fy / resolution[1] * 2, py/resolution[0]-0.5],
        [0.0000, 0.0000, 1.0000]]
        )

fov = 2* torch.atan(0.5 * resolution[1] / K[0, 0]) * 180 / 3.1415926
print(fov)
camera = pyredner.Camera(
    intrinsic_mat=intrinsic_mat,
    # cam_to_world=cam2world,
    position = torch.tensor([0, 0, 0.], dtype=torch.float32),
    look_at = torch.tensor([0, 0, -1.], dtype=torch.float32),
    up = torch.tensor([0, 1., 0], dtype=torch.float32),
    # fov = torch.tensor([fov], dtype=torch.float32),
    resolution=resolution,
)
print(camera.__dict__)
# dirlight = pyredner.DirectionalLight(
#     direction = torch.tensor([0.0, 0.0, -1.0]), 
#     intensity = torch.ones(3)*3.0,
# )
light = pyredner.AmbientLight(intensity=torch.tensor([1., 1., 1.]))
# envmap = pyredner.EnvironmentMap(torch.tensor(bg_img))

objects = pyredner.load_obj('data/models/021_bleach_cleanser/textured_simple.obj', return_objects=True)

obj_redner = pyredner.Object(
    vertices=apply_transform_to_mesh(objects[0].vertices, anno),
    indices=objects[0].indices, 
    uvs=objects[0].uvs,
    uv_indices=objects[0].uv_indices,
    material=pyredner.Material(
        diffuse_reflectance = torch.pow(objects[0].material.diffuse_reflectance._texels.to(pyredner.get_device()), 2.2),
        specular_reflectance = objects[0].material._specular_reflectance._texels.to(pyredner.get_device()),
        roughness = objects[0].material.roughness._texels.to(pyredner.get_device()),
    )
)


# create scene
scene = pyredner.Scene(
    camera = camera, 
    objects = [
        mano_redner, 
        obj_redner,
        ]
    )

# Render the scene.
render = pyredner.render_deferred(scene, lights=[light], alpha=True)    

fig, axs = plt.subplots(1, 2, figsize=(10, 5))
for ax in axs:
    ax.axis('off')
    ax.imshow(bg_img)
axs[1].imshow(torch.pow(render, 1.0/2.2).cpu())
plt.show()


# %%
