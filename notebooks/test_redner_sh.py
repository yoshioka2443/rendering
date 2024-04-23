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

# %%
bg_img = np.array(Image.open('data/HO3D_v3/train/ABF10/rgb/0000.jpg'), dtype=np.float32)/255.0
anno = load_ho_meta('data/HO3D_v3/train/ABF10/meta/0000.pkl')
mano_layer = ManoLayer()
mano_layer.load_textures()
mano = mano_layer(anno)

resolution = bg_img.shape[:2]

uvs = torch.stack([mano_layer.uv[..., 0], 1-mano_layer.uv[..., 1]], -1)

mano_object = pyredner.Object(
    vertices = mano.vertices[0], 
    indices = mano_layer.faces.to(torch.int32), 
    uvs = torch.tensor(uvs, dtype=torch.float32),
    uv_indices = torch.tensor(mano_layer.face_uvs, dtype=torch.int32),
    material=pyredner.Material(
        diffuse_reflectance = mano_layer.tex_diffuse_mean.to(pyredner.get_device()),
        specular_reflectance = mano_layer.tex_spec_mean.to(pyredner.get_device()),
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

fov = 2* torch.atan(0.5 * resolution[1] / K[0, 0]) * 180 / 3.1415926
camera = pyredner.Camera(
    intrinsic_mat=intrinsic_mat,
    # cam_to_world=cam2world,
    position = torch.tensor([0, 0, 0.], dtype=torch.float32),
    look_at = torch.tensor([0, 0, -1.], dtype=torch.float32),
    up = torch.tensor([0, 1., 0], dtype=torch.float32),
    # fov = torch.tensor([fov], dtype=torch.float32),
    resolution=resolution,
)

dirlight = pyredner.AmbientLight(intensity=torch.tensor([1., 1., 1.]))

# envmap = pyredner.EnvironmentMap(torch.tensor(bg_img))

# %%
objects = pyredner.load_obj('data/models/021_bleach_cleanser/textured_simple.obj', return_objects=True)

# sRGB to Linear RGB
object_diffuse_texels = objects[0].material.diffuse_reflectance._texels
objects[0].material.diffuse_reflectance._texels = torch.pow(object_diffuse_texels, 2.2)

obj_object = pyredner.Object(
    vertices=apply_transform_to_mesh(objects[0].vertices, anno),
    indices=objects[0].indices, 
    uvs=objects[0].uvs,
    uv_indices=objects[0].uv_indices,
    material=objects[0].material
)

# create scene
scene = pyredner.Scene(
    camera = camera, 
    objects = [
        mano_object, 
        obj_object,
        ]
    )

# Render the scene.
render_albedo = pyredner.render_albedo(scene, alpha=True)
# render_deferred = pyredner.render_deferred(scene, lights=[dirlight], alpha=True)
# print(render_img.shape)

mask = render_albedo[..., -1]

show_dict = {
    'input': bg_img,
    'albedo': torch.pow(render_albedo, 1.0/2.2).cpu(),
}

fig, axs = plt.subplots(1, len(show_dict), figsize=(5*len(show_dict), 5))
for ax, (k, v) in zip(axs, show_dict.items()):
    ax.imshow(bg_img)
    ax.set_title(k)
    ax.imshow(v)

plt.show()

# %%

def deringing(coeffs, window):
    deringed_coeffs = torch.zeros_like(coeffs)
    deringed_coeffs[:, 0] += coeffs[:, 0]
    deringed_coeffs[:, 1:1 + 3] += \
        coeffs[:, 1:1 + 3] * math.pow(math.sin(math.pi * 1.0 / window) / (math.pi * 1.0 / window), 4.0)
    deringed_coeffs[:, 4:4 + 5] += \
        coeffs[:, 4:4 + 5] * math.pow(math.sin(math.pi * 2.0 / window) / (math.pi * 2.0 / window), 4.0)
    deringed_coeffs[:, 9:9 + 7] += \
        coeffs[:, 9:9 + 7] * math.pow(math.sin(math.pi * 3.0 / window) / (math.pi * 3.0 / window), 4.0)
    return deringed_coeffs

# Reset the coefficients to some constant color, repeat the same process as in target envmap
coeffs = torch.tensor([[ 0.5,
                         0.0, 0.0, 0.0,
                         0.0, 0.0, 0.0, 0.0, 0.0,
                         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], # coeffs for red
                       [ 0.5,
                         0.0, 0.0, 0.0,
                         0.0, 0.0, 0.0, 0.0, 0.0,
                         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], # coeffs for green
                       [ 0.5,
                         0.0, 0.0, 0.0,
                         0.0, 0.0, 0.0, 0.0, 0.0,
                         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], # coeffs for blue
                       device = pyredner.get_device(),
                       requires_grad = True)
res = (128, 128)

materials = [
    pyredner.Material(
        diffuse_reflectance = mano_layer.tex_diffuse_mean.to(pyredner.get_device()),
        specular_reflectance = mano_layer.tex_spec_mean.to(pyredner.get_device()),
    ),
    objects[0].material,
]

vertex_normals = calc_vertex_normals(mano.vertices[0], mano_layer.faces)

mano_shape = pyredner.Shape(\
    vertices = mano.vertices[0], 
    indices = mano_layer.faces.to(torch.int32), 
    uvs = torch.tensor(uvs, dtype=torch.float32),
    uv_indices = torch.tensor(mano_layer.face_uvs, dtype=torch.int32),
    normals = torch.tensor(vertex_normals, dtype=torch.float32),
    normal_indices=mano_layer.faces.to(torch.int32),
    material_id = 0)

obj_shape = pyredner.Shape(
    vertices=apply_transform_to_mesh(objects[0].vertices, anno),
    indices=objects[0].indices, 
    uvs=objects[0].uvs,
    uv_indices=objects[0].uv_indices,
    material_id=1
)
shapes = [mano_shape, obj_shape]
render = pyredner.RenderFunction.apply

target = torch.pow(torch.tensor(bg_img, device=pyredner.get_device()), 2.2)
# Finally we can start the Adam iteration
optimizer = torch.optim.Adam([coeffs,], lr=3e-2)

save_dir = "results/joint_material_envmap_sh_ho"
# Path('results/joint_material_envmap_sh/').mkdir(parents=True, exist_ok=True)
for t in range(200):
    print('iteration:', t)
    optimizer.zero_grad()
    # Repeat the envmap generation & material for the gradients
    deringed_coeffs = deringing(coeffs, 6.0)
    envmap = pyredner.SH_reconstruct(deringed_coeffs, res)
    if t > 0 and t % (10 ** int(math.log10(t))) == 0:
        pyredner.imwrite(envmap.cpu(), f'{save_dir}/envmap_{t}.exr')
        pyredner.imwrite(envmap.cpu(), f'{save_dir}/envmap_{t}.png')
    envmap = pyredner.EnvironmentMap(envmap)
    # diffuse_reflectance = diffuse_reflectance_param.abs()
    # specular_reflectance = specular_reflectance_param.abs()
    # roughness = roughness_param.abs() # avoid going below zero
    # materials[0] = pyredner.Material(\
    #     diffuse_reflectance = diffuse_reflectance,
    #     specular_reflectance = specular_reflectance,
    #     roughness = roughness)
    scene = pyredner.Scene(camera = camera,
                           shapes = shapes,
                           materials = materials,
                           envmap = envmap)
    scene_args = pyredner.RenderFunction.serialize_scene(
                    scene = scene,
                    num_samples = 4,
                    max_bounces = 1, 
                    )
    img = render(t+1, *scene_args)
    loss = torch.pow(img*mask.unsqueeze(-1) - target*mask.unsqueeze(-1), 2).sum()
    print('loss:', f"{loss.item():.04f}")

    loss.backward()
    optimizer.step()

    if t > 0 and t % (10 ** int(math.log10(t))) == 0:
        pyredner.imwrite(img.cpu(), f'{save_dir}/iter_{t}.png')
    # Print the gradients of the coefficients, material parameters
    # print('coeffs.grad:', coeffs.grad)
    # Print the current parameters
    # print('coeffs:', coeffs)

# %%
materials = [
    pyredner.Material(
        diffuse_reflectance = mano_layer.tex_diffuse_mean.to(pyredner.get_device()),
        specular_reflectance = mano_layer.tex_spec_mean.to(pyredner.get_device()),
    ),
    objects[0].material,
    ]

vertex_normals = calc_vertex_normals(mano.vertices[0], mano_layer.faces)

mano_shape = pyredner.Shape(\
    vertices = mano.vertices[0], 
    indices = mano_layer.faces.to(torch.int32), 
    uvs = torch.tensor(uvs, dtype=torch.float32),
    uv_indices = torch.tensor(mano_layer.face_uvs, dtype=torch.int32),
    # normal_indices=mano_layer.faces.to(torch.int32),
    normals = torch.tensor(vertex_normals, dtype=torch.float32),
    material_id = 0)

# objects = pyredner.load_obj('data/models/021_bleach_cleanser/textured_simple.obj', return_objects=True)
obj_shape = pyredner.Shape(
    vertices=apply_transform_to_mesh(objects[0].vertices, anno),
    indices=objects[0].indices, 
    uvs=objects[0].uvs,
    uv_indices=objects[0].uv_indices,
    material_id=1
)
shapes = [mano_shape, obj_shape] 
scene = pyredner.Scene(camera = camera,
                        shapes = shapes,
                        materials = materials,
                        envmap = envmap)
scene_args = pyredner.RenderFunction.serialize_scene(\
    scene = scene,
    num_samples = 512,
    max_bounces = 1)
img = render(202, *scene_args)
pyredner.imwrite(img.cpu(), f'{save_dir}/final.exr')
pyredner.imwrite(img.cpu(), f'{save_dir}/final.png')
# pyredner.imwrite(torch.abs(target - img).cpu(), f'{save_dir}/final_diff.png')

# %
deferred_img = pyredner.render_deferred(scene, lights=[dirlight], alpha=True)
mask_ho = deferred_img[..., -1]
img_alpha = torch.cat([img, mask_ho.unsqueeze(-1)], -1)
# pyredner.imwrite(img_alpha.cpu(), f'{save_dir}/final_alpha.png')

fig, axs = plt.subplots(1,2, figsize=(10, 5))
axs[0].imshow(bg_img)
axs[1].imshow(bg_img)
axs[1].imshow(torch.pow(img_alpha.detach().cpu(), 1/2.2))
plt.show()
# %%
