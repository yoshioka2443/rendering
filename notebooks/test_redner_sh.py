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


# %%
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

uvs = torch.stack([mano_layer.uv[..., 0], 1-mano_layer.uv[..., 1]], -1)

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

objects = pyredner.load_obj(f'data/models/{anno["objName"]}/textured_simple.obj', return_objects=True)

# sRGB to Linear RGB
# object_diffuse_texels = objects[0].material.diffuse_reflectance._texels
# objects[0].material.diffuse_reflectance._texels = torch.pow(object_diffuse_texels, 2.2)

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

coeffs_sh = torch.zeros((3, 16), device=pyredner.get_device())
coeffs_sh[:, 0] += 1.0
# coeffs_sh = torch.tensor([
#     [ 8.7388e-01,  7.1441e-01,  8.0384e-01,  2.6932e-01, -3.2511e-03,
#         1.1062e-02,  1.5726e+00,  9.0622e-01,  7.1904e-01, -3.0765e-01,
#         -2.4147e-01,  2.4304e+00,  2.4662e+00,  3.3280e-01,  7.7696e-01,
#         -5.5327e-01],
#     [ 8.1642e-01,  6.3501e-01,  9.1502e-01,  2.7539e-01,  9.8240e-02,
#         4.7828e-01,  1.3868e+00,  9.9202e-01,  8.0545e-01, -3.1640e-01,
#         -3.1013e-01,  1.8274e+00,  2.1913e+00, -1.4797e-01,  3.2672e-01,
#         -6.1430e-01],
#     [ 7.7144e-01,  5.9789e-01,  1.0238e+00,  2.1427e-01,  1.0612e-01,
#         7.8751e-01,  1.4998e+00,  1.1051e+00,  8.9485e-01, -2.0760e-01,
#         -9.5950e-03,  1.8183e+00,  2.2059e+00, -1.9734e-03,  3.5333e-01,
#         -6.3638e-01]], device=pyredner.get_device())
res = (128, 128)
deringed_coeffs_sh = deringing(coeffs_sh, 6.0)
envmap = pyredner.SH_reconstruct(deringed_coeffs_sh, res)

# create scene
scene = pyredner.Scene(
    camera = camera, 
    objects = [
        mano_redner, 
        obj_redner,
        ],
    envmap = pyredner.EnvironmentMap(envmap)
    )

scene_mano = pyredner.Scene(
    camera = camera, 
    objects = [
        mano_redner, 
        ]
    )

# Render the scene.
# render_deferred = pyredner.render_deferred(scene, lights=[dirlight], alpha=True)
scene_args = pyredner.RenderFunction.serialize_scene(
    scene = scene,
    num_samples = 512,
    max_bounces = 1,
    sampler_type=pyredner.sampler_type.independent,
    # sampler_type = pyredner.sampler_type.sobol,
    use_primary_edge_sampling=True,
    use_secondary_edge_sampling=True,
    channels = [
        pyredner.channels.radiance, 
        # pyredner.channels.alpha, 
        pyredner.channels.geometry_normal,
        pyredner.channels.shading_normal,
        # pyredner.channels.diffuse_reflectance,
        ]
    )
render = pyredner.RenderFunction.apply
render_buffer = render(202, *scene_args)
# Render the scene.
render_albedo = pyredner.render_albedo(scene, alpha=True)
render_albedo_mano = pyredner.render_albedo(scene_mano, alpha=True)

mask = render_albedo[..., -1]
mask_mano = render_albedo_mano[..., -1]

show_dict = {
    'input': bg_img,
    # 'albedo(mano+obj)': torch.pow(render_albedo, 1.0/2.2).cpu(),
    # 'albedo(mano)': torch.pow(render_albedo_mano, 1.0/2.2).cpu(),
    'radiance': torch.pow(render_buffer[..., :3].cpu().detach(), 1/2.2),
    'geometry_normal': render_buffer[..., 3:6].cpu().detach(),
    'shading_normal': render_buffer[..., 6:9].cpu().detach(),
}

fig, axs = plt.subplots(1, len(show_dict), figsize=(5*len(show_dict), 5))
for ax, (k, v) in zip(axs, show_dict.items()):
    # ax.imshow(bg_img)
    ax.axis('off')
    ax.set_title(k)
    ax.imshow(v)

plt.show()



# Reset the coefficients to some constant color, repeat the same process as in target envmap
# coeffs = torch.tensor([[ 0.5,
#                          0.0, 0.0, 0.0,
#                          0.0, 0.0, 0.0, 0.0, 0.0,
#                          0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], # coeffs for red
#                        [ 0.5,
#                          0.0, 0.0, 0.0,
#                          0.0, 0.0, 0.0, 0.0, 0.0,
#                          0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], # coeffs for green
#                        [ 0.5,
#                          0.0, 0.0, 0.0,
#                          0.0, 0.0, 0.0, 0.0, 0.0,
#                          0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], # coeffs for blue
#                        device = pyredner.get_device(),
#                        requires_grad = True)
coeffs_sh = torch.zeros((3, 16), device=pyredner.get_device())
coeffs_sh[:, 0] += 0.5
coeffs_sh.requires_grad = True
res = (128, 128)

materials = [
    pyredner.Material(
        diffuse_reflectance = mano_layer.tex_diffuse_mean.to(pyredner.get_device()),
        specular_reflectance = mano_layer.tex_spec_mean.to(pyredner.get_device()),
        normal_map = mano_layer.tex_normal_mean.to(pyredner.get_device()),
    ),
    objects[0].material,
]

mano_shape = pyredner.Shape(\
    vertices = mano.vertices[0], 
    indices = mano_layer.faces.to(torch.int32), 
    uvs = torch.tensor(uvs, dtype=torch.float32),
    uv_indices = torch.tensor(mano_layer.face_uvs, dtype=torch.int32),
    normals = pyredner.compute_vertex_normal(mano.vertices[0], mano_layer.faces),
    material_id = 0)

obj_shape = pyredner.Shape(
    vertices=apply_transform_to_mesh(objects[0].vertices, anno),
    indices=objects[0].indices, 
    uvs=objects[0].uvs,
    uv_indices=objects[0].uv_indices,
    material_id=1
)
shapes = [mano_shape, obj_shape]

target = torch.pow(torch.tensor(bg_img, device=pyredner.get_device()), 2.2)

coeffs_tex = torch.zeros(10, device=pyredner.get_device(), requires_grad=True)

# Finally we can start the Adam iteration
optimizer = torch.optim.Adam([
    coeffs_sh, 
    coeffs_tex
    ], 
    lr=3e-2)
# optimizer = torch.optim.SGD([
#     coeffs_sh, 
#     coeffs_tex
#     ], 
#     lr=1e-4)

save_dir = "results/joint_material_envmap_fixnormal"
# Path('results/joint_material_envmap_sh/').mkdir(parents=True, exist_ok=True)
loss_log = []


lambda_reg_tex = 1e2
lambda_reg_sh = 1e2

pyredner.set_print_timing(False)
# pp = ProgressPlot()
pp = ProgressPlot(line_names=['loss', 'loss_mse', 'reg_tex', 'reg_sh'], plot_names=['loss'])
# pp_coeffs = ProgressPlot(line_names=[f'coeff{i}' for i in range(coeffs_tex.shape[0])])
for t in range(100):
    # print('iteration:', t)
    optimizer.zero_grad()
    # Repeat the envmap generation & material for the gradients
    deringed_coeffs_sh = deringing(coeffs_sh, 6.0)
    envmap = pyredner.SH_reconstruct(deringed_coeffs_sh, res)
    if t > 0 and t % (10 ** int(math.log10(t))) == 0:
        pyredner.imwrite(envmap.cpu(), f'{save_dir}/envmap_{t}.exr')
        pyredner.imwrite(envmap.cpu(), f'{save_dir}/envmap_{t}.png')
    envmap = pyredner.EnvironmentMap(envmap)
    diffuse_reflectance = torch.sum(coeffs_tex * mano_layer.tex_diffuse_basis.to(pyredner.get_device()), dim=-1) + mano_layer.tex_diffuse_mean.to(pyredner.get_device())
    specular_reflectance = torch.sum(coeffs_tex * mano_layer.tex_spec_basis.to(pyredner.get_device()), dim=-1) + mano_layer.tex_spec_mean.to(pyredner.get_device())
    normal_map = torch.sum(coeffs_tex * mano_layer.tex_normal_basis.to(pyredner.get_device()), dim=-1) + mano_layer.tex_normal_mean.to(pyredner.get_device())
    materials[0] = pyredner.Material(
        diffuse_reflectance = diffuse_reflectance,
        specular_reflectance = specular_reflectance,
        # normal_map = normal_map,
    )
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
    # loss_mse = torch.pow((img - target) * mask_mano.unsqueeze(-1), 2).sum()
    loss_mse = torch.pow((img - target) * mask.unsqueeze(-1), 2).sum()
    reg_tex = torch.pow(coeffs_tex, 2).sum() * lambda_reg_tex
    reg_sh = torch.pow(coeffs_sh[:, 1:], 2).sum() * lambda_reg_sh
    loss = loss_mse + reg_tex + reg_sh
    # print(f"loss_total: {loss.item():.04f}, loss_mse: {loss_mse.item():.04f}, reg_tex: {reg_tex.item():.04f}")

    loss.backward()
    optimizer.step()

    loss_log.append(loss.item())
    
    pp.update({
        'loss': {
            'loss': loss.item(),
            'loss_mse': loss_mse.item(),
            'reg_tex': reg_tex.item(),
            'reg_sh': reg_sh.item(),
        },
    })

    if t > 0 and t % (10 ** int(math.log10(t))) == 0:
        pyredner.imwrite(img.cpu(), f'{save_dir}/iter_{t}.png')

pp.finalize()

# %%
with torch.no_grad():
    print('coeffs_sh:', coeffs_sh)
    print('coeffs_tex:', coeffs_tex)
    diffuse_reflectance = torch.sum(coeffs_tex * mano_layer.tex_diffuse_basis.to(pyredner.get_device()), dim=-1) + mano_layer.tex_diffuse_mean.to(pyredner.get_device())
    specular_reflectance = torch.sum(coeffs_tex * mano_layer.tex_spec_basis.to(pyredner.get_device()), dim=-1) + mano_layer.tex_spec_mean.to(pyredner.get_device())    
    normal_map = torch.sum(coeffs_tex * mano_layer.tex_normal_basis.to(pyredner.get_device()), dim=-1) + mano_layer.tex_normal_mean.to(pyredner.get_device())    
    materials = [
        pyredner.Material(
            diffuse_reflectance = diffuse_reflectance,
            specular_reflectance = specular_reflectance,
            # normal_map = normal_map
        ),
        objects[0].material,
        ]

    vertex_normals = pyredner.compute_vertex_normal(mano.vertices[0], mano_layer.faces)

    mano_vertices = mano.vertices[0]
    mano_indices = mano_layer.faces.to(torch.int32)

    mano_shape = pyredner.Shape(
        vertices = mano_vertices, 
        indices = mano_indices, 
        uvs = torch.tensor(uvs, dtype=torch.float32),
        uv_indices = torch.tensor(mano_layer.face_uvs, dtype=torch.int32),
        normals = vertex_normals, 
        material_id = 0
        )

    obj_shape = pyredner.Shape(
        vertices=apply_transform_to_mesh(objects[0].vertices, anno),
        indices=objects[0].indices, 
        uvs=objects[0].uvs,
        uv_indices=objects[0].uv_indices,
        material_id=1,
    )
    shapes = [mano_shape, obj_shape] 
    scene = pyredner.Scene(
                camera = camera,
                shapes = shapes,
                materials = materials,
                envmap = envmap)
    scene_args = pyredner.RenderFunction.serialize_scene(
        scene = scene,
        num_samples = 512,
        max_bounces = 1,
        channels = [
            pyredner.channels.radiance, 
            pyredner.channels.alpha, 
            pyredner.channels.shading_normal,
            pyredner.channels.diffuse_reflectance,
            pyredner.channels.geometry_normal,
            ]
        )
    render_buffer = render(202, *scene_args)
    # img = pyredner.render_pathtracing(scene)
    img = render_buffer[..., :3]
    img_alpha = render_buffer[..., :4]
    shading_normal = render_buffer[..., 4:7]
    diffuse = render_buffer[..., 7:10]
    geometry_normal = render_buffer[..., 10:13]

    # pyredner.imwrite(img.cpu(), f'{save_dir}/final.exr')
    # pyredner.imwrite(img.cpu(), f'{save_dir}/final.png')
    # pyredner.imwrite(torch.abs(target - img).cpu(), f'{save_dir}/final_diff.png')

    # %

    render_albedo_ = pyredner.render_albedo(scene, alpha=True)

    show_dict = {
        'input': bg_img,
        # 'albedo': torch.pow(render_albedo, 1.0/2.2).cpu(),
        'optimized': torch.pow(img_alpha[..., :3].detach().cpu(), 1.0/2.2),
        'overlay': torch.pow(img_alpha.detach().cpu(), 1.0/2.2),
        # 'geometry_normal': geometry_normal.cpu(),
        # 'shading_normal': shading_normal.cpu(),
        # 'diffuse': diffuse.cpu(),
        
        # 'albedo': torch.pow(render_albedo_.detach().cpu(), 1.0/2.2),
    }

    fig, axs = plt.subplots(1, len(show_dict), figsize=(5*len(show_dict), 5))
    for ax, (k, v) in zip(axs, show_dict.items()):
        ax.imshow(bg_img)
        ax.set_title(k)
        ax.imshow(v)
        ax.axis('off')
    plt.show()

    show_dict = dict(
        mean = torch.pow(mano_layer.tex_diffuse_mean.cpu(), 1/2.2),
        optimized = torch.pow(diffuse_reflectance.cpu(), 1/2.2),
    )

    # fig, axs = plt.subplots(1, len(show_dict), figsize=(5*len(show_dict), 5))
    # for ax, (k, v) in zip(axs, show_dict.items()):
    #     ax.imshow(v)
    #     ax.set_title(k)
    #     ax.axis('off')
    # plt.show()

# %%
# plt.imsave('mask.png', img_alpha.detach().cpu().numpy()[..., -1], cmap='gray')
plt.imsave('optimized_normalindices.png', torch.pow(img_alpha.detach().cpu(), 1.0/2.2).numpy())
plt.imsave('shading_normal.png', shading_normal.cpu().numpy() * 0.5 + 0.5)
plt.imsave('geometry_normal.png', geometry_normal.cpu().numpy() * 0.5 + 0.5)

# %%
print('coeffs_sh:', coeffs_sh)
print('coeffs_tex:', coeffs_tex)
# %%
