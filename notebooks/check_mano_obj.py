# %%
import openmesh
# import torch
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.collections import PatchCollection
import numpy as np
import trimesh

import os
os.chdir(os.path.join(os.path.dirname(__file__), '..'))


def load_obj(obj_file):
    with open(obj_file, 'r') as fp:
        verts = []
        faces = []
        vts = []
        vns = []
        faces_vts = []
        faces_vns = []

        for line in fp:
            line = line.rstrip()
            line_splits = line.split()
            prefix = line_splits[0]

            if prefix == 'v':
                verts.append(np.array([line_splits[1], line_splits[2], line_splits[3]], dtype=np.float32))

            elif prefix == 'vn':
                vns.append(np.array([line_splits[1], line_splits[2], line_splits[3]], dtype=np.float32))

            elif prefix == 'vt':
                vts.append(np.array([line_splits[1], line_splits[2]], dtype=np.float32))

            elif prefix == 'f':
                f = []
                f_vt = []
                f_vn = []
                for p_str in line_splits[1:4]:
                    p_split = p_str.split('/')
                    f.append(p_split[0])
                    f_vt.append(p_split[1])
                    f_vn.append(p_split[2])

                faces.append(np.array(f, dtype=np.int32) - 1)
                faces_vts.append(np.array(f_vt, dtype=np.int32) - 1)
                faces_vns.append(np.array(f_vn, dtype=np.int32) - 1)
            elif prefix == '#':
                continue
            else:
                print(prefix)
                raise ValueError(prefix)

        obj_dict = {
            'vertices': np.array(verts, dtype=np.float32),
            'faces': np.array(faces, dtype=np.int32),
            'vts': np.array(vts, dtype=np.float32),
            'vns': np.array(vns, dtype=np.float32),
            'faces_vts': np.array(faces_vts, dtype=np.int32),
            'faces_vns': np.array(faces_vns, dtype=np.int32)
        }

        return obj_dict


def load_nimble(nimble_path="./data/NIMBLE"):
    from pathlib import Path
    import pickle
    
    pklfiles = Path(nimble_path).glob("*.pkl")
    nimble_data = {}
    for fp in pklfiles:
        with open(fp, "rb") as f:
            data = pickle.load(f, encoding="latin1")
        nimble_data[fp.stem] = data
    
    # f v1/vt1/vn1
    fvs, fvts, vts, vs = [], [], [], []
    # for line in nimble_data["NIMBLE_TEX_FUV"]:
    # for line in nimble_data["NIMBLE_TEX_FUV"]:
    with open("./data/NIMBLE/NIMBLE_skin.obj", "r") as f:
        NIMBLE_skin = f.readlines()
    for line in NIMBLE_skin:
        if len(line.split()) == 0:
            continue
        if line.split()[0] == "f":
            for vvv in line.split()[1:]:
                fv, fvt, _ = vvv.split("/")
                fvs.append(int(fv)-1)
                fvts.append(int(fvt)-1)
        elif line.split()[0] == "vt":
            vts.append(list(map(float, line.split()[1:])))
        elif line.split()[0] == "v":
            vs.append(list(map(float, line.split()[1:])))

    # fvs = np.array(fvs).reshape(-1, 3)
    face_uvs = np.array(fvts).reshape(-1, 3)
    uvs = np.array(vts).reshape(-1, 2)
    skin_v_sep = nimble_data['NIMBLE_DICT_9137']['skin_v_sep']
    verts = nimble_data['NIMBLE_DICT_9137']['vert'][skin_v_sep:].cpu().numpy()
    faces = nimble_data['NIMBLE_DICT_9137']['skin_f'].cpu().numpy()
    print(faces)
    # verts = np.array(vs).reshape(-1, 3)
    # faces = fvs

    lmk_faces_idx = nimble_data['NIMBLE_MANO_VREG']['lmk_faces_idx']
    lmk_bary_coords = nimble_data['NIMBLE_MANO_VREG']['lmk_bary_coords']

    tex_diff_mean  = nimble_data['NIMBLE_TEX_DICT']['diffuse']['mean'].reshape(1024,1024,3)[..., [2,1,0]]

    # print(verts.shape, faces.shape, fvs.shape, face_uvs.shape, uvs.shape, lmk_faces_idx.shape, lmk_bary_coords.shape, tex_diff_mean.shape)
    return verts, faces, fvs, face_uvs, uvs, lmk_faces_idx, lmk_bary_coords, tex_diff_mean


# method = 'openmesh'
# method = 'hoig'

def add_triangles2D(ax, uv, faces, **kwargs):
    patch_list = [
        patches.Polygon(f) for f in uv[faces]
    ]
    colors = plt.cm.viridis(np.linspace(0, 1, len(patch_list)))
    pc = PatchCollection(patch_list, alpha=1, edgecolor='None', **kwargs)
    ax.add_collection(pc)


for method in ['openmesh', 'trimesh', 'trimesh_maintain_order', 'hoig']:
    if method == 'openmesh':
        mesh = openmesh.read_trimesh('./data/mano_v1_2/MANO_UV_right.obj', vertex_tex_coord=True)
        uv = mesh.vertex_texcoords2D()
        faces = mesh.face_vertex_indices()
    elif method == 'trimesh':
        mesh = trimesh.load_mesh(
            './data/mano_v1_2/MANO_UV_right.obj', 
            process=False, maintain_order=False)
        uv = mesh.visual.uv
        faces = mesh.faces
    elif method == 'trimesh_maintain_order':
        mesh = trimesh.load_mesh(
            './data/mano_v1_2/MANO_UV_right.obj', 
            process=False, maintain_order=True)
        uv = mesh.visual.uv
        faces = mesh.faces
    elif method == 'hoig':
        mesh = load_obj('./data/mano_v1_2/MANO_UV_right.obj', )
        uv = mesh['vts']
        faces = mesh['faces_vts']
    print(method, uv.shape, faces.shape)

    fig, ax = plt.subplots()
    ax.set_title(method + f'{str(uv.shape)} {str(faces.shape)}')
    add_triangles2D(ax, uv, faces,)

# %%
import matplotlib.colors as mcolors
# colors = mcolors.CSS4_COLORS
colors = mcolors.XKCD_COLORS
print("num colors:", len(colors))
# print(colors[0])
print(colors.keys())
print(colors.values())

method = 'nimble'
verts, faces, fvs, face_uvs, uv, lmk_faces_idx, lmk_bary_coords, tex_diff_mean = load_nimble()
print(verts.shape, uv.shape)
# verts = verts.cpu().numpy()
with open('./data/NIMBLE/NIMBLE_SKIN_TEX.obj', 'w') as f:
    for v in verts:
        f.write(f'v {v[0]} {v[1]} {v[2]}\n')
    for vt in uv:
        f.write(f'vt {vt[0]} {vt[1]}\n')
    for fv, fvt in zip(faces, face_uvs):
        f.write(f'f {fv[0]+1}/{fvt[0]+1} {fv[1]+1}/{fvt[1]+1} {fv[2]+1}/{fvt[2]+1}\n')
    # for fv in faces:
    #     f.write(f'f {fv[0]+1} {fv[1]+1} {fv[2]+1}\n')



face_vts = (np.stack([uv[face_uvs, 0], 1-uv[face_uvs, 1]], -1) * 1024).astype(np.int)
face_colors = tex_diff_mean[face_vts[:, :, 1], face_vts[:, :, 0], :]
face_colors_mean = face_colors.mean(axis=1)
# face_colors_mean = (face_colors_mean * 255).astype(np.uint8)
print(face_colors.shape, face_colors_mean.shape)

plt.imshow(tex_diff_mean)
plt.scatter(face_vts[:, :, 0], face_vts[:, :, 1], s=0.01)
plt.show()

import plotly.graph_objects as go
from matplotlib.colors import rgb2hex
import random

face_colors_hex = np.array([rgb2hex(c) for c in face_colors_mean])
# print(face_colors_hex)

lmk_face_colors = np.array(['#ffffff' for _ in range(len(faces))])
for i, lmk_face in enumerate(lmk_faces_idx.T):
    lmk_face = lmk_face.numpy().astype(int) - 1
    lmk_face_colors[lmk_face[0]] = random.choice(list(colors.values()))


fig = go.Figure()
fig.add_trace(go.Mesh3d(x=verts[:,0], y=verts[:,1], z=verts[:,2], i=faces[:,0], j=faces[:,1], k=faces[:,2], 
    colorscale='Viridis', flatshading=True, name='mano',
    facecolor=lmk_face_colors))
fig.show()
# plt.imshow(tex_diff_mean)

# %%

from PIL import Image
# print(tex_diff_mean)
img = Image.fromarray((tex_diff_mean*255).astype(np.uint8))
# plt.imshow(img)
material = trimesh.visual.texture.SimpleMaterial(image=img)
color_visuals = trimesh.visual.texture.TextureVisuals(uv=uv, image=img, material=material, )#face_materials=face_uvs)
mesh=trimesh.Trimesh(vertices=verts, faces=faces, visual=color_visuals, validate=False, process=True, smooth=True)
# mesh=trimesh.Trimesh(vertices=verts, faces=faces, validate=True, process=False)
# mesh.show()
scene = trimesh.Scene([mesh])
# scene.show()

print(faces.shape, face_uvs.shape, faces.shape)
fig, ax = plt.subplots()
ax.set_title( f'{method}: {str(uv.shape)} {str(face_uvs.shape)}')
for i, lmk_face in enumerate(lmk_faces_idx.T):
    lmk_triangles = face_uvs[lmk_face]
    lmk_uvs = uv[lmk_triangles]
    # print(lmk_uvs.shape)
    lmk_uv_mean = lmk_uvs.mean(axis=1)
    ax.plot(
        lmk_uv_mean[:, 0], lmk_uv_mean[:, 1], 
        c=list(colors.values())[i%len(colors)], linewidth=0.1
        )


#     # add_triangles2D(ax, uv, face_uvs[lmk_face], facecolors=list(colors.values())[i%len(colors)])
# print(lmk_faces_idx.shape, lmk_bary_coords.shape)
# # print(lmk_faces_idx, lmk_bary_coords)
# %%
