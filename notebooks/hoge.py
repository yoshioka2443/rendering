# %%
import trimesh
import torch

mesh = trimesh.load_mesh('../data/mano_v1_2/MANO_UV_right.obj')
uv = torch.tensor(mesh.visual.uv, dtype=torch.float32)

for k, v in mesh.visual.__dict__.items():
    print(k, v)
# print(uv)

for k in dir(mesh.visual):
    print(k)

print(mesh.visual.face_materials)
# %%
