# %%
import os
os.chdir(os.path.join(os.path.dirname(__file__), '..'))
print(os.getcwd())

from utils.mano import ManoLayer
from utils.utils import load_ho_meta
from utils.vis import PlotlyVisualizer

mano_layer = ManoLayer()
anno = load_ho_meta()
mano = mano_layer(anno)

fig = PlotlyVisualizer()
fig.add_mesh(mano.vertices[0], mano_layer.faces)
fig.show()

# %%
import trimesh
from trimesh.visual import texture, TextureVisuals
from PIL import Image

img = Image.open(f"data/NIMBLE/rand_0_diffuse.png")

mesh = trimesh.Trimesh(
    vertices=mano.vertices[0],
    faces=mano_layer.faces,
    visual=trimesh.visual.texture.TextureVisuals(
        uv=mano_layer.uv,
        image=img,
        material = texture.SimpleMaterial(image=img)
    ),
    process=False,
    validate=True,
    smooth=True,
)
# light = trimesh.scene.lighting.AmbientLight(color=(1, 1, 1))
# mesh.show()

# %%
import kaolin
import torch

print(anno.keys())
print(anno['objName'], anno['objLabel'])
obj_fp = f"data/models/{anno['objName']}/textured_simple.obj"
mesh = kaolin.io.obj.import_mesh(
    obj_fp, 
    with_materials=True, with_normals=False, )
mesh.allow_auto_compute = True

# print(mesh)
# print(mesh.materials[0].keys())
# print(mesh.materials[0]['Kd'])

#%%
# objRot = torch.tensor(anno['objRot'])
# objTrans = torch.tensor(anno['objTrans'])
print(objRot.shape, objTrans.shape)
camMat = anno['camMat']
fx, fy = camMat.diagonal()[:2]
cx, cy = camMat[:2, 2]
w2cs = [torch.diag(torch.tensor([-1.,1.,-1.,1.]))]

params = [torch.tensor([0,0,fx,fy], dtype=torch.float32)]
intrinsics = kaolin.render.camera.PinholeIntrinsics(
    params=torch.stack(params, dim=0),
    width=640,
    height=480,
)
extrinsics = kaolin.render.camera.CameraExtrinsics.from_view_matrix(
    torch.stack(w2cs, dim=0),
    # device='cuda'
)
cam = kaolin.render.camera.Camera(
    extrinsics=extrinsics, intrinsics=intrinsics)

from utils.utils import apply_transform_to_mesh
# import imageio
import numpy as np

# verts = apply_transform_to_mesh(mesh.vertices, anno)
verts = mano.vertices[0]
faces = mano_layer.faces.to(torch.int64)
uvs = mano_layer.uv.to(torch.float32)

mesh = kaolin.rep.SurfaceMesh(
    vertices=verts, faces=faces, uvs=uvs, face_uvs_idx=faces
)
print(mesh)
# print(mesh.face_uvs)

texture = torch.tensor(
    np.array(
        Image.open(f"data/NIMBLE/rand_0_diffuse.png")
        ).astype(np.float32) / 255.0
    )
print(texture.shape, texture.max())

# %%

print(verts.dtype, faces.dtype)
vertices_camera = cam.extrinsics.transform(mesh.vertices)
vertices_ndc = cam.intrinsics.transform(vertices_camera)

face_vertices_camera = kaolin.ops.mesh.index_vertices_by_faces(vertices_camera, mesh.faces)
face_vertices_image = kaolin.ops.mesh.index_vertices_by_faces(vertices_ndc[..., :2], mesh.faces)
face_vertices_z = face_vertices_camera[..., -1]

im_features, face_idx = kaolin.render.mesh.rasterize(
    480, 640, 
    face_vertices_z, face_vertices_image[..., :2],
    [mesh.face_uvs, mesh.face_normals],
    backend=''

)
hard_mask = face_idx != -1
hard_mask = hard_mask
uv_map = im_features[0]
im_world_normal = im_features[1]
albedo = kaolin.render.mesh.texture_mapping(uv_map, texture)
albedo = torch.clamp(albedo * hard_mask.unsqueeze(-1), min=0., max=1.)

# if __name__ == '__main__':
#     pass

# %%
