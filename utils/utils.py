import pickle
import numpy as np
import torch
from scipy.spatial.transform import Rotation

import pickle
import numpy as np
import torch
from scipy.spatial.transform import Rotation
import trimesh
import plotly.graph_objects as go



def load_ho_meta(
        meta_fp = "data/HO3D_v3/train/ABF10/meta/0000.pkl"
    ):
    with open(meta_fp, 'rb') as f:
        meta = pickle.load(f, encoding='latin1')

    anno = {}
    for k, v in meta.items():
        print(k, v)
        if v is None or v == 'None':
            anno[k] = None
        else:
            anno[k] = np.array(v)

    return anno


def get_obj_mesh(anno):
    obj_filename = f'data/models/{anno["objName"]}/textured_simple.obj'
    obj_trimesh = trimesh.load_mesh(obj_filename)

    obj_verts = torch.tensor(obj_trimesh.vertices, dtype=torch.float32)
    obj_faces = torch.tensor(obj_trimesh.faces, dtype=torch.int64)
    obj_uv = torch.tensor(obj_trimesh.visual.uv, dtype=torch.float32)
    obj_map = torch.from_numpy(np.array(obj_trimesh.visual.material.image, dtype=np.float32)) / 255.0

    return obj_verts, obj_faces, obj_uv, obj_map



def apply_transform_to_mesh(verts, anno):
    objRmat = torch.tensor(Rotation.from_rotvec(anno['objRot'][:, 0]).as_matrix(), dtype=torch.float32).to(verts.device)
    objTrans = torch.tensor(anno['objTrans']).to(verts.device)
    # verts = (objRmat @ verts.T + objTrans.T).T
    print(verts.shape, objRmat.shape, objTrans.shape)
    verts = verts @ objRmat.T + objTrans
    return verts

def calc_face_normals(verts, faces):
    v1 = verts[faces[:,0].type(torch.long)]
    v2 = verts[faces[:,1].type(torch.long)]
    v3 = verts[faces[:,2].type(torch.long)]
    return np.cross(v2-v1, v3-v1)

def calc_vertex_normals(verts, faces):
    face_normals = calc_face_normals(verts, faces)
    vertex_normals = np.zeros(verts.shape)
    for i, face in enumerate(faces):
        vertex_normals[face] += face_normals[i]
    vertex_normals = vertex_normals / np.linalg.norm(vertex_normals, axis=1)[:,None]
    return vertex_normals


def calc_proximity(mano_verts, mano, obj_mesh, max_distance=0.1):
    proximity_inds = np.array(
        "228 168  10  74  75 288 379 142 266  64  64   9  62 150 151 132  74  77 \
        74 776 770 275 271 275  63 775 774 152  93  63  69  63 148 147  67 157 \
        268  66 775  72  73  73 268  69 148  70 285 607 582 625 682 604 489 496 \
        496 470 469 507 565 550 564 141 379 386 358 358 357 397 454 439 453 171 \
        194  48  47 238 341 342 329 342 171 704 700 714 760 756 761 763 764 768 \
        744 735 745 759 763 683 695 159 157 157  99  27  25  24".split()).astype(int)
    
    proximity_verts = mano_verts[0][proximity_inds]
    proximity_normals = calc_vertex_normals(mano_verts[0], mano.faces)[proximity_inds]

    ray_origins = proximity_verts.numpy()
    ray_directions = proximity_normals

    locations, index_ray, index_tri = obj_mesh.ray.intersects_location(
        ray_origins=ray_origins, 
        ray_directions=ray_directions
    )

    ray_distances = np.linalg.norm(locations - ray_origins[index_ray], axis=1)

    proximity_d = proximity_normals.copy()
    proximity_d[index_ray] *= np.minimum(ray_distances[:, None], max_distance)
    index_out = np.setdiff1d(np.arange(len(proximity_verts)), index_ray)
    proximity_d[index_out] *= max_distance

    return proximity_d, proximity_inds, index_ray, index_out


def add_trimesh_to_fig(fig, mesh, **kwargs):
    # fig = go.Figure()
    x, y, z = mesh.vertices.T
    i, j, k = mesh.faces.T
    fig.add_trace(
        go.Mesh3d(x=x,y=y,z=z,i=i,j=j,k=k, **kwargs)
    )
    return fig

def add_mesh_to_fig(fig, verts, faces, **kwargs):
    # fig = go.Figure()
    x, y, z = verts.T
    i, j, k = faces.T
    fig.add_trace(
        go.Mesh3d(x=x,y=y,z=z,i=i,j=j,k=k, **kwargs)
    )
    return fig

def get_obj_mesh_print(anno):
    obj_filename = f'data/models/{anno["objName"]}/textured_simple.obj'
    obj_trimesh = trimesh.load_mesh(obj_filename)
    print(obj_trimesh)
    obj_verts = torch.tensor(obj_trimesh.vertices, dtype=torch.float32)
    obj_faces = torch.tensor(obj_trimesh.faces, dtype=torch.int64)
    obj_uv = torch.tensor(obj_trimesh.visual.uv, dtype=torch.float32)
    obj_map = torch.from_numpy(np.array(obj_trimesh.visual.material.image, dtype=np.float32)) / 255.0

    print(obj_verts.shape)
    print(obj_faces.shape)
    print(obj_uv.shape)
    print(obj_map.shape)
    return obj_verts, obj_faces, obj_uv, obj_map

# def load_ho_meta(
#         meta_fp = "data/HO3D_v3/train/ABF10/meta/0000.pkl"
#     ):
#     with open(meta_fp, 'rb') as f:
#         meta = pickle.load(f, encoding='latin1')

#     anno = {}
#     for k, v in meta.items():
#         if v is None or v == 'None':
#             anno[k] = None
#         else:
#             anno[k] = np.array(v)

#     return anno



# def apply_transform_to_mesh(verts, anno):
#     objRmat = torch.tensor(Rotation.from_rotvec(anno['objRot'][:, 0]).as_matrix(), dtype=torch.float32).to(verts.device)
#     objTrans = torch.tensor(anno['objTrans']).to(verts.device)
#     # verts = (objRmat @ verts.T + objTrans.T).T
#     print(verts.shape, objRmat.shape, objTrans.shape)
#     verts = verts @ objRmat.T + objTrans
#     return verts

