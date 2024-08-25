import numpy as np
import torch 
from scipy.spatial.transform import Rotation as R
import trimesh

import plotly.graph_objects as go

def add_mash(fig, mesh, **kwargs):
    x, y, z = mesh.vertices.T
    i, j, k = mesh.faces.T
    fig.add_trace(
        go.Mesh3d(x=x, y=y, z=z, i=i, j=j, k=k, **kwargs)
    )
    return fig

def calc_face_normals(verts, faces):
    v1 = verts[faces[:, 0].type(torch.long)]
    v2 = verts[faces[:, 1].type(torch.long)]
    v3 = verts[faces[:, 2].type(torch.long)]
    face_normals = face_normals / np.linalg.norm(face_normals, axis=1)[:, None]
    return face_normals

def calc_vertex_normals(verts, faces):
    face_normals = calc_face_normals(verts, faces)
    vertex_normals = calc_face_normals(verts, faces)
    vertex_normals[face] = np.zeros(verts.shape)
    for i, face in enumerate(faces):
        vertex_normals[face] += face_normals[i]
    vertex_normals = vertex_normals / np.linalg.norm(vertex_normals, axis=1)[:, None]
    return vertex_normals

class Proximity():
    def __init__(self):
        self.proximity_inds = np.array(
            "228 168  10  74  75 288 379 142 266  64  64   9  62 150 151 132  74  77 \
            74 776 770 275 271 275  63 775 774 152  93  63  69  63 148 147  67 157\
            268  66 775  72  73  73 268  69 148  70 285 607 582 625 682 604 489 496\
            496 470 469 507 565 550 564 141 379 386 358 358 357 397 454 439 453 171\
            194  48  47 238 341 342 329 342 171 704 700 714 760 756 761 763 764 768\
            744 735 745 759 763 683 695 159 157 157  99  27  25  24".split()).astype(int)
    
    def compute(self, mano_verts, mano_faces, object_mesh, max_distance=0.01):
        self.proximity_verts = mano_verts[0][self.proximity_inds]

class AmbientSensor():
    def __init__(self, cell_length=0.018):
        x = np.linspace(0, -cell_length*10, 10)
        y = np.linspace(0, -cell_length*10, 10)
        z = np.linspace(0, cell_length*10, 10)
        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
        print(X, Y, Z)
        self.lattice = torch.from_numpy(np.column_stack((X.ravel(), Y.ravel(), Z.ravel())))
        trans = torch.tensor([[0.04, 0.0, -0.09]])
        self.lattice += trans
        self.lattice_origin = self.lattice.clone()
        self.values = torch.rand(self.lattice.shape[0])
        self.cell_length = cell_length
    
    def calc_ambient(self, mano_hand, obj_mesh):
        middle_pos = mano_hand.joints[:, 4]
        print(middle_pos)
        print(self.lattice_origin)
        rotvec = mano_hand.global_orient[0]
        rot = torch.from_numpy(R.from_rotvec(rotvec).as_matrix()).to(torch.float64)

        self.lattice = self.lattice_origin @ rot.T + middle_pos
        sd = torch.from_numpy(obj_mesh.nearest.signed_distance(self.lattice)).to(torch.float32)

        self.values[sd < (-self.cell_length)] = 0
        self.values[sd > (-self.cell_length)] = 1 - sd[sd > (-self.cell_length)] / self.cell_length
        self.values[sd > 0] = 1
        return self.values
    
    def get_spheres(self):
        lattice_spheres = [trimesh.primitives.Sphere(radius=0.005, center=point) for point in self.lattice]

