# %%
import smplx
import torch
import numpy as np
import trimesh
import pyredner
# from trimesh import load_mesh

class ManoLayer:
    def __init__(self):
        self.smplx_path = 'data/mano_v1_2/models'
        self.mano_layer = {
            'right': smplx.create(self.smplx_path, 'mano', use_pca=False, is_rhand=True, flat_hand_mean=True),
            'left': smplx.create(self.smplx_path, 'mano', use_pca=False, is_rhand=False, flat_hand_mean=True),
        }
        self.faces = torch.tensor(self.mano_layer['right'].faces.astype(np.int32), dtype=torch.int32)
        # self.mesh = trimesh.load_mesh('data/mano_v1_2/MANO_UV_right.obj')
        # self.uv = torch.tensor(self.mesh.visual.uv, dtype=torch.float32)
        material_map, mesh_list, light_map = pyredner.load_obj('data/mano_v1_2/MANO_UV_right.obj', return_objects=True)[0]
        self.uv = mesh_list.uvs
        self.face_uvs = mesh_list.uv_indices

        self.map = torch.ones([256, 256, 3], dtype=torch.float32) * torch.tensor([[[1.0, 0.5, 0.5]]])
        self.parents = self.mano_layer['right'].parents

        self.proximity_inds = np.array(
            "228 168  10  74  75 288 379 142 266  64  64   9  62 150 151 132  74  77 \
            74 776 770 275 271 275  63 775 774 152  93  63  69  63 148 147  67 157\
            268  66 775  72  73  73 268  69 148  70 285 607 582 625 682 604 489 496\
            496 470 469 507 565 550 564 141 379 386 358 358 357 397 454 439 453 171\
            194  48  47 238 341 342 329 342 171 704 700 714 760 756 761 763 764 768\
            744 735 745 759 763 683 695 159 157 157  99  27  25  24".split()).astype(int)

    def __call__(self, anno, is_rhand=True):
        '''
        anno: dict, annotation of the hand
        is_rhand: bool, True if right hand, False if left hand
        return: dict, vertices of the hand
                dict_keys(['vertices', 'joints', 'full_pose', 'global_orient', 'transl', 'v_shaped', 'betas', 'hand_pose'])
                vertices torch.Size([1, 778, 3])
                joints torch.Size([1, 16, 3])
                global_orient torch.Size([1, 3]) 
                betas torch.Size([1, 10])   shape parameters
                hand_pose torch.Size([1, 45])  pose parameters
        '''
        root_pose = torch.from_numpy(anno['handPose']).view(-1, 3)[0].view(1, 3).float()#.cuda()
        hand_pose = torch.from_numpy(anno['handPose']).view(-1, 3)[1:].view(1, -1).float()#.cuda()
        betas = torch.from_numpy(anno['handBeta']).view(1, -1).float()#.cuda()
        trans = torch.from_numpy(anno['handTrans']).view(1, 3).float()#.cuda()
        output = self.mano_layer['right' if is_rhand else 'left'](global_orient=root_pose, hand_pose=hand_pose, betas=betas, transl=trans, return_full_pose=True)
        return output#.vertices

    def load_nimble(self, nimble_path="data/NIMBLE"):
        from pathlib import Path
        import pickle
        
        pklfiles = Path(nimble_path).glob("*.pkl")
        self.nimble_data = {}
        for fp in pklfiles:
            with open(fp, "rb") as f:
                data = pickle.load(f, encoding="latin1")
            self.nimble_data[fp.stem] = data
        
        # f v1/vt1/vn1
        fvs, fvts, vts = [], [], []
        for line in self.nimble_data["NIMBLE_TEX_FUV"]:
            if len(line.split()) == 0:
                continue
            if line.split()[0] == "f":
                for vvv in line.split()[1:]:
                    fv, fvt, _ = vvv.split("/")
                    fvs.append(int(fv)-1)
                    fvts.append(int(fvt)-1)
            elif line.split()[0] == "vt":
                vts.append(list(map(float, line.split()[1:])))
        self.nimble_faces = torch.tensor(np.array(fvs).reshape(-1, 3), dtype=torch.long)
        self.nimble_face_uvs = torch.tensor(np.array(fvts).reshape(-1, 3), dtype=torch.long)
        self.nimble_uvs = torch.tensor(np.array(vts).reshape(-1, 2), dtype=torch.float32)
        # print(fvs.shape, fvts.shape, vts.shape)

        self.tex_diff_basis = torch.tensor(self.nimble_data['NIMBLE_TEX_DICT']['diffuse']['basis'])
        self.tex_diff_mean  = torch.tensor(self.nimble_data['NIMBLE_TEX_DICT']['diffuse']['mean']).reshape(1024,1024,3)[..., [2,1,0]]
        self.tex_diff_std   = torch.tensor(self.nimble_data['NIMBLE_TEX_DICT']['diffuse']['std'])
        # print(self.tex_diff_basis.shape, self.tex_diff_mean.shape, self.tex_diff_std.shape)

        self.lmk_faces_idx = self.nimble_data['NIMBLE_MANO_VREG']['lmk_faces_idx']
        self.lmk_bary_coords = self.nimble_data['NIMBLE_MANO_VREG']['lmk_bary_coords']

        # print('NIMBLE_DICT_9137', self.nimble_data['NIMBLE_DICT_9137'].keys())
        self.nimble_vert = torch.tensor(self.nimble_data['NIMBLE_DICT_9137']['vert'])

        nimble_uvw = torch.cat([self.nimble_uvs, torch.zeros_like(self.nimble_uvs[..., -1:])], dim=-1)
        self.nimble_mano_uv_bary = torch.cat([
            vertices2landmarks(
                nimble_uvw.unsqueeze(0), 
                self.nimble_face_uvs, 
                self.lmk_faces_idx[i], 
                self.lmk_bary_coords[i],
                )[..., :2] for i in range(20)], dim=0)
        self.nimble_mano_uvs = self.nimble_mano_uv_bary.mean(dim=0)

def vertices2landmarks(
    vertices,
    faces,
    lmk_faces_idx,
    lmk_bary_coords
):
    ''' 
        Calculates landmarks by barycentric interpolation
        Parameters
        ----------
        vertices: torch.tensor BxVx3, dtype = torch.float32
            The tensor of input vertices
        faces: torch.tensor Fx3, dtype = torch.long
            The faces of the mesh
        lmk_faces_idx: torch.tensor L, dtype = torch.long
            The tensor with the indices of the faces used to calculate the
            landmarks.
        lmk_bary_coords: torch.tensor Lx3, dtype = torch.float32
            The tensor of barycentric coordinates that are used to interpolate
            the landmarks
        Returns
        -------
        landmarks: torch.tensor BxLx3, dtype = torch.float32
            The coordinates of the landmarks for each mesh in the batch
        
        Modified from https://github.com/vchoutas/smplx
    '''
    # Extract the indices of the vertices for each face
    # BxLx3
    batch_size, num_verts = vertices.shape[:2]
    device = vertices.device
    
    lmk_faces = torch.index_select(
        faces, 0, lmk_faces_idx.view(-1)).view(1, -1, 3)
    lmk_faces = lmk_faces.repeat([batch_size,1,1])

    lmk_faces += torch.arange(
        batch_size, dtype=torch.long, device=device).view(-1, 1, 1) * num_verts

    lmk_vertices = vertices.reshape(-1, 3)[lmk_faces].view(
        batch_size, -1, 3, 3)

    landmarks = torch.einsum(
        'blfi,lf->bli', [lmk_vertices, lmk_bary_coords])
    return landmarks

if __name__ == '__main__':
    import os
    os.chdir(os.path.join(os.path.dirname(__file__), '..'))
    print(os.getcwd())
    from utils import load_ho_meta
    from PIL import Image
    import matplotlib.pyplot as plt

    anno = load_ho_meta()
    mano = ManoLayer()
    print(mano.mano_layer['right'].__dict__.keys())
    for k, v in mano(anno).__dict__.items():
        if v is not None:
            print(k, v.shape)
        # print(k, v)
    
    # mano = ManoLayer()
    mano.load_nimble()
    print(mano.nimble_faces.shape)

    print(mano.nimble_data["NIMBLE_MANO_VREG"].keys())
    lmk_faces_idx = mano.nimble_data['NIMBLE_MANO_VREG']['lmk_faces_idx']
    lmk_bary_coords = mano.nimble_data['NIMBLE_MANO_VREG']['lmk_bary_coords']
    # print(lmk_faces_idx.shape, lmk_bary_coods.shape)
    # print(lmk_faces_idx.max())
    # print(lmk_bary_coods)

    import plotly.graph_objects as go

    # fig = go.Figure()
    # fig.add_trace(
    #     go.Mesh3d(
    #         x=lmk_bary_coords[0, :, 0],
    #         y=lmk_bary_coords[0, :, 1],
    #         z=lmk_bary_coords[0, :, 2],
    #     )
    # )
    print(f"vert {mano.nimble_vert.shape}")
    print(f"uv {mano.nimble_uvs.shape}")
    print(f"faces {mano.nimble_faces.shape}", mano.nimble_faces.max())
    print(f"face_uvs {mano.nimble_face_uvs.shape}", mano.nimble_face_uvs.max())
    print(f"lmk_faces_idx {mano.lmk_faces_idx.shape}")
    print(f"lmk_bary_coords {mano.lmk_bary_coords.shape}")

    # nimble_uvw = torch.cat([
    #     mano.nimble_uvs, 
    #     torch.zeros_like(mano.nimble_uvs[..., -1:])], dim=-1)

    # nimble_mano_uv = torch.cat([
    #     vertices2landmarks(
    #         nimble_uvw.unsqueeze(0), 
    #         mano.nimble_face_uvs, 
    #         lmk_faces_idx[i], 
    #         lmk_bary_coords[i],
    #         ) for i in range(20)], dim=0)
    # nimble_mano_uvs = nimble_mano_uv.mean(dim=0)


    img = mano.tex_diff_mean
    fig, axs = plt.subplots(4, 5, figsize=(20, 16))
    axs = axs.ravel()
    for nimble_mano_uvs, ax in zip(mano.nimble_mano_uv_bary, axs):
        ax.axis('off')
        ax.imshow(img)
        ax.scatter(
            nimble_mano_uvs[:, 0] * img.shape[0], 
            (1-nimble_mano_uvs[:, 1]) * img.shape[1], 
            s=1, c=np.linspace(0,1,778))
        # plt.scatter(mano.nimble_uvs[:, 0] * img.shape[0], (1-mano.nimble_uvs[:, 1]) * img.shape[1], s=0.1)
    plt.show()

    plt.imshow(img)
    plt.scatter(
        mano.nimble_mano_uvs[:, 0] * img.shape[0], 
        (1-mano.nimble_mano_uvs[:, 1]) * img.shape[1], 
        s=1, c=np.linspace(0,1,778))
    plt.show()
    

#    with open("data/NIMBLE/NIMBLE_TEX_FUV.obj", 'w') as f:
#        f.writelines(mano.nimble_data["NIMBLE_TEX_FUV"])


# %%
