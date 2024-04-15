# %%
import smplx
import torch
import numpy as np
import trimesh
# from trimesh import load_mesh

class ManoLayer:
    def __init__(self):
        self.smplx_path = 'data/mano_v1_2/models'
        self.mano_layer = {
            'right': smplx.create(self.smplx_path, 'mano', use_pca=False, is_rhand=True, flat_hand_mean=True),
            'left': smplx.create(self.smplx_path, 'mano', use_pca=False, is_rhand=False, flat_hand_mean=True),
        }
        self.faces = torch.tensor(self.mano_layer['right'].faces.astype(np.int32), dtype=torch.int32)
        self.mesh = trimesh.load_mesh('data/mano_v1_2/MANO_UV_right.obj')
        self.uv = torch.tensor(self.mesh.visual.uv, dtype=torch.float32)
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

    def load_nimble(self, nimble_path="/home/shugo/workspace/rendering/data/NIMBLE"):
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
                    v, vt, _ = vvv.split("/")
                    fvs.append(int(v))
                    fvts.append(int(vt))
            elif line.split()[0] == "vt":
                vts.append(list(map(float, line.split()[1:])))
        self.nimble_faces = np.array(fvs).reshape(-1, 3)
        self.nimble_face_uvs = np.array(fvts).reshape(-1, 3)
        self.nimble_uvs = np.array(vts).reshape(-1, 2)
        # print(fvs.shape, fvts.shape, vts.shape)

        self.tex_diff_basis = torch.tensor(self.nimble_data['NIMBLE_TEX_DICT']['diffuse']['basis'])
        self.tex_diff_mean  = torch.tensor(self.nimble_data['NIMBLE_TEX_DICT']['diffuse']['mean']).reshape(1024,1024,3)[..., [2,1,0]]
        self.tex_diff_std   = torch.tensor(self.nimble_data['NIMBLE_TEX_DICT']['diffuse']['std'])
        print(self.tex_diff_basis.shape, self.tex_diff_mean.shape, self.tex_diff_std.shape)




if __name__ == '__main__':
    import os
    os.chdir(os.path.join(os.path.dirname(__file__), '..'))
    from utils.utils import load_ho_meta

    anno = load_ho_meta()
    mano = ManoLayer()
    print(mano.mano_layer['right'].__dict__.keys())
    for k, v in mano(anno).__dict__.items():
        if v is not None:
            print(k, v.shape)
        # print(k, v)
    
    mano = ManoLayer()
    mano.load_nimble()
    print(mano.nimble_data["NIMBLE_MANO_VREG"].keys())
    

    from PIL import Image
    import matplotlib.pyplot as plt

    # img = Image.open(f"data/NIMBLE/rand_0_diffuse.png")
    img = mano.tex_diff_mean
    plt.imshow(img)
    plt.scatter(mano.nimble_uvs[:, 0] * img.shape[0], (1-mano.nimble_uvs[:, 1]) * img.shape[1], s=0.1)
    plt.show()


#    with open("data/NIMBLE/NIMBLE_TEX_FUV.obj", 'w') as f:
#        f.writelines(mano.nimble_data["NIMBLE_TEX_FUV"])
