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

if __name__ == '__main__':
    from utils import load_ho_meta
    anno = load_ho_meta()
    mano = ManoLayer()
    print(mano.mano_layer['right'].__dict__.keys())
    for k, v in mano(anno).__dict__.items():
        if v is not None:
            print(k, v.shape)
        # print(k, v)

# class ManoLayer:
#     def __init__(self):
#         self.smplx_path = 'data/mano_v1_2/models'
#         self.mano_layer = {
#             'right': smplx.create(self.smplx_path, 'mano', use_pca=False, is_rhand=True, flat_hand_mean=True),
#             'left': smplx.create(self.smplx_path, 'mano', use_pca=False, is_rhand=False, flat_hand_mean=True),
#         }
#         self.faces = torch.tensor(self.mano_layer['right'].faces.astype(np.int32), dtype=torch.int32)
#         self.mesh = trimesh.load_mesh('data/mano_v1_2/MANO_UV_right.obj')
#         self.uv = torch.tensor(self.mesh.visual.uv, dtype=torch.float32)
#         self.map = torch.ones([256, 256, 3], dtype=torch.float32) * torch.tensor([[[1.0, 0.5, 0.5]]])

#     def __call__(self, anno, is_rhand=True):
#         root_pose = torch.from_numpy(anno['handPose']).view(-1, 3)[0].view(1, 3).float()#.cuda()
#         hand_pose = torch.from_numpy(anno['handPose']).view(-1, 3)[1:].view(1, -1).float()#.cuda()
#         betas = torch.from_numpy(anno['handBeta']).view(1, -1).float()#.cuda()
#         trans = torch.from_numpy(anno['handTrans']).view(1, 3).float()#.cuda()
#         output = self.mano_layer['right' if is_rhand else 'left'](global_orient=root_pose, hand_pose=hand_pose, betas=betas, transl=trans)
#         return output.vertices

# if __name__ == '__main__':
#     mano = ManoLayer()