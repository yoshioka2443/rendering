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

    def __call__(self, anno, is_rhand=True):
        root_pose = torch.from_numpy(anno['handPose']).view(-1, 3)[0].view(1, 3).float()#.cuda()
        hand_pose = torch.from_numpy(anno['handPose']).view(-1, 3)[1:].view(1, -1).float()#.cuda()
        betas = torch.from_numpy(anno['handBeta']).view(1, -1).float()#.cuda()
        trans = torch.from_numpy(anno['handTrans']).view(1, 3).float()#.cuda()
        output = self.mano_layer['right' if is_rhand else 'left'](global_orient=root_pose, hand_pose=hand_pose, betas=betas, transl=trans)
        return output.vertices

if __name__ == '__main__':
    mano = ManoLayer()