import pickle
import numpy as np
import torch
from scipy.spatial.transform import Rotation

def load_ho_meta(
        meta_fp = "data/HO3D_v3/train/ABF10/meta/0000.pkl"
    ):
    with open(meta_fp, 'rb') as f:
        meta = pickle.load(f, encoding='latin1')

    anno = {}
    for k, v in meta.items():
        if v is None or v == 'None':
            anno[k] = None
        else:
            anno[k] = np.array(v)

    return anno



def apply_transform_to_mesh(verts, anno):
    objRmat = torch.tensor(Rotation.from_rotvec(anno['objRot'][:, 0]).as_matrix(), dtype=torch.float32).to(verts.device)
    objTrans = torch.tensor(anno['objTrans']).to(verts.device)
    # verts = (objRmat @ verts.T + objTrans.T).T
    print(verts.shape, objRmat.shape, objTrans.shape)
    verts = verts @ objRmat.T + objTrans
    return verts
