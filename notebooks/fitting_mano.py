# %%
import os
import sys
from pathlib import Path

print(os.path.dirname(__file__))
module_path = os.path.dirname(os.path.dirname(__file__))
if module_path not in sys.path:
    sys.path.append(module_path)

os.chdir(module_path)
print(os.getcwd())

# hov3_flame_number = 1000
hov3_flame_number = 12000
filepaths_pkl = Path("data/HO3D_v3/train/").glob("*/meta/*.pkl")
filepaths_pkl_list = list(filepaths_pkl)
filepaths_bg = Path("data/HO3D_v3/train/").glob("*/rgb/*.jpg")
filepaths_bg_list = list(filepaths_bg)
#max 0~20
object_number = 9
filepaths_obj = Path("data/models/").glob("*/textured_simple.obj")
filepaths_obj_list = list(filepaths_obj)

# %%
from utils.mano import ManoLayer
from utils import utils
import plotly.graph_objects as go

mano_layer = ManoLayer()

# add
anno = utils.load_ho_meta(filepaths_pkl_list[hov3_flame_number])
mano_hand = mano_layer(anno)
mano = mano_hand

# anno = utils.load_ho_meta()
# mano = mano_layer(anno)


fig = go.Figure()
utils.add_mesh_to_fig(fig, mano.vertices[0], mano_layer.faces, opacity=0.2)
fig.add_trace(
    go.Scatter3d(
        x=mano.joints[0][:, 0],
        y=mano.joints[0][:, 1],
        z=mano.joints[0][:, 2],
        mode='markers+text',
        marker=dict(
            size=2,
            color='red',
            opacity=0.5,
        ),
        text=[f'{i:02d}' for i in range(mano.joints.shape[1])],
        # hoverinfo='text',
        textposition='top center',
        name='mano.joints',
    )
)
fig.add_trace(
    go.Scatter3d(
        x=anno['handJoints3D'][:, 0],
        y=anno['handJoints3D'][:, 1],
        z=anno['handJoints3D'][:, 2],
        mode='markers',
        marker=dict(
            size=1,
            color='blue',
        ),
        name='handJoints3D',
    )
)

for i in range(16):
    if mano_layer.parents[i] == -1:
        continue
    fig.add_trace(
        go.Scatter3d(
            x=[mano.joints[0][i, 0], mano.joints[0][mano_layer.parents[i], 0]],
            y=[mano.joints[0][i, 1], mano.joints[0][mano_layer.parents[i], 1]],
            z=[mano.joints[0][i, 2], mano.joints[0][mano_layer.parents[i], 2]],
            mode='lines',
            line=dict(
                color='black',
                width=1,
            ),
        )
    )
fig.update_layout(
    scene=dict(
        xaxis=dict(showticklabels=False, showgrid=False, ),
        yaxis=dict(showticklabels=False, showgrid=False, ),
        zaxis=dict(showticklabels=False, showgrid=False, ),
    ))
fig.show()

# %%
print(mano.joints.shape)
print(mano_layer.mano_layer['right'].__dict__.keys())
print(mano_layer.mano_layer['right'].parents)
# dir(mano_layer.mano_layer['right'])

# %%
anno.keys()

# %%
# anno['handPose'].shape
print(anno['handJoints3D'])

# %%
17*3 + 5*3 + 4*3*2 + 17*6 + 17+5

# %%
214-22

# %%
import numpy as np
# Xmean = np.fromfile("Xmean.bin", dtype=np.float32)
# Ymean = np.fromfile("Ymean.bin", dtype=np.float32)
# Ymean.shape

# %%
import numpy as np
Xmean_fp = "modules/ManipNet/Code/Unity/ManipNetBIN/Xmean.bin"
Ymean_fp = "modules/ManipNet/Code/Unity/ManipNetBIN/Ymean.bin"
# manipnet_pred_fp = "/Users/shugo/workspace/rendering/data/output/manipnet_pred.txt"
# manipnet_pred_fp = "data/output/Xmean_pred.txt"
# manipnet_pred_fp = "data/output/manipnet_pred.txt"
# manipnet_pred_fp = "data/output/Ymean+zeros_pred.txt"
# manipnet_pred_fp = "data/output/zeros_pred.txt"
# manipnet_pred_fp = "data/output/Xmean_F_pred.txt"
# manipnet_pred_fp = "data/output/norm_pred.txt"
# manipnet_pred_fp = "data/output/norm_hov3_pred.txt"
manipnet_pred_fp = "data/output/norm_hov3_pred3.txt"

with open(Xmean_fp, "rb") as f:
    Xmean = np.fromfile(f, dtype=np.float32)
with open(Ymean_fp, "rb") as f:
    Ymean = np.fromfile(f, dtype=np.float32)
manipnet_pred = np.loadtxt(manipnet_pred_fp)

nJoint = 17
nFingers = 5
nPalm = 8


Xmean_pose = Xmean[:nJoint*3 + nFingers*3 + nPalm*3 + nJoint*6].reshape(-1, 3)
Xmean_dist = Xmean[nJoint*3 + nFingers*3 + nPalm*3 + nJoint*6:]
Ymean_pose = Ymean[:nJoint*3 + nFingers*3 + nPalm*3 + nJoint*6].reshape(-1, 3)
Ymean_dist = Ymean[nJoint*3 + nFingers*3 + nPalm*3 + nJoint*6:]
manipnet_pose = manipnet_pred[:nJoint*3 + nFingers*3 + nPalm*3 + nJoint*6].reshape(-1, 3)


Xmean_joints = Xmean_pose[:nJoint]
Xmean_fingers = Xmean_pose[nJoint:nJoint+nFingers]
Xmean_palms = Xmean_pose[nJoint+nFingers:nJoint+nFingers+nPalm]

Ymean_joints = Ymean_pose[:nJoint]
Ymean_fingers = Ymean_pose[nJoint:nJoint+nFingers]
Ymean_palms = Ymean_pose[nJoint+nFingers:nJoint+nFingers+nPalm]

manipnet_joints = manipnet_pose[:nJoint]
manipnet_fingers = manipnet_pose[nJoint:nJoint+nFingers]
manipnet_palms = manipnet_pose[nJoint+nFingers:nJoint+nFingers+nPalm]

# print(Ymean.shape)
# print(joints.reshape(-1, 3))
print(Xmean_joints)
print(Ymean_joints)
print(Ymean_fingers)

# %%
pose = 17*3 + 5*3 + 4*3 + 17*6
print(pose)
print(pose * 21)
print(pose + 21 * pose + 1000 + 17*3 + 17*3)
print(1000 + 104*6 + 22*9)

# %%
Xmean.shape

# %%
import plotly.graph_objects as go


def _add_points(fig, points, name, color):
    fig.add_trace(
        go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            mode='markers+text',
            marker=dict(
                size=2,
                color=color,
            ),
            name=name,
            text=[f'{name}{i:02d}' for i in range(points.shape[0])],
            hoverinfo='text',
            textposition='middle center',
            # textcolor=color,
        ))

def _add_lines(fig, points, indices, name, color):
    for inds in indices:
        fig.add_trace(
            go.Scatter3d(
                x=points[inds, 0],
                y=points[inds, 1],
                z=points[inds, 2],
                mode='lines',
                line=dict(
                    color=color,
                    width=2
                ),
                name=name
            )
        )

fig = go.Figure()
# utils.add_mesh_to_fig(fig, mano.vertices[0], mano_layer.faces, )

# _add_points(fig, Xmean_joints, 'Xmean_joints', 'blue')
# _add_points(fig, Xmean_fingers, 'Xmean_fingers', 'magenta')
_add_points(fig, Ymean_joints, 'j','red')
_add_points(fig, Ymean_fingers, 'f', 'green')
_add_points(fig, Ymean_palms, 'p', 'orange')
_add_lines(fig, Ymean_joints, [[0,1,2,3,4], [0,5,6,7], [0,8,9,10], [0,11,12,13], [0,14,15,16]], 'Ymean_joints', 'red')
_add_lines(fig, Ymean_palms, [[0,1,2,3], [0,4], [4,5,6,7], [3,7]], 'parms', 'orange')

fig1 = go.Figure()
_add_points(fig1, manipnet_joints,'j', 'red')
_add_points(fig1, manipnet_fingers,'f', 'green')
_add_points(fig1, manipnet_palms,'p', 'orange')
_add_lines(fig1, manipnet_joints, [[0,1,2,3,4], [0,5,6,7], [0,8,9,10], [0,11,12,13], [0,14,15,16]], 'manipnet_joints', 'red')
_add_lines(fig1, manipnet_palms, [[0,1,2,3], [0,4], [4,5,6,7], [3,7]], 'parms', 'orange')



fig.update_layout(
    scene=dict(
        xaxis=dict(showticklabels=False, showgrid=False, ),
        yaxis=dict(showticklabels=False, showgrid=False, ),
        zaxis=dict(showticklabels=False, showgrid=False, ),
        camera=dict(projection=dict(type='orthographic')),
    )
)
fig.update_layout(dict(
    plot_bgcolor='rgba(0, 0, 0, 0)',
    margin=dict(l=0, r=0, b=0, t=0)
))
fig.show()

fig1.update_layout(
    scene=dict(
        xaxis=dict(showticklabels=False, showgrid=False, ),
        yaxis=dict(showticklabels=False, showgrid=False, ),
        zaxis=dict(showticklabels=False, showgrid=False, ),
        camera=dict(projection=dict(type='orthographic')),
    )
)
fig1.update_layout(dict(
    plot_bgcolor='rgba(0, 0, 0, 0)',
    margin=dict(l=0, r=0, b=0, t=0)
))
fig1.show()

# %%
from tqdm import tqdm
import numpy as np
import torch
from vctoolkit import Timer

def model_run(params):
    params = torch.tensor(params, dtype=torch.float32)
    global_orient = params[:3].unsqueeze(0)
    hand_pose = params[3:3+45].unsqueeze(0)
    betas = params[3+45:3+45+10].unsqueeze(0)
    # betas = None
    transl = params[3+45+10:3+45+10+3].unsqueeze(0)
    model_output = mano_layer.mano_layer['right'](
        global_orient=global_orient, hand_pose=hand_pose, betas=betas, transl=transl)
    return model_output.joints[0].detach().numpy()

class Solver:
    def __init__(self, eps=1e-5, max_iter=30, mse_threshold=1e-8, verbose=False):
        """
        Parameters
        ----------
        eps : float, optional
        Epsilon for derivative computation, by default 1e-5
        max_iter : int, optional
        Max iterations, by default 30
        mse_threshold : float, optional
        Early top when mse change is smaller than this threshold, by default 1e-8
        verbose : bool, optional
        Print information in each iteration, by default False
        """
        self.eps = eps
        self.max_iter = max_iter
        self.mse_threshold = mse_threshold
        self.verbose = verbose
        self.timer = Timer()

    def get_derivative(self, model, params, n):
        """
        Compute the derivative by adding and subtracting epsilon

        Parameters
        ----------
        model : object
        Model wrapper to be manipulated.
        params : np.ndarray
        Current model parameters.
        n : int
        The index of parameter.

        Returns
        -------
        np.ndarray
        Derivative with respect to the n-th parameter.
        """
        params1 = np.array(params)
        params2 = np.array(params)

        params1[n] += self.eps
        params2[n] -= self.eps

        res1 = model(params1)
        res2 = model(params2)

        d = (res1 - res2) / (2 * self.eps)

        return d.ravel()

    def solve(self, model, target, init=None, u=1e-3, v=1.5):
        """
        Solver for the target.

        Parameters
        ----------
        model : object
        Wrapper to be manipulated.
        target : np.ndarray
        Optimization target.
        init : np,ndarray, optional
        Initial parameters, by default None
        u : float, optional
        LM algorithm parameter, by default 1e-3
        v : float, optional
        LM algorithm parameter, by default 1.5

        Returns
        -------
        np.ndarray
        Solved model parameters.
        """
        if init is None:
            init = np.zeros(61)
        out_n = np.shape(model(init).ravel())[0]
        jacobian = np.zeros([out_n, init.shape[0]])

        last_update = 0
        last_mse = 0
        params = init
        for i in tqdm(range(self.max_iter)):
            residual = (model(params) - target).reshape(out_n, 1)
            mse = np.mean(np.square(residual))

            if abs(mse - last_mse) < self.mse_threshold:
                return params

            for k in range(params.shape[0]):
                jacobian[:, k] = self.get_derivative(model, params, k)

            jtj = np.matmul(jacobian.T, jacobian)
            jtj = jtj + u * np.eye(jtj.shape[0])

            update = last_mse - mse
            delta = np.matmul(
                np.matmul(np.linalg.inv(jtj), jacobian.T), residual
            ).ravel()
            params -= delta

            if update > last_update and update > 0:
                u /= v
            else:
                u *= v

            last_update = update
            last_mse = mse

            if self.verbose:
                print(i, self.timer.tic(), mse)

        return params

init = np.concatenate([
    mano_hand.global_orient.reshape(-1).detach().numpy(),
    np.zeros(45),
    np.zeros(10),
    np.zeros(3),
])
# init = None

solver = Solver(verbose=True)
params_solved = solver.solve(
    model_run, 
    manipnet_joints[1:], # - mano_hand.joints.numpy()[:, 0:1],
    init=init)
# %%

import pyredner # pyredner will be the main Python module we import for redner.
import torch # We also import PyTorch
from PIL import Image
import numpy as np
from matplotlib.pyplot import imshow
from utils.utils import load_ho_meta, apply_transform_to_mesh
from utils.mano import ManoLayer

from pathlib import Path



print(filepaths_pkl_list[hov3_flame_number])
bg_img = np.array(Image.open(filepaths_bg_list[hov3_flame_number]), dtype=np.float32)/255.0
anno = load_ho_meta(filepaths_pkl_list[hov3_flame_number])
mano = ManoLayer()
mano_hand = mano(anno)
resolution = bg_img.shape[:2]

mano_object = pyredner.Object(
    vertices=mano_hand.vertices[0], 
    indices=mano.faces, 
    uvs=mano.uv,
    material=pyredner.Material(
        diffuse_reflectance=pyredner.Texture(mano.map.to(pyredner.get_device()))
        # diffuse_reflectance=torch.tensor((0.5, 0.5, 0.5), device=pyredner.get_device())))
    )
)
print(mano.faces.dtype)

world2cam = torch.eye(4)
R = torch.diag(torch.tensor([-1.,1.,-1.]))
world2cam[:3,:3] = R
cam2world = world2cam.inverse()
K = torch.tensor(anno['camMat'], dtype=torch.float32)
fx, fy = K.diagonal()[:2]
px, py = K[:2,2]
print(K)
intrinsic_mat = torch.tensor([
        [fx / resolution[1] * 2, 0.0000, px/resolution[1]-0.5],
        [0.0000, fy / resolution[1] * 2, py/resolution[0]-0.5],
        [0.0000, 0.0000, 1.0000]]
        )

fov = 2* torch.atan(0.5 * resolution[1] / K[0, 0]) * 180 / 3.1415926
print(fov)
camera = pyredner.Camera(
    intrinsic_mat=intrinsic_mat,
    # cam_to_world=cam2world,
    position = torch.tensor([0, 0, 0.], dtype=torch.float32),
    look_at = torch.tensor([0, 0, -1.], dtype=torch.float32),
    up = torch.tensor([0, 1., 0], dtype=torch.float32),
    # fov = torch.tensor([fov], dtype=torch.float32),
    resolution=resolution,
)
print(camera.__dict__)
dirlight = pyredner.DirectionalLight(
    direction = torch.tensor([0.0, 0.0, -1.0]), 
    intensity = torch.ones(3)*3.0,
)
# envmap = pyredner.EnvironmentMap(torch.tensor(bg_img))

objects = pyredner.load_obj(filepaths_obj_list[object_number], return_objects=True)
obj_object = pyredner.Object(
    vertices=apply_transform_to_mesh(objects[0].vertices, anno),
    indices=objects[0].indices, 
    uvs=objects[0].uvs,
    uv_indices=objects[0].uv_indices,
    material=objects[0].material
)

# create scene
scene = pyredner.Scene(
    camera = camera, 
    objects = [
        mano_object, 
        obj_object,
        ]
    )

# Render the scene.
# render = pyredner.render_albedo(scene, alpha=True)
render = pyredner.render_deferred(scene, lights=[dirlight], alpha=True)

# add
print("params_solved", params_solved.shape)
params = torch.tensor(params_solved, dtype=torch.float32)
global_orient = params[:3].unsqueeze(0)
hand_pose = params[3:3+45].unsqueeze(0)
betas = params[3+45:3+45+10].unsqueeze(0)
transl = params[3+45+10:3+45+10+3].unsqueeze(0)

mano_solved = mano_layer.mano_layer['right'](global_orient=global_orient, hand_pose=hand_pose, betas=betas, transl=transl)
print(mano_solved.vertices[0])
mano_object2 = pyredner.Object(
    vertices=mano_solved.vertices[0], 
    indices=mano.faces, 
    uvs=mano.uv,
    material=pyredner.Material(
        diffuse_reflectance=pyredner.Texture(mano.map.to(pyredner.get_device()))
        # diffuse_reflectance=torch.tensor((0.5, 0.5, 0.5), device=pyredner.get_device())))
    )
)

scene2 = pyredner.Scene(
    camera = camera, 
    objects = [
        mano_object2, 
        obj_object,
        ]
    )
render2 = pyredner.render_deferred(scene2, lights=[dirlight], alpha=True)

import matplotlib.pyplot as plt

fig, axs = plt.subplots(1, 3, figsize=(10, 5))
axs[0].imshow(bg_img)
axs[1].imshow(bg_img)
axs[1].imshow(torch.pow(render, 1.0/2.2).cpu())
axs[2].imshow(bg_img)
axs[2].imshow(torch.pow(render2, 1.0/2.2).cpu())

# %%
fig3 = go.Figure()
utils.add_mesh_to_fig(fig3, mano_solved.vertices[0], mano_layer.faces, opacity=0.2)
fig3.add_trace(
    go.Scatter3d(
        x=mano_solved.joints[0][:, 0],
        y=mano_solved.joints[0][:, 1],
        z=mano_solved.joints[0][:, 2],
        mode='markers+text',
        marker=dict(
            size=2,
            color='red',
            opacity=0.5,
        ),
        text=[f'{i:02d}' for i in range(mano_solved.joints.shape[1])],
        # hoverinfo='text',
        textposition='top center',
        name='mano_solved.joints',
    )
)
fig3.add_trace(
    go.Scatter3d(
        x=anno['handJoints3D'][:, 0],
        y=anno['handJoints3D'][:, 1],
        z=anno['handJoints3D'][:, 2],
        mode='markers',
        marker=dict(
            size=1,
            color='blue',
        ),
        name='handJoints3D',
    )
)

for i in range(16):
    if mano_layer.parents[i] == -1:
        continue
    fig3.add_trace(
        go.Scatter3d(
            x=[mano_solved.joints[0][i, 0], mano_solved.joints[0][mano_layer.parents[i], 0]],
            y=[mano_solved.joints[0][i, 1], mano_solved.joints[0][mano_layer.parents[i], 1]],
            z=[mano_solved.joints[0][i, 2], mano_solved.joints[0][mano_layer.parents[i], 2]],
            mode='lines',
            line=dict(
                color='black',
                width=1,
            ),
        )
    )
fig3.update_layout(
    scene=dict(
        xaxis=dict(showticklabels=False, showgrid=False, ),
        yaxis=dict(showticklabels=False, showgrid=False, ),
        zaxis=dict(showticklabels=False, showgrid=False, ),
    ))
fig3.show()
# %%
