# %%
import os
import sys

module_path = os.path.dirname(os.path.dirname(__file__))
if module_path not in sys.path:
    sys.path.append(module_path)
os.chdir(module_path)
print(os.getcwd())

# %%
from utils.mano import ManoLayer
from utils import utils
import plotly.graph_objects as go

mano_layer = ManoLayer()
anno = utils.load_ho_meta()
mano = mano_layer(anno)


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
Ymean.shape

# %%
import numpy as np

Xmean_fp = "modules/ManipNet/Code/Unity/ManipNetBIN/Xmean.bin"
Ymean_fp = "modules/ManipNet/Code/Unity/ManipNetBIN/Ymean.bin"
# manipnet_pred_fp = "/Users/shugo/workspace/rendering/data/output/manipnet_pred.txt"
manipnet_pred_fp = "data/output/Xmean_pred.txt"

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



