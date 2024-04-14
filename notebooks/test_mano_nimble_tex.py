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
light = trimesh.scene.lighting.AmbientLight(color=(1, 1, 1))
mesh.show()

# %%

if __name__ == '__main__':
    pass
