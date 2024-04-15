import plotly.graph_objects as go

class PlotlyVisualizer():
    def __init__(self):
        self.fig = go.Figure()

    def add_mesh(self, verts, faces, **kwargs):
        x, y, z = verts[:,0], verts[:,1], verts[:,2]
        i, j, k = faces[:,0], faces[:,1], faces[:,2]
        self.fig.add_trace(
            go.Mesh3d(
                x=x, y=y, z=z, i=i, j=j, k=k, 
                **kwargs
                )
            )

    def add_points(self, points, texts=None, color='blue'):
        x, y, z = points[:,0], points[:,1], points[:,2]
        self.fig.add_trace(
            go.Scatter(
                x=x, y=y, z=z, 
                mode='markers+text' if texts is not None else'markers', 
                text=texts if texts is not None else None,
                marker=dict(
                    size=2, color=color),
            )
        )
    
    def show(self):
        self.fig.show()