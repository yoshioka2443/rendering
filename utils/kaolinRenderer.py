import kaolin
import torch

class KaolinRenderer:
    def __init__(self, verts, faces, uvs, tex_diff=None, tex_spec=None, tex_normal=None):
        verts = verts.to(torch.float32)
        faces = faces.to(torch.int64)
        uvs = uvs.to(torch.float32)

        self.mesh = kaolin.rep.SurfaceMesh(
            vertices=verts, faces=faces, uvs=uvs, face_uvs_idx=faces
        )
        self.tex_diff = tex_diff.to(torch.float32).permute(2,0,1).unsqueeze(0) if tex_diff is not None else None
        self.tex_spec = tex_diff.to(torch.float32).unsqueeze(0) if tex_spec is not None else None
        self.tex_normal = tex_diff.to(torch.float32).permute(2,0,1).unsqueeze(0) if tex_normal is not None else None
        self.device = 'cuda:0'

    def set_camera(self, K, w2c, width, height):
        fx, fy = K.diagonal()[:2]
        cx, cy = K[:2, 2]
        # w2cs = [torch.diag(torch.tensor([-1.,1.,-1.,1.]))]
        w2c[:3, :3] = w2c[:3, :3] @ torch.diag(torch.tensor([1., -1., -1.]))
        w2c[:3, 3:] = torch.diag(torch.tensor([1., -1., -1.])) @ w2c[:3, 3:]
        w2cs = [w2c]

        params = [torch.tensor([0,0,fx,fy], dtype=torch.float32)]
        intrinsics = kaolin.render.camera.PinholeIntrinsics(
            params=torch.stack(params, dim=0),
            width=width,
            height=height,
        )
        extrinsics = kaolin.render.camera.CameraExtrinsics.from_view_matrix(
            torch.stack(w2cs, dim=0),
        )
        self.cam = kaolin.render.camera.Camera(
            extrinsics=extrinsics, intrinsics=intrinsics)

    def render(self, width, height):
        vertices_camera = self.cam.extrinsics.transform(self.mesh.vertices)
        vertices_ndc = self.cam.intrinsics.transform(vertices_camera)

        face_vertices_camera = kaolin.ops.mesh.index_vertices_by_faces(vertices_camera, self.mesh.faces)
        face_vertices_image = kaolin.ops.mesh.index_vertices_by_faces(vertices_ndc[..., :2], self.mesh.faces)
        face_vertices_z = face_vertices_camera[..., -1]

        im_features, face_idx = kaolin.render.mesh.rasterize(
            height, width,
            face_vertices_z.to(self.device), face_vertices_image[..., :2].to(self.device),
            [self.mesh.face_uvs.to(self.device), self.mesh.face_normals.to(self.device)],
            backend='cuda'
        )
        self.hard_mask = face_idx != -1
        print(self.hard_mask.shape)
        self.uv_map = im_features[0]
        self.im_world_normal = im_features[1]
        im_diff_albedo = kaolin.render.mesh.texture_mapping(self.uv_map, self.tex_diff.to(self.device))
        self.im_diff_albedo = torch.clamp(im_diff_albedo * self.hard_mask.unsqueeze(-1), min=0., max=1.)
        # self.im_diff_albedo = torch.clamp(im_diff_albedo, min=0., max=1.)


if __name__ == '__main__':
    pass
    # tex_diff_resized = cv2.resize(tex_diff, (256, 256), interpolation=cv2.INTER_LINEAR)
    # print(tex_diff_resized.shape)
    # renderer = KaolinRenderer(mano.vertices[0], mano_layer.faces, mano_layer.uv, torch.tensor(tex_diff_resized))
    # renderer.set_camera(
    #     K=torch.tensor(anno['camMat']),
    #     w2c=torch.diag(torch.tensor([-1.,1.,-1.,1.])),
    #     width=640, height=480)
    # renderer.render(width=640, height=480)

    # plt.imshow(renderer.im_diff_albedo.detach().cpu()[0])