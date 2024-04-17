import kaolin
import torch

class KaolinRenderer:
    def __init__(self, verts, faces, uvs, tex_diff=None, tex_spec=None, tex_normal=None):
        verts = verts.to(torch.float32)
        faces = faces.to(torch.int64)
        uvs = uvs.to(torch.float32)

        self.device = 'cuda:0'
        self.mesh = kaolin.rep.SurfaceMesh(
            verts, faces, uvs=uvs, face_uvs_idx=faces
        ).to(self.device)
        self.tex_diff = tex_diff.to(torch.float32).permute(2,0,1).unsqueeze(0) if tex_diff is not None else None
        self.tex_spec = tex_diff.to(torch.float32).unsqueeze(0) if tex_spec is not None else None
        self.tex_normal = tex_diff.to(torch.float32).permute(2,0,1).unsqueeze(0) if tex_normal is not None else None

    def set_verts(self, verts):
        self.mesh.vertices = torch.tensor(verts).to(self.device, dtype=torch.float32)

    def set_camera(self, K, w2c, width, height):
        fx, fy = K.diagonal()[:2]
        cx, cy = K[:2, 2]
        # w2cs = [torch.diag(torch.tensor([-1.,1.,-1.,1.]))]
        #w2c[:3, :3] = w2c[:3, :3] #@ torch.diag(torch.tensor([1., -1., -1.]))
        #w2c[:3, 3:] = torch.diag(torch.tensor([1., -1., -1.])) @ w2c[:3, 3:]
        w2c[:3, :3] = w2c[:3, :3] * np.array([[1,-1,-1]]).T
        w2c[:3, 3:] = w2c[:3, 3:] * np.array([[1,-1,-1]]).T
        w2cs = [torch.tensor(w2c).to(torch.float32)]

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
            extrinsics=extrinsics, intrinsics=intrinsics).to(self.device)

    def rasterize(self, width, height):
        self.mesh.to(self.device)
        vertices_camera = self.cam.extrinsics.transform(self.mesh.vertices)
        vertices_ndc = self.cam.intrinsics.transform(vertices_camera)

        face_vertices_camera = kaolin.ops.mesh.index_vertices_by_faces(vertices_camera, self.mesh.faces)
        face_vertices_image = kaolin.ops.mesh.index_vertices_by_faces(vertices_ndc[..., :2], self.mesh.faces)
        face_vertices_z = face_vertices_camera[..., -1]

        #print(self.mesh)
        #print(self.mesh.face_normals)
        #print(self.mesh.face_uvs)
        face_features = [
            self.mesh.face_uvs, #.to(self.device), 
            self.mesh.face_normals, #.to(self.device),
            torch.ones_like(self.mesh.face_normals)#.to(self.device),
        ]

        use_dibr = False
        if use_dibr:
            im_features, soft_max, face_idx = kaolin.render.mesh.dibr_rasterization(
                height, width,
                face_vertices_z.to(self.device), 
                face_vertices_image[..., :2].to(self.device),
                face_features, 
                face_normals_z = self.mesh.face_normals[..., 2:].to(self.device),
                rast_backend = 'cuda'
            )
        else:
            im_features, face_idx = kaolin.render.mesh.rasterize(
                height, width,
                face_vertices_z.cuda(), #.to(self.device), 
                face_vertices_image[..., :2].to(self.device),
                face_features, backend='cuda'
            )
        self.hard_mask = face_idx != -1
        self.uv_map = im_features[0]
        self.im_world_normal = im_features[1]
        self.backface_mask = torch.where(self.im_world_normal[..., 2] > 0, torch.tensor([1.]).to(self.device), torch.tensor([0.]).to(self.device))
        im_diff_albedo = kaolin.render.mesh.texture_mapping(self.uv_map, self.tex_diff.to(self.device))
        self.im_diff_albedo = torch.clamp(im_diff_albedo * self.hard_mask.unsqueeze(-1), min=0., max=1.)
        # self.im_diff_albedo = torch.clamp(im_diff_albedo, min=0., max=1.)
        
    def renderSH9(self, coef_sh9):
        irradiance_r = kaolin.render.lighting.sh9_irradiance(coef_sh9[0], self.im_world_normal[self.hard_mask])
        irradiance_g = kaolin.render.lighting.sh9_irradiance(coef_sh9[1], self.im_world_normal[self.hard_mask])
        irradiance_b = kaolin.render.lighting.sh9_irradiance(coef_sh9[2], self.im_world_normal[self.hard_mask])
        irradiance = torch.stack([irradiance_r, irradiance_g, irradiance_b], dim=-1)
        self.render_diffuse = torch.zeros_like(self.im_diff_albedo)
        self.render_diffuse[self.hard_mask] = irradiance * self.im_diff_albedo[self.hard_mask]
    
    def renderSG(self, azimuth, elevation, amplitude, sharpness, im_world_normal,
                diff_albedo, spec_albedo, roughness, rays_d):
        directions = torch.stack(kaolin.ops.coords.spherical2cartesian(azimuth, elevation), dim=-1)
        img = torch.zeros((directions.shape[0], *im_world_normal.shape), device='cuda')
        # Render diffuse component
        diffuse_effect = kaolin.render.lighting.sg_diffuse_fitted(
            amplitude,
            directions,
            sharpness,
            im_world_normal[self.hard_mask],
            diff_albedo[self.hard_mask]
        )
        img[self.hard_mask] += diffuse_effect
        
        # Render specular component
        specular_effect = kaolin.render.lighting.sg_warp_specular_term(
            amplitude,
            directions,
            sharpness,
            im_world_normal[self.hard_mask],
            roughness[self.hard_mask],
            rays_d[self.hard_mask],
            spec_albedo[self.hard_mask]
        )
        img[self.hard_mask] += specular_effect
        return img, diffuse_effect, specular_effect
    
    def optimizeSH9(self, gt_img, mask=None, n_iter=100, lr=1.0):
        self.coef_sh9 = torch.zeros(3, 9, device=self.device, requires_grad=True)
        self.optimizer_sh9 = torch.optim.SGD([self.coef_sh9], lr=lr)
        for i in range(n_iter):
            self.optimizer_sh9.zero_grad()
            self.renderSH9(self.coef_sh9)
            if mask is not None:
                loss = torch.nn.functional.mse_loss(self.render_diffuse * mask, gt_img*mask)
            else:
                loss = torch.nn.functional.mse_loss(self.render_diffuse[self.hard_mask], gt_img[self.hard_mask])
            loss.backward()
            self.optimizer_sh9.step()
            if i > 0 and i % (10 ** int(math.log10(i))) == 0:
                print(i, loss.item())
    
    def optimizeSG(self, gt_img, mask=None, nb_sg=2, n_iter=100, lr=1.0):
        #self.coef_sh9 = torch.zeros(3, 9, device=self.device, requires_grad=True)
        #self.optimizer_sh9 = torch.optim.SGD([self.coef_sh9], lr=lr)
        
        self.azimuth = torch.rand((nb_sg), device='cuda', requires_grad=True)
        self.elevation = torch.zeros((nb_sg), device='cuda', requires_grad=True)
        self.amplitude = torch.ones((nb_sg, 3), device='cuda') / nb_sg
        self.amplitude.requires_grad = True
        self.sharpness = torch.ones((nb_sg), device='cuda', requires_grad=True)

        self.roughness = torch.full(
            size=(height, width), 
            fill_value=0.1, 
            device='cuda')
        
        self.rays_d = generate_pinhole_rays_dir(self.cam)
        
        self.optimizer_sg = torch.optim.SGD([
            self.azimuth, self.elevation, self.amplitude, self.sharpness], lr=lr)

        for i in range(n_iter):
            self.optimizer_sg.zero_grad()
            self.renderSG(
                self.azimuth, self.elevation, self.amplitude, self.sharpness, 
                self.im_world_normal, self.im_diff_albedo, self.tex_spec, self.roughness, 
                self.rays_d)
            
            if mask is not None:
                loss = torch.nn.functional.mse_loss(self.render_diffuse * mask, gt_img*mask)
            else:
                loss = torch.nn.functional.mse_loss(self.render_diffuse[self.hard_mask], gt_img[self.hard_mask])

            loss.backward()
            self.optimizer_sg.step()

            if i > 0 and i % (10 ** int(math.log10(i))) == 0:
                print(i, loss.item())
        

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