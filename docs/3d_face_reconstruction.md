# 3D Face Reconstruction

The final reconstruction path uses the official 3DDFA_V2 model rather than the early
weak perspective mesh baseline.

## Pipeline

1. Detect and crop the face.
2. Run 3DDFA_V2 dense 3D face reconstruction.
3. Export OBJ and PLY meshes.
4. Render front, left, right, top, bottom, and three-quarter views with OpenGL.
5. Save a multiview grid and machine-readable reports.

## Commands

```powershell
python src/reconstruction/run_3ddfa_v2_reconstruction.py --help
python src/reconstruction/render_3ddfa_mesh_opengl.py --help
```

The local upstream repository and pretrained assets are expected at
`third_party/3DDFA_V2`.

## Outputs

- Meshes and projected visualization: `outputs/3d_reconstruction/3ddfa_v2`
- Reconstruction metrics: `outputs/reports/face_3d_reconstruction_3ddfa_v2_result.*`
- OpenGL render metrics: `outputs/reports/face_3d_reconstruction_opengl_render_result.*`

The representative reconstruction contains 38,365 vertices and 76,073 triangles.
