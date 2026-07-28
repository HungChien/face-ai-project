# Third-Party Dependencies

The final 3D reconstruction pipeline uses the official 3DDFA_V2 implementation.
Keep external repositories local rather than committing their source, Git metadata,
and pretrained weights into this project.

```bash
git clone https://github.com/cleardusk/3DDFA_V2.git third_party/3DDFA_V2
cd third_party/3DDFA_V2
sh ./build.sh
```

Follow the upstream repository instructions to obtain its pretrained assets. The local
runner expects the repository at `third_party/3DDFA_V2`.
