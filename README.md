# NERF with Hash Encoding (+ extra)

- Install Colmap (If custom dataset is needed)
- Create an environment from given `*.yml` file
- Install `Segment-anything`:
    - `pip install git+https://github.com/facebookresearch/segment-anything.git`
    - `wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth` [inside models folder]

- NOTE: train_hash2.py is the latest trainer and requires large amount of RAM

To start training:
`python train_hash2.py --num_samples 128 --write`
- This would start training the lego model, if placed insider data
For more options, see:
`python train_hash2.py --help`

### Mesh Reconstruction
- We use marching cubes for forming a mesh out of the density values.
- Run: `python nerf2mesh.py` for generating a mesh
- It uses `bounds` values calculated during train time
- It also uses the saved nerf and encoder models

### Human Reconstruction
- First run `colmap2nerf` on the recorded video:
    - `python colmap2nerf.py --video_in <Path_to_video> --run_colmap`
- Edit the `config.yaml` with the input and output image directory
- Copy the `transforms.json` and segmented images to the same directory
- Rename `transforms.json` to `transforms_train.json` 
- Make a `transforms_tmp.json` containing the test image
- Run Nerf with the `--data_path` flag
