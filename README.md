# CoherentGS

> CoherentGS: Sparse Novel View Synthesis with Coherent 3D Gaussians  
> [Avinash Paliwal](http://avinashpaliwal.com/),
> [Wei Ye](https://ywwwer.github.io/), 
> [Jinhui Xiong](https://jhxiong.github.io/), 
> [Dmytro Kotovenko](https://scholar.google.com/citations?user=T_U8yxwAAAAJ&hl), 
> [Rakesh Ranjan](https://scholar.google.com/citations?user=8KF99lYAAAAJ&hl), 
> [Vikas Chandra](https://v-chandra.github.io/), 
> [Nima Khademi Kalantari](http://nkhademi.com/)   
> ECCV 2024

[![Paper](https://img.shields.io/badge/cs.CV-Paper-b31b1b?logo=arxiv&logoColor=red)](https://arxiv.org/abs/2403.19495)
[![Project Page](https://img.shields.io/badge/CoherentGS-Website-blue?logo=googlechrome&logoColor=blue)](https://people.engr.tamu.edu/nimak/Papers/CoherentGS/index.html)
[![Video](https://img.shields.io/badge/YouTube-Video-c4302b?logo=youtube&logoColor=red)](https://youtu.be/WxtSM12M81A)

---------------------------------------------------
<p align="center" >
  <a href="">
    <img src="assets/video.gif?raw=true" alt="demo" width="45%">
  </a>
  <a href="">
    <img src="assets/videoD.gif?raw=true" alt="demo" width="45%">
  </a>
</p>


## Prerequisites
You can setup the anaconda environment using:
```bash
conda env create --file environment.yml
conda activate coherentgs
```
**CUDA 11.7** is strongly recommended.


## Data Preparation
You can download the processed [LLFF dataset here](https://drive.google.com/file/d/1VwT8cXjCVVM1Q3UGP6FSZXi5d5JDFZ_p/view?usp=sharing). We will add optimized pointclouds soon.

## Training
Training on LLFF dataset with 3 views. You can choose from `[2, 3, 4]` views
``` bash
python train.py --source_path path/nerf_llff_data/flower --eval --model_path output/flower --num_cameras 3
``` 


## Rendering
Run the following script to render the video.  

```bash
python renderpath.py -source_path path/nerf_llff_data/flower --eval --model_path output/flower
```


## Acknowledgement
The repo is built on top of [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting)  
The modified rasterizer to render depth and eval script is from [FSGS](https://github.com/VITA-Group/FSGS)

## Citation
If you find our work useful for your project, please consider citing the following paper.
```
@inproceedings{paliwal2024coherentgs,
  title={Coherentgs: Sparse novel view synthesis with coherent 3d gaussians},
  author={Paliwal, Avinash and Ye, Wei and Xiong, Jinhui and Kotovenko, Dmytro and Ranjan, Rakesh and Chandra, Vikas and Kalantari, Nima Khademi},
  booktitle={European Conference on Computer Vision},
  pages={19--37},
  year={2024},
  organization={Springer}
}
```
