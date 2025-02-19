# Data Preprocessing
We include our data preprocessing scripts (for flow and depth estimation) here. I haven't had time to refactor the code yet, so the users will have to manually edit code to make it work with different datasets. You might also have to install additional packages to run the scripts.

## Optical Flow
We use [FlowFormer++](https://github.com/XiaoyuShi97/FlowFormerPlusPlus) to generate optical flows between training views. You can setup the original repo and then copy the `gen_flow.py` and `backwarp.py` scripts to the root folder. To generate flow with masks, run:

```bash
python gen_flow.py
```

## Depth Estimation
We use [Depth Anything](https://github.com/LiheYoung/Depth-Anything) to estimate monocular depths for training views. You can setup the original repo and then copy the `run.py` script to the root folder. To generate depth, run:

```bash
python run.py
```


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
