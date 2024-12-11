# <img src="./assets/imgs/dog_icon.png" style="width:50px;height:auto"> DOGS

<b>DOGS</b>: Distributed-Oriented Gaussian Splatting for Large-Scale 3D Reconstruction Via Gaussian Consensus 

[[🌐 Project Page](https://aibluefisher.github.io/DOGS) | [arXiv](https://arxiv.org/abs/2405.13943)] (**NeurIPS 2024**)

-----------------------------------------------

## 🛠️ Installation

Install the conda environment of ZeroGS.

```sh
conda create -n dogs python=3.9
conda activate dogs
cd DOGS/scripts
./scripts/env/install.sh
```

## 🤷 Introduction

Our method accelerates the training of 3DGS by 6+ times when evaluated on large-scale scenes while concurrently achieving state-of-the-art rendering quality.

<img src="./assets/imgs/dogaussian_pcl.gif" style="width:480px;height:auto" />
<img src="./assets/imgs/dogaussian.gif" style="width:480px;height:auto" />

## 🚀 TODO & Roadmap

- ✔️ Release evaluation code 🎉
- 🔲 Release pre-trained models on `Mill19`, `UrbanScene3D`, and `MatrixCity`
- 🔲 Release web-viewer.
- ✔️ Release training code
    - ✔️ Gaussian Splatting trainer 🎉
    - ✔️ Scaffold-GS trainer 🎉
    - 🔲 ADMM Gaussian Splatting trainer
- 🔲 Test on street-view scenes
- 🔲 Support distributed training of `Scaffold-GS` and `Octree-GS`

## 📋 Train & Test

### ⚗️ Preprocess Large-Scale dataset

We first run the provided script to pre-process a large-scale scene into several blocks:
```bash
cd scripts/preprocess
./preprocess_large_scale_data.sh 0 urban3d gaussian_splatting
```


<details>
<summary><b>Visualize scene splitting</b></summary>

Please check and compile [my modification of COLMAP](https://github.com/AIBluefisher/colmap). After installation, launch COLMAP's GUI. I extended the original model files of COLMAP with an additional `cluster.txt` file, where each line of the file follows the format: [image_id, cluster_id]. Once COLMAP's GUI finds this file, it will render each image with its color corresponding to its cluster ID. Below are some examples of scene splitting:

![sci-art_blocks_2x4_cameras](https://github.com/user-attachments/assets/218ff44e-0f9a-43ab-bb72-99421f5702a4)

![campus_blocks_2x4_cameras](https://github.com/user-attachments/assets/dea576c7-a480-4c12-886e-46113e08465b)


</details>

### ⌛ Train 3D Gaussian Splatting

#### Train 3DGS on a single GPU
```bash
cd scripts/train
./train_nvs.sh 0 $EXP_SUFFIX urban3d gaussian_splatting
```

#### Train 3DGS on multiple GPUs
Here we provide scripts and an example to show how to run DOGS on three compute nodes with 9 GPUs in total (1 GPU on a master node and 4 GPUs each of two slave nodes).

Before running the program, we may need to modify the parameters in the provided scripts:
(1) `scripts/train/train_admm_master.sh`:
- set `NUM_TOTAL_NODES` to the correct total number of GPUs (In this example, we use 9 GPUs as described above)
- set `ETHERNET_INTERFACE` to the ethernet interface of your computer(we can get the correct interface of your server by typing `ifconfig` in the terminal of a Linux machine)
- set `DATASET` to the dataset you want to reconstruct
- set the correct IP address of the master node `--master_addr=xx.xx.xx.xx`

(2) Modify the above mentioned parameters accordingly in `scripts/train/train_admm_worker1.sh` and `scripts/train/train_admm_worker2.sh`.

At first, in the terminal of the master node, we run:
```bash
cd scripts/train
./train_admm_master.sh $EXP_SUFFIX urban3d_admm
```

Then, we establish workers in the terminal for each of the two slave nodes:

##### On the slave node #1
```bash
cd scripts/train
./train_admm_worker1.sh $EXP_SUFFIX urban3d_admm
```

##### On the slave node #2
```bash
cd scripts/train
./train_admm_worker2.sh $EXP_SUFFIX urban3d_admm
```

### 📊 Evaluate 3D Gaussian Splatting

```bash
cd scripts/eval
./eval_nvs.sh 0 $EXP_SUFFIX urban3d gaussian_splatting
```

After that, we can have a cup of coffee and wait the master node connects with the slave nodes and finishes the training.

## ✏️ Cite

If you find this project useful for your research, please consider citing our paper:
```bibtex
@inproceedings{yuchen2024dogaussian,
    title={DOGS: Distributed-Oriented Gaussian Splatting for Large-Scale 3D Reconstruction Via Gaussian Consensus},
    author={Yu Chen, Gim Hee Lee},
    booktitle={arXiv},
    year={2024},
}
```

## 🙌 Acknowledgements

This work is built upon [3d-gaussian-splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/). We sincerely thank the authors for releasing their code. Yu Chen is also partially supported by a Google PhD Fellowship for finishing this project.

## 🪪 License

Copyright © 2024, Chen Yu.
All rights reserved.
Please see the [license file](LICENSE) for terms.
