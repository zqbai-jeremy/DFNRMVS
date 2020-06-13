# Deep Facial Non-Rigid Multi-View Stereo
Source code for CVPR 2020 paper "Deep Facial Non-Rigid Multi-View Stereo" 
[[paper]](http://openaccess.thecvf.com/content_CVPR_2020/papers/Bai_Deep_Facial_Non-Rigid_Multi-View_Stereo_CVPR_2020_paper.pdf) 
[[supp]](http://openaccess.thecvf.com/content_CVPR_2020/supplemental/Bai_Deep_Facial_Non-Rigid_CVPR_2020_supplemental.zip) 
[[video]](https://drive.google.com/file/d/1xtPhG1czqjiYBgAXuzd4bWe1gcr3ow8D/view?usp=sharing).

![](webpage_files/5106-teaser.gif)

## Installation

(1) Create an Anaconda environment with python 3.6.

```bash
conda create -n DFNRMVS python=3.6
source activate DFNRMVS
```

(2) Clone the repository and install dependencies.

```bash
git clone https://github.com/zqbai-jeremy/DFNRMVS.git
cd DFNRMVS
conda install --yes --file requirements_conda.txt
pip install -r requirements_pip.txt
```

(3) Setup 3DMM

- Clone [this repository](https://github.com/zqbai-jeremy/face3d.git), which is forked and modified from 
[YadiraF/face3d](https://github.com/YadiraF/face3d.git), to "\<DFNRMVS directory\>/external/".

```bash
mkdir external
cd external
git clone https://github.com/zqbai-jeremy/face3d.git
cd face3d
```

- Setup face3d as in [YadiraF/face3d](https://github.com/YadiraF/face3d#getting-started).

- Download "Exp_Pca.bin" from [Guo et al.](https://github.com/Juyong/3DFace) (in "CoarseData" link of their repository)
and copy to "\<DFNRMVS directory\>/external/face3d/examples/Data/BFM/Out/".

- Download "std_exp.txt" from [Deng et al.](https://github.com/microsoft/Deep3DFaceReconstruction/blob/master/BFM/std_exp.txt)
and copy to "\<DFNRMVS directory\>/external/face3d/examples/Data/BFM/Out/".

(4) Install [face-alignment](https://github.com/1adrianb/face-alignment).

```bash
conda install -c 1adrianb face_alignment
```

(5) Download pre-trained model ([2views_model.pth](https://drive.google.com/file/d/1HvsJ8IztdUS8VUNSLlkdWsOOJ3AgZ2Pe/view?usp=sharing) 
or [3views_finetune_model.pth](https://drive.google.com/file/d/1GITjxOq_QzJqY3KTfB3UmAYt8ZRaUDO8/view?usp=sharing); 
May be used for research purpose only) to "\<DFNRMVS directory\>/net_weights/". Need to create the folder.

## Run Demo

- Modify directory paths in demo.py and run

```bash
cd <DFNRMVS_directory>
python demo.py
```

- All images in the input directory will be used for reconstruction. Per-view results will be saved to the output directory.

- Some examples are in "\<DFNRMVS directory\>/examples/". The corresponding outputs are in "\<DFNRMVS directory\>/out_dir/".
