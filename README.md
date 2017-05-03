# ColorHandPose3D network

![Teaser](teaser.png)

ColorHandPose3D is a Convolutional Neural Network estimating 3D Hand Pose from a single RGB Image. See the [project page](https://lmb.informatik.uni-freiburg.de/projects/hand3d/) for the dataset used and additional information.


## Usage
The network ships with a minimal example, that performs a forward pass and shows the predictions.

- Download [data](https://lmb.informatik.uni-freiburg.de/projects/hand3d/ColorHandPose3D_data.zip) and unzip it into the projects root folder
- *run.py* - Will run a forward pass of the network on the provided sample images

You can compare your results to the content of the folder "./results/", which shows the predictions we get on our system.


## Recommended system
Recommended system (tested):
- Ubuntu 16.04.2 (xenial)
- Tensorflow 0.11.0 RC0 GPU build with CUDA 8.0.44 and CUDNN 5.1
- Python 3.5.2

Python packages used by the example provided and their recommended version:
- tensorflow==0.11.0rc0
- numpy==1.11.3
- scipy==0.18.1
- matplotlib==1.5.3

## License and Citation
This project is licensed under the terms of the GPL v2 license. By using the software, you are agreeing to the terms of the [license agreement](https://github.com/lmb-freiburg/hand3d/LICENSE).


Please cite us in your publications if it helps your research:

@inproceedings{zim2017hand3d,
author = {C. Zimmermann and T. Brox},
title = {Learning to Estimate 3D Hand Pose from Single RGB Images},
year = {2017}
}

