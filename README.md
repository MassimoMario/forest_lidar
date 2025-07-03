# forest-lidar 🌳🌳🛩️

Python implementation of a pipeline to classify vegetation (tree) points in Airborne LiDAR data using zero-shot learning abilities of DeepForest and SAM model.

[DeepForest](https://github.com/weecology/DeepForest) is a CNN-based Deep Learning model to detect individual trees in a RGB remote sensing image. The reference paper is [G.Weinstein et al. 2019](https://www.mdpi.com/2072-4292/11/11/1309) 

SAM, Segment Anything Model ([A.Kirillof et al. 2023](https://arxiv.org/abs/2304.02643)), is an AI model developed by Meta AI for image segmentation designed to segment objects in an image given a prompt (points, boxes, text).

Since DeepForest outputs bounding boxes for individual trees, in order to obtain a mask of the tree crown those boxes are given as input to SAM.

In this project SAM is loaded using the [segment-geospatial package](https://github.com/opengeos/segment-geospatial).

Using a model for 2D segmentation to classify 3D points, such as LiDAR data, is possible following the [segment-lidar](https://github.com/Yarroudh/segment-lidar) pipeline. First the LiDAR points are projected in a 2D image, then the model segments it. After obtaining the segmentation masks the third dimension is restored adding a new variable to each point storing the class label.

# Table of Contents
1. [Installation](#Installation)
2. [Requirements](#Requirements)
3. [Usage](#Usage)
4. [Repository structure](#Repository-structure)


# Installation

To start using the repository, first clone it:

```
git clone https://github.com/MassimoMario/forest_lidar.git
```

# Requirements
This project requires **Python &ge; 3.8** and the following libraries:
- `numpy`
- `matplotlib`
- `tqdm`
- `deepforest`
- `segment-geospatial`
- `laspy`
- `albumentations`==2.0.5

To install them you can run on the Bash:

```
pip install -r requirements.txt
```

# Usage
The main script [`main.py`](main.py) can be runned from the command line providing different command line arguments:

```
python main.py --path my_lidar.las --resolution 0.4 --window_size 60 --patch_overlap 0.25 --save True --save_path labeled_lidar.las
```

The output `.las` file will be equal as the input, except for an extra dimension called `tree_labels` storing the tree labels (0/1 for 'tree'/'non tree') for each point.

The command line parameters mean:

* `path`: path to the input .las file containing LiDAR data
* `resolution`: resolution parameter for the 2D projection, controlling the projected image dimension
* `window_size`: size of the window, in meters, to batch the image. It's a parameter required by DeepForest which batch the image before acting
* `patch_overlap`: percentage of windows patch overla, required by DeepForest
* `save`: option to save the resulting .las file with labels information
* `save_path`: name of the saved .las file

  
## :information_source: Help
For a complete list of parameters and their descriptions, run:

```
python simulation.py --help
```


# Repository structure
The repository contains the following folders and files:

- [`main.py`](main.py) is the main script for classifying tree points in a given LiDAR point data cloud
- [`forest_lidar_class.py`](forest_lidar_class.py) is the script containing the class ForestLidar
- [`requirements.txt`](requirements.txt) file contains the list of dependencies required for the project
