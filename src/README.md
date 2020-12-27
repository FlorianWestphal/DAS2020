## Requirements

 - Python 3
 - numpy
 - pandas
 - pillow
 - scikit-learn
 - scikit-image
 - [irlbpy](https://github.com/bwlewis/irlbpy)

## Installation

### gedlib

 - change into `gedlib` subdirectory
 - build library as follows:
```sh
python install.py --lib gxl
```

## Workflow

The workflow for extracting the different image representations and performing 
the image selection are described in the following.
The scripts assume that the word images are located in separate folders, one for
each word class, which may look as follows:
```
konzil
  +--- 0
       +--- 0.png
       ...
  +--- 1
       +--- 0.png
       ...
  +--- 2
  +--- 3
  ... 
```
Furthermore, the availability of a CSV file mapping from the folder id to the
corresponding transcription is assumed.
In the following, we will refer to this file as `label_list.csv` and it should 
be formatted as follows:
```
0,Furthermore
1,the
2,availability
...
```

### Temporal Pyramid Representation (TP-R)

#### Computing the Distance Matrix

The distance matrix for TP-R can be be computed from the word images formatted
as described above using the following command:
```sh
python tools/compute_distance_matrix.py level level_dist.csv --img_dir data/konzil --clean
```
In this example, the images are located in the folder `data/konzil`.
The `--clean` option indicates that word fragments at all four borders will be 
removed using seam carving before computing the TP-R.
The final distance matrix will be stored in the file `level_dist.csv`.

#### Selecting the Word Images

Based on the produced distance matrix, the training samples can be selected as 
follows:
```sh
python tools/select_samples.py data/konzil/ selected/ 100 --distance_file level_dist.csv
```
The selected images will be stored in the folder `selected` in following format:
```
selected
   +--- <label_id>_<img_id>.png
```

### Graph Representation (G-R)

#### Extracting the Word Graphs

The graphs for each of the word images can be extracted as follows:
```sh
python tools/extract_graphs.py data/konzil graphs/ 16 10
```
This script assumes the same folder structure as described above for the source
folder `data/konzil`.
The output graphs are written to the provided output folder, here `graphs`, as
follows:
```
graphs
  +--- <label_id>_<img_id>.gxl
  +--- graphs_dv<dv>_dh<dh>.xml
```
The `.xml` file contains a list of all extracted graphs and is needed in the 
next step when computing the distance matrix.

#### Computing the Distance Matrix

In order to compute the distance matrix between all graphs in the `.xml` file,
the program `parallel_hed` has to be built first.
This can be done as follows, assuming that `gedlib` has been already installed:

```sh
cd hed
make
```

The graph edit distances can then be computed as follows:
```sh
. set_library_path.sh
./parallel_hed ../graphs/ ../graphs/graphs_dv16_dh10.xml
```
This will produce a CSV file called `distances_tv1_te1_a05_b05.csv`, which 
contains the distance matrix.
The vertex and edge insertion/deletion and the alpha and beta parameter can be 
configured with the command line options `--tv`, `--te`, `--alpha` and `--beta`
respectively.

#### Selecting the Word Images

The images can then be selected as before:
```sh
python tools/select_samples.py data/konzil/ ged_selected/ 100 --distance_file distances_tv1_te1_a05_b05.csv
```

### PHOC Representation (PHOC-R)

#### Computing the Distance Matrix

In order to compute the distance matrix, first one has to predict the PHOC 
descriptors for each image using a pre-trained PHOCNet.
This can be done as follows:
```sh
cd phocnet
python tools/predict_phocs.py --img_dir data/konzil --pretrained_phocnet phocnet_nti500000_pul2-3-4-5.binaryproto --deploy_proto protofiles/deploy_phocnet.prototxt
```
This script will produce an NPZ file, which is then used to compute the distance
matrix as follows:
```sh
python tools/compute_distance_matrix.py pretrain pretrain_dist.csv --phoc_file predicted_output_sigmoid.npz
```

#### Selecting the Word Images

The images can then be selected as before:
```sh
python tools/select_samples.py data/konzil/ phoc_selected/ 100 --distance_file pretrain_dist.csv
```

## Training on the Selected Images

### Create LMDB

In order to train only on the selected images, we need to create a new LMDB file
for training.
This can be achieved as follows:
```sh
cd phocnet
python tools/create_train_lmdb.py phoc_selected/ label_list.csv lmdbs/ Konzilsprotokolle --n_train_images 500000 --augment transform
``` 
Here, `label_list.csv` is the CSV file mapping the folder id to the
corresponding word transcription, as described above.
Furthermore, `Konzilsprotokolle` is the dataset name as encoded in the LMDB 
files originally created when running PHOCNet as described 
[here](https://github.com/ssudholt/phocnet) 

### Train PHOCNet

After creating the training LMDB file, it can be used to fine tune a
pre-trained PHOCNet model.
This can be done as follows:
```sh
cd phocnet
python tools/train_phocnet -dm --gpu_id 0 --dataset_name Konzilsprotokolle --proto_dir protofiles/ --lmdb_dir lmdbs/ --save_net_dir models/ --solverstate models/pretrained/snapshot_iter_80000 -lr 0.00001 --solver SGD --max_iter 80000
```
