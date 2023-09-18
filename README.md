# hierarchical_agglom

[![](https://img.shields.io/pypi/pyversions/mwatershed.svg)](https://pypi.python.org/pypi/mwatershed)
[![](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)


## A package for parallelized hierarchical-merge-tree-based image segmentation, built on MongoDB and C++.



* Free software: Apache 2.0 License

### Installation
A full installation script can be found in [`install.sh`](https://github.com/brianreicher/hierarchical_agglom/blob/main/install.sh), or line-by-line installs can be ran as follows:

Install C++ dependencies:

```bash
sudo apt install libboost-dev
```


Install MongoDB:

```bash
curl -fsSL https://pgp.mongodb.com/server-6.0.asc | sudo gpg -o /usr/share/keyrings/mongodb-server-6.0.gpg --dearmor
```


And initialize a MongoDB server in a screen on your machine:

```bash
screen
mongod
```

Install ``graph_tool``

```bash
conda install -c conda-forge -c ostrokach-forge -c pkgw-forge graph-tool
```

Install `hglom`:

```bash
pip install git+https://github.com/brianreicher/hierarchical_agglom.git
```

### Features

* A hierarchical-merge-tree implementation to go from affininities to a full segmentation
* Pipeline utilizes a MongoDB RAG to produce successive iterations of segmentations
* Should handle up to 6 dimensional arrays of fragments (7 dimensions if you include offsets in affinities) but only tested on 2 and 3 dimensions

### Usage

Example usage for generating a post-processing segmentation from affinities:

```python
import hglom

pp: hglom.PostProcessor = hglom.PostProcessor(
            affs_file="../data/raw_predictions.zarr",
            affs_dataset="pred_affs_latest",
        )
pp.segment()
```
where:
* `affs_file` is a path (relative or absolute) to the zarr file containing predicted affinities to generate fragments for.
* `affs_dataset` is the name of the affinities dataset in the affs_file to read from.


### Credits

This package builds upon [`waterz`](https://github.com/funkey/waterz/tree/master), developed at the Funke Lab.