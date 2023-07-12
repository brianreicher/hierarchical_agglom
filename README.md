# hierarchical_agglom

[![](https://img.shields.io/pypi/pyversions/mwatershed.svg)](https://pypi.python.org/pypi/mwatershed)
[![](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)


## A package for parallelized hierarchical-merge-tree-based image segmentation, built on MongoDB and C++.



* Free software: Apache 2.0 License

### Installation

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

Install `hierarchical_agglom`:

```bash
pip install git+https://github.com/brianreicher/hierarchical_agglom.git
```

### Features


### Usage

### Credits

This package builds upon [`waterz`](https://github.com/funkey/waterz/tree/master), developed at the Funke Lab.