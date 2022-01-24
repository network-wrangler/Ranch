# Ranch
This is the Python package for building standard roadway and transit networks from scratch. Before becoming `Ranch`, it was also referred to as the `Standard Network Building Pipeline`, which consisted of a series of Python notebooks.

It aims to have the following functionality:

 1. build OSM-based roadway network in standard [`Network Wrangler`](https://github.com/wsp-sag/network_wrangler) format
 2. build GTFS-based transit network in standard [`Network Wrangler`](https://github.com/wsp-sag/network_wrangler) format
 3. build travel model centroid connectors

For the latest code developments, check out the `develop` branch.

## Installation

### Requirements
- Ranch requires user to install [SharedStreets' Node.js implementation](https://github.com/sharedstreets/sharedstreets-js#sharedstreets-nodejs--javascript), using the Docker installation. User will first install Docker on the machine, and then build SharedStreets image using this [Dockerfile](https://github.com/wsp-sag/Ranch/blob/develop/ranch/Dockerfile). See more instruction below.

- Ranch uses Python 3.7 and above.  Requirements are stored in `requirements.txt` but are automatically installed when using `pip` as are development requirements (for now) which are located in `dev-requirements.txt`.

  The intallation instructions use the [`conda`](https://conda.io/en/latest/) virtual environment manager and some use the ['git'](https://git-scm.com/downloads) version control software.

### Ranch Installation
If you are managing multiple python versions, we suggest using [`virtualenv`](https://virtualenv.pypa.io/en/latest/) or [`conda`](https://conda.io/en/latest/) virtual environments. All commands should executed in a conda command prompt, not powershell or the system command prompt. Do not add conda to the system path during installation. This may cause problems with other programs that require python 2.7 to be placed in the system path.

Example using a conda environment and using the package manager [pip](https://pip.pypa.io/en/stable/) to install Ranch from the source on GitHub.

#### Basic Installation

```bash
conda config --add channels conda-forge
conda create python=3.7 rtree geopandas folium osmnx -n <my_ranch_environment>
conda activate <my_ranch_environment>
pip install git+https://github.com/wsp-sag/Ranch@master
```
Ranch can be installed in several ways depending on the user's needs. The above installation is the simplest method and is appropriate when the user does not anticipate needing to update Ranch. An update will require rebuilding the ranch environment. Installing from clone is slightly more involved and requires the user to have a git manager on their machine, but permits the user to install Ranch with the -e, edit, option so that their Ranch installation can be updated through pulling new commits from the Ranch repo without a full reinstallation.

Note: if you wanted to install from a specific tag/version number or branch, replace `@master` with `@<branchname>`  or `@tag`

#### From Clone (Recommended for project team)
If you are going to be working on Ranch locally, you might want to clone it to your local machine and install it from the clone.  The -e will install it in [editable mode](https://pip.pypa.io/en/stable/reference/pip_install/?highlight=editable#editable-installs).

```bash
conda config --add channels conda-forge
conda create python=3.7 rtree geopandas folium osmnx -n <my_ranch_environment>
conda activate <my_ranch_environment>
git clone -b develop https://github.com/wsp-sag/Ranch
cd Ranch
pip install -e .
```

Note: if you are not part of the project team and want to contribute code bxack to the project, please fork before you clone and then add the original repository to your upstream origin list per [these directions on github](https://help.github.com/en/articles/fork-a-repo).

### Docker & SharedStreets
Before running Ranch, user need to install SharedStreets via Docker. The steps are as following:

 1. Install Docker.
 2. Open command line prompt, change base directory to the Ranch directory
 ```bash
 cd path/to/your/ranch/directory/
 ```
 3. Build SharedStreets image using this [Dockerfile](https://github.com/wsp-sag/Ranch/blob/develop/ranch/Dockerfile), name the local image as `shst`
 ```bash
 cd ranch
 docker build -t shst .
 ```
 4. To test if SharedStreets image is built successfully, user can run the following to start a container node
 ```bash
 docker run -it --rm -v /path/to/your/local/directory:/usr/node/ shst:latest /bin/bash
 ```
 if succeeded, user can see a node being launched

## Documentation

Documentation is located at [https://wsp-sag.github.io/Ranch/](https://wsp-sag.github.io/Ranch/)

Edit the source of the documentation  in the `/docs` folder.

To build the documentation locally requires additional packages found in the `dev_requirements.txt` folder.  

To install these into your conda python environment execute the following from the command line in the Ranch folder:

```bash
conda activate <my_ranch_environment>
pip install -r dev-requirements.txt
```

To build the documentation, navigate to the `/docs` folder and execute the command: `make html`

```bash
cd docs
make html
```

## Usage

To learn basic Ranch functionality, please refer to the following jupyter notebooks in the `/notebooks` directory:  

 - `Ranch-Demo.ipynb`

Jupyter notebooks can be started by activating the Ranch conda environment and typing `jupyter notebook`:

```bash
conda activate <my_ranch_environment>
jupyter notebook
```

## Troubleshooting

**Issue: Conda is unable to install a library or to update to a specific library version**
Try installing libraries from conda-forge

```bash
conda install -c conda-forge *library*
```

**Issue: User does not have permission to install in directories**
Try running Anaconda Prompt as an administrator.

## Client Contact and Relationship
Before becoming `Ranch`, the `Standard Network Building Pipeline` consisted of a series of python notebooks, which have been implemented to build networks for regions including Minneapolis (Met Council), Bay Area (MTC), and Southeast Florida (SERPM). Those agencies all had different variations of the Standard Network Building Pipeline notebooks due to many reasons, e.g. steps in the pipeline have been getting revised and methods have been evolving, also agencies required different network features. However, the fundamental steps in the process are the same across regions. To standardize the network building process so that the same basic steps can be applied to any region, `Ranch` was created as a Python package that hosts universal network building steps. Our agency partners so far include Met Council, MTC, Miami-Dade TPO, and BART.
