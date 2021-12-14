# Ranch
This is the package for building standard roadway and transit networks from scratch

It aims to have the following functionality:

 1. build OSM-based roadway network in standard [`Network Wrangler`](https://github.com/wsp-sag/network_wrangler) format
 2. build GTFS-based transit network in standard Network Wrangler format
 3. build travel model centroid connectors

## Installation

### Requirements
Ranch uses Python 3.7 and above.  Requirements are stored in `requirements.txt` but are automatically installed when using `pip` as are development requirements (for now) which are located in `dev-requirements.txt`.

The intallation instructions use the [`conda`](https://conda.io/en/latest/) virtual environment manager and some use the ['git'](https://git-scm.com/downloads) version control software.

### Basic instructions
If you are managing multiple python versions, we suggest using [`virtualenv`](https://virtualenv.pypa.io/en/latest/) or [`conda`](https://conda.io/en/latest/) virtual environments. All commands should executed in a conda command prompt, not powershell or the system command prompt. Do not add conda to the system path during installation. This may cause problems with other programs that require python 2.7 to be placed in the system path.

Example using a conda environment (recommended) and using the package manager [pip](https://pip.pypa.io/en/stable/) to install Ranch from the source on GitHub.

```bash
conda config --add channels conda-forge
conda create python=3.7 rtree geopandas folium osmnx -n <my_ranch_environment>
conda activate <my_ranch_environment>
pip install git+https://github.com/wsp-sag/Ranch@develop
```
Ranch can be installed in several ways depending on the user's needs. The above installation is the simplest method and is appropriate when the user does not anticipate needing to update Ranch. An update will require rebuilding the network wrangler environment. Installing from clone is slightly more involved and requires the user to have a git manager on their machine, but permits the user to install Ranch with the -e, edit, option so that their Ranch installation can be updated through pulling new commits from the Ranch repo without a full reinstallation.


#### Bleeding Edge
If you want to install a more up-to-date or development version you can do so by installing it from

```bash
conda config --add channels conda-forge
conda create python=3.7 rtree geopandas folium osmnx -n <my_ranch_environment>
conda activate <my_ranch_environment>
pip install git+https://github.com/wsp-sag/Ranch@develop
```

Note: if you wanted to install from a specific tag/version number or branch, replace `@master` with `@<branchname>`  or `@tag`

#### From Clone
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
WSP team member responsible for this repository is [Sijia Wang](sijia.wang@wsp.com).
