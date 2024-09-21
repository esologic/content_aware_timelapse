# Content-Aware Timelapse - content_aware_timelapse 

Create timelapses from video that change speed based on the content found in the video. Boring sections are sped over.

## Table of Contents
[TOC]

## Usage

####  Environment variables



#### Correct usage

With the virtual env activated, run: 

```
python main.py --fill-me-in
```

#### Incorrect usage

Add if needed


## Getting Started

### GPU Acceleration

https://www.tensorflow.org/install/pip#software_requirements

wget https://developer.download.nvidia.com/compute/cuda/12.3.0/local_installers/cuda_12.3.0_545.23.06_linux.run

wget https://developer.download.nvidia.com/compute/cudnn/9.4.0/local_installers/cudnn-local-repo-ubuntu2004-9.4.0_1.0-1_amd64.deb
sudo dpkg -i cudnn-local-repo-ubuntu2004-9.4.0_1.0-1_amd64.deb
sudo cp /var/cudnn-local-repo-ubuntu2004-9.4.0/cudnn-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cudnn

sudo apt-get -y install cudnn-cuda-12

### Python Dependencies

Poetry is required to manage Python dependencies. You can install it easily by following the
operating system specific instructions [here](https://python-poetry.org/docs/#installation).

`pyproject.toml` contains dependencies for required Python modules for building, testing, and 
developing. They can all be installed in a [virtual environment](https://docs.python.org/3/library/venv.html) 
using the follow commands:

```
python3.10 -m venv .venv
source ./.venv/bin/activate
poetry install
```

There's also a bin script to do this, and will install poetry if you don't already have it:

```
./tools/create_venv.sh
```

#### Adding new dependencies


## Developer Guide

The following is documentation for developers that would like to contribute
to Content-Aware Timelapse.

### Pycharm Note

Make sure you mark `content_aware_timelapse` and `./test` as source roots!

### Testing

This project uses pytest to manage and run unit tests. Unit tests located in the `test` directory 
are automatically run during the CI build. You can run them manually with:

```
./tools/run_tests.sh
```

### Local Linting

There are a few linters/code checks included with this project to speed up the development process:

* Black - An automatic code formatter, never think about python style again.
* Isort - Automatically organizes imports in your modules.
* Pylint - Check your code against many of the python style guide rules.
* Mypy - Check your code to make sure it is properly typed.

You can run these tools automatically in check mode, meaning you will get an error if any of them
would not pass with:

```
./tools/run_checks.sh
```

Or actually automatically apply the fixes with:

```
./tools/apply_linters.sh
```

There are also scripts in `./tools/` that include run/check for each individual tool.


### Using pre-commit

Upon cloning the repo, to use pre-commit, you'll need to install the hooks with:

```
pre-commit install --hook-type pre-commit --hook-type pre-push
```

By default:

* black
* pylint
* isort
* mypy

Are all run in apply-mode and must pass in order to actually make the commit.

Also by default, pytest needs to pass before you can push.

If you'd like skip these checks you can commit with:

```
git commit --no-verify
```

If you'd like to quickly run these pre-commit checks on all files (not just the staged ones) you
can run:

```
pre-commit run --all-files
```


