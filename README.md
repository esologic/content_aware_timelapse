# Content-Aware Timelapse - content_aware_timelapse 

Create timelapses from video that change speed based on the content found in the video. Boring sections are sped over.

## Usage

With the virtual env activated, run: 

```
python catcli.py \
--input "/home/devon/Desktop/Overhead Camera/pwm_driver_module/pwm_drive_module_v1.2.0_1.mp4" \
--input "/home/devon/Desktop/Overhead Camera/pwm_driver_module/pwm_drive_module_v1.2.0_2.mp4" \
--input "/home/devon/Desktop/Overhead Camera/pwm_driver_module/pwm_drive_module_v1.2.0_3.mp4" \
--duration 30 \
--output-fps 60 \
--batch-size 1200 \
--vectors-path ./pwm_module_assembly.hdf5 \
--output-path ./big_mean.mp4
```

## Getting Started

### GPU Acceleration

You need to have NVidia drivers and a CUDA development environment to be able to use GPU
acceleration. These are the steps I ran:

```
wget https://developer.download.nvidia.com/compute/cuda/12.3.0/local_installers/cuda_12.3.0_545.23.06_linux.run
wget https://developer.download.nvidia.com/compute/cudnn/9.4.0/local_installers/cudnn-local-repo-ubuntu2004-9.4.0_1.0-1_amd64.deb
sudo dpkg -i cudnn-local-repo-ubuntu2004-9.4.0_1.0-1_amd64.deb
sudo cp /var/cudnn-local-repo-ubuntu2004-9.4.0/cudnn-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cudnn cudnn-cuda-12
```

Eventually, it'd be great to wrap this in a docker container so only the driver is required. 

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

## Problems & Roadmap

1. A massive inefficiency in the application is how fast video can be read from disk. Currently,
for the video that we're working with, I'm able to read the video at ~250 frames per second. I
have tried a few things to make this go faster:

* Using `ffmpeg` over openCV.
* Creating worker processes to parallelize the read.
* Reading from the video in the background and writing the frames to disk as HDF5 arrays, then
loading the HDF5 files back into memory when required.

Nothing has been faster than just straight openCV. An in-memory buffer is also used to do this
load in the background, but then you run into memory limits.