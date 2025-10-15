# **Starting kit**

We provide a Pytorch starting kit to run some baselines. The following documentation describes how to install the, download the LTDv2 dataset, and run some experiments.

## Installation

To install the project, clone the repository:
```sh
git clone https://github.com/MarcoParola/RTIOD.git
cd RTIOD
cd starting_kit
mkdir data
```

Then, create a virtual environment and install the necessary dependencies.

On Windows
```sh
python -m venv env
env\Scripts\activate
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
```

On Linux
```sh
python -m venv env
. env/bin/activate
python -m pip install torch torchvision
python -m pip install -r requirements.txt
```

## Get the LTDv2 dataset from HuggingFace

- download the frames and the annotations
- unzip the frames
- convert the the dataset from coco to yolo format

## Usage

- how to run basic training
