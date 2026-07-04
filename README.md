# Tiger Watch
# Cascaded Edge-AI Wildlife & Pet Identification Pipeline

Objective: A two-stage computer vision pipeline designed to detect humans, birds, and pets, 
    and identify 37 specific breeds of cats and dogs. 
This project bridges the gap between high-end desktop training (**NVIDIA RTX 4060ti**) 
and resource-constrained edge deployment (**NVIDIA Jetson AGX Xavier**) using **TensorRT** optimization.

Computer Vision Task: Object detection and classify: 
    animal vs object, if animal: animal.type(wild, pet, human, bird), 
    if animal.type(pet): classify dog(breed) or cat(breed).
    if animal.type(wild): classify wild.kind(tiger, bear, hawk, etc.)
Classes: animal (background = no label), human, pet, dog, cat. 

Success metric: 

Hardware: 
    - training: AMD Ryzen 7, NVIDIA RTX 4060ti
    - deployment: NVIDIA Jetson AGX Xavier,  
                  wifi adapter  
                  Intel RealSense camera


Dataset source : Open Images Dataset v7 from Google | https://storage.googleapis.com/openimages/web/download_v7.html  
Dataset management: Voxel51 | https://docs.voxel51.com/index.html#

### SW Environment setup for model training
Venv setup in terminal  
```bash
python -m venv .venv
source .venv/bin/activate            # Linux/macOS
python --version                     # confirm 3.9+  
```

```bash
pip install -r requirements.txt
```
"pip install ultralytics" gets you following: 
- ultralytics python package (up to version 12 currently supported)
- Yolo command-line tool
- a pinned PyTorch matching your platform

To access environment with ultralytics and torch: 
```bash
source .venv/bin/activate
```

#### yolo check
```
(.venv) (base) shon@s2:~/PycharmProjects/tiger_watch$ yolo check
Ultralytics 8.4.82 🚀 Python-3.12.3 torch-2.12.1+cu130 CUDA:0 (NVIDIA GeForce RTX 4060 Ti, 7806MiB)
Setup complete ✅ (16 CPUs, 31.0 GB RAM, 292.6/914.8 GB disk)

OS                     Linux-6.17.0-29-generic-x86_64-with-glibc2.39
Environment            Linux
Python                 3.12.3
Install                git
Path                   /home/shon/PycharmProjects/tiger_watch/.venv/lib/python3.12/site-packages/ultralytics
RAM                    30.99 GB
Disk                   292.6/914.8 GB
CPU                    AMD Ryzen 7 8700F 8-Core Processor
CPU count              16
GPU                    NVIDIA GeForce RTX 4060 Ti, 7806MiB
GPU count              1
CUDA                   13.0

numpy                  ✅ 2.5.0>=1.23.0
matplotlib             ✅ 3.11.0>=3.3.0
opencv-python          ✅ 4.13.0.92>=4.6.0
pillow                 ✅ 12.2.0>=7.1.2
pyyaml                 ✅ 6.0.3>=5.3.1
requests               ✅ 2.34.2>=2.23.0
torch                  ✅ 2.12.1>=1.8.0
torch                  ✅ 2.12.1!=2.4.0,>=1.8.0; sys_platform == "win32"
torchvision            ✅ 0.27.1>=0.9.0
psutil                 ✅ 7.2.2>=5.8.0
polars                 ✅ 1.42.0>=0.20.0
nvidia-ml-py           ✅ 13.610.43>=12.0.0
ultralytics-thop       ✅ 2.0.20>=2.0.18
(.venv) (base) shon@s2:~/PycharmProjects/tiger_watch$ 
```

#### Inference sanity
```bash
yolo predict model=yolo26n.pt source='https://ultralytics.com/images/bus.jpg'
```
### FiftyOne downloads and prepare dataset.

Dataset source : Open Images Dataset v7 from Google | https://storage.googleapis.com/openimages/web/download_v7.html  
Dataset management: Voxel51 | https://docs.voxel51.com/index.html#
Using FiftyOne API we download datasets, split to "train" and "val", create YOLO friendly YAML 
Define variables in the header of the script before run, i.e.:
```
DATASET_NAME = "my_wildlife_dataset"
EXPORT_DIR = "/home/shon/Sandbox/datasets/YOLO_wildlife"
CLASSES = ["Person", "Dog", "Cat", "Tiger", "Bird", "Snake", "Bear"]
```
```bash
python 51_dataset_to_YAML.py
```
This script creates dataset.yaml and images and labels folders in EXPORT_DIR
```
(.venv) (base) shon@s2:~/PycharmProjects/tiger_watch$ ls /home/shon/Sandbox/datasets/YOLO_wildlife
dataset.yaml  images  labels
```
Content of created dataset.yaml:
```
names:
  0: Person
  1: Dog
  2: Cat
  3: Tiger
  4: Bird
  5: Snake
  6: Bear
path: /home/shon/Sandbox/datasets/YOLO_wildlife
val: ./images/val/
```

### Training 
Train a YOLO model:

```bash
python train_Ultralytics_YOLO_model.py --data path/to/dataset.yaml
```




