# Tiger Watch
# Cascaded Edge-AI Wildlife & Pet Identification Pipeline

Objective: A two-stage computer vision pipeline designed to detect humans, birds, and pets, 
    and identify 37 specific breeds of cats and dogs. 
This project bridges the gap between high-end desktop training (**NVIDIA RTX 4060ti**) 
and resource-constrained edge deployment (**NVIDIA Jetson AGX Xavier**) using **TensorRT** optimization.

Computer Vision Task: Object detection and classify: 
    animal vs object, if animal: animal.type(wild, pet, human, bird), 
    if animal.type(pet): classify dog(breed) or cat(breed).
Classes: animal (background = no label), human, pet, dog, cat. 

Success metric: 

Hardware: 
    - training: AMD Ryzen 7, NVIDIA RTX 4090
    - deployment: NVIDIA Jetson AGX Xavier,  
                  wifi adapter  
                  Intel RealSense camera


## Training

Install dependencies:

```bash
pip install -r requirements.txt
```
pip install ultralytics installs: 
- ultralytics python package
- Yolo command-line tool
- a pinned PyTorch matching your platform

Train a YOLO model:

```bash
python train_Ultralytics_YOLO_model.py --data path/to/dataset.yaml
```

