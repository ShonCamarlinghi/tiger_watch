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
CLASSES = ["Dog", "Cat", "Tiger", "Bird", "Snake", "Bear"]
```
Due to enourmouse size of open-images dataset, I deliberately avoided default open-images-v7.yaml from Ultralytics https://docs.ultralytics.com/datasets/detect/open-images-v7#applications 
Instead I set fiftyOne dataset container small by picking split="validation" and max_samples=20000  and predefined list of CLASSES in my script.
```
    dataset = foz.load_zoo_dataset(
        "open-images-v7",
        split="validation",         # This keeps the sample count manageable on an RTX 4060 Ti.
        classes=CLASSES,
        label_types=["detections"], 
        label_field="ground_truth",
        max_samples=20000,          # Set max samples to 20,000
        dataset_name=DATASET_NAME,
        persistent=True,
```

```bash
python 51_dataset_to_YAML.py
```
51_dataset_to_YAML.py creates dataset.yaml and images, labels folders with train/val split and downloads dataset of predefined classes of limited sample size.
```
(.venv) (base) shon@s2:~/PycharmProjects/tiger_watch$ ls /home/shon/Sandbox/datasets/YOLO_wildlife
dataset.yaml  images  labels
```
dataset.yaml content:
``
names:
  0: Dog
  1: Cat
  2: Tiger
  3: Bird
  4: Snake
  5: Bear
path: /home/shon/Sandbox/datasets/YOLO_wildlife
train: ./images/train/
val: ./images/val/
```

### Training 
Train a YOLO model:

```bash
python train_Ultralytics_YOLO_model.py --data path/to/dataset.yaml
```
Snapshot of training start:
```
(.venv) (base) shon@s2:~/PycharmProjects/tiger_watch$ python train_Ultralytics_YOLO_model.py --data /home/shon/Sandbox/datasets/YOLO_wildlife/dataset.yaml
Current ultralytics settings... JSONDict("/home/shon/.config/Ultralytics/settings.json"):
{
  "settings_version": "0.0.6",
  "datasets_dir": "/home/shon/PycharmProjects/datasets",
  "weights_dir": "/home/shon/PycharmProjects/tiger_watch/weights",
  "runs_dir": "/home/shon/PycharmProjects/tiger_watch/runs",
  "uuid": "b84c7a844290d326925dbcbae9fb553d3363b1f32c00e2ba424fdc349f27a78f",
  "sync": true,
  "api_key": "",
  "openai_api_key": "",
  "clearml": true,
  "comet": true,
  "dvc": true,
  "hub": true,
  "mlflow": true,
  "neptune": true,
  "raytune": true,
  "tensorboard": false,
  "wandb": false,
  "vscode_msg": true,
  "openvino_msg": true
}
New https://pypi.org/project/ultralytics/8.4.88 available 😃 Update with 'pip install -U ultralytics'
Ultralytics 8.4.82 🚀 Python-3.12.3 torch-2.12.1+cu130 CUDA:0 (NVIDIA GeForce RTX 4060 Ti, 7806MiB)
engine/trainer: agnostic_nms=False, amp=True, angle=1.0, augment=False, auto_augment=randaugment, batch=16, bgr=0.0, box=7.5, cache=False, cfg=None, classes=None, close_mosaic=10, cls=0.5, cls_pw=0.0, compile=False, conf=None, copy_paste=0.0, copy_paste_mode=flip, cos_lr=False, cutmix=0.0, data=/home/shon/Sandbox/datasets/YOLO_wildlife/dataset.yaml, degrees=0.0, deterministic=True, device=None, dfl=1.5, dis=6.0, distill_model=None, dnn=False, dropout=0.0, dynamic=False, embed=None, end2end=None, epochs=100, erasing=0.4, exist_ok=False, fliplr=0.5, flipud=0.0, format=torchscript, fraction=1.0, freeze=None, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, imgsz=640, iou=0.7, keras=False, kobj=1.0, line_width=None, lr0=0.01, lrf=0.01, mask_ratio=4, max_det=300, mixup=0.0, mode=train, model=yolov8n.pt, momentum=0.937, mosaic=1.0, multi_scale=0.0, name=tiger_watch_yolo-5, nbs=64, nms=False, opset=None, optimize=False, optimizer=auto, overlap_mask=True, patience=100, perspective=0.0, plots=True, pose=12.0, pretrained=True, profile=False, project=runs/train, quantize=None, rect=False, resume=False, retina_masks=False, rle=1.0, save=True, save_conf=False, save_crop=False, save_dir=/home/shon/PycharmProjects/tiger_watch/runs/detect/runs/train/tiger_watch_yolo-5, save_frames=False, save_json=False, save_period=-1, save_txt=False, scale=0.5, seed=0, shear=0.0, show=False, show_boxes=True, show_conf=True, show_labels=True, simplify=True, single_cls=False, source=None, split=val, stream_buffer=False, task=detect, time=None, tracker=tracktrack.yaml, translate=0.1, val=True, verbose=True, vid_stride=1, visualize=False, warmup_bias_lr=0.1, warmup_epochs=3.0, warmup_momentum=0.8, weight_decay=0.0005, workers=8, workspace=None
Downloading https://ultralytics.com/assets/Arial.ttf to '/home/shon/.config/Ultralytics/Arial.ttf': 100% ━━━━━━━━━━━━ 755.1KB 11.5MB/s 0.1s
Overriding model.yaml nc=80 with nc=6

                   from  n    params  module                                       arguments                     
  0                  -1  1       464  ultralytics.nn.modules.conv.Conv             [3, 16, 3, 2]                 
  1                  -1  1      4672  ultralytics.nn.modules.conv.Conv             [16, 32, 3, 2]                
  2                  -1  1      7360  ultralytics.nn.modules.block.C2f             [32, 32, 1, True]             
  3                  -1  1     18560  ultralytics.nn.modules.conv.Conv             [32, 64, 3, 2]                
  4                  -1  2     49664  ultralytics.nn.modules.block.C2f             [64, 64, 2, True]             
  5                  -1  1     73984  ultralytics.nn.modules.conv.Conv             [64, 128, 3, 2]               
  6                  -1  2    197632  ultralytics.nn.modules.block.C2f             [128, 128, 2, True]           
  7                  -1  1    295424  ultralytics.nn.modules.conv.Conv             [128, 256, 3, 2]              
  8                  -1  1    460288  ultralytics.nn.modules.block.C2f             [256, 256, 1, True]           
  9                  -1  1    164608  ultralytics.nn.modules.block.SPPF            [256, 256, 5]                 
 10                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          
 11             [-1, 6]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 12                  -1  1    148224  ultralytics.nn.modules.block.C2f             [384, 128, 1]                 
 13                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          
 14             [-1, 4]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 15                  -1  1     37248  ultralytics.nn.modules.block.C2f             [192, 64, 1]                  
 16                  -1  1     36992  ultralytics.nn.modules.conv.Conv             [64, 64, 3, 2]                
 17            [-1, 12]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 18                  -1  1    123648  ultralytics.nn.modules.block.C2f             [192, 128, 1]                 
 19                  -1  1    147712  ultralytics.nn.modules.conv.Conv             [128, 128, 3, 2]              
 20             [-1, 9]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 21                  -1  1    493056  ultralytics.nn.modules.block.C2f             [384, 256, 1]                 
 22        [15, 18, 21]  1    752482  ultralytics.nn.modules.head.Detect           [6, 16, None, [64, 128, 256]] 
Model summary: 130 layers, 3,012,018 parameters, 3,012,002 gradients, 8.2 GFLOPs

Transferred 319/355 items from pretrained weights
Freezing layer 'model.22.dfl.conv.weight'
AMP: running Automatic Mixed Precision (AMP) checks...
AMP: checks passed ✅
train: Fast image access ✅ (ping: 0.0±0.0 ms, read: 6191.4±1911.9 MB/s, size: 209.1 KB)
train: Scanning /home/shon/Sandbox/datasets/YOLO_wildlife/labels/train... 3200 images, 2275 backgrounds, 0 corrupt: 100% ━━━━━━━━━━━━ 3200/3200 7.3Kit/s 0.4s
train: New cache created: /home/shon/Sandbox/datasets/YOLO_wildlife/labels/train.cache
val: Fast image access ✅ (ping: 0.0±0.0 ms, read: 6595.7±6415.5 MB/s, size: 301.5 KB)
val: Scanning /home/shon/Sandbox/datasets/YOLO_wildlife/labels/val... 800 images, 597 backgrounds, 0 corrupt: 100% ━━━━━━━━━━━━ 800/800 4.4Kit/s 0.2s
val: New cache created: /home/shon/Sandbox/datasets/YOLO_wildlife/labels/val.cache
optimizer: 'optimizer=auto' found, ignoring 'lr0=0.01' and 'momentum=0.937' and determining best 'optimizer', 'lr0' and 'momentum' automatically... 
optimizer: AdamW(lr=0.001, momentum=0.9) with parameter groups 57 weight(decay=0.0), 64 weight(decay=0.0005), 63 bias(decay=0.0)
Plotting labels to /home/shon/PycharmProjects/tiger_watch/runs/detect/runs/train/tiger_watch_yolo-5/labels.jpg... 
Image sizes 640 train, 640 val
Using 8 dataloader workers
Logging results to /home/shon/PycharmProjects/tiger_watch/runs/detect/runs/train/tiger_watch_yolo-5
Starting training for 100 epochs...

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      1/100      2.09G     0.8904      3.502      1.383          8        640: 100% ━━━━━━━━━━━━ 200/200 8.5it/s 23.4s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 25/25 5.0it/s 5.0s
                   all        800        256      0.657      0.202      0.212      0.108

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      2/100      2.84G      1.063      3.074      1.516         12        640: 100% ━━━━━━━━━━━━ 200/200 9.0it/s 22.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 25/25 5.7it/s 4.3s
                   all        800        256      0.686      0.198      0.215      0.104

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      3/100      2.85G      1.241      2.692      1.636         15        640: 100% ━━━━━━━━━━━━ 200/200 9.1it/s 21.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 25/25 5.9it/s 4.2s
                   all        800        256      0.658      0.194      0.189      0.107
...

```



