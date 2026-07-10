Jetson SDK: JetPack 5.1.6 | [https://developer.nvidia.com/embedded/jetpack-sdk-516] | 
| only available for host Ubuntu version 20.04 or less

1. Use SDK-manager on Host machine to flash Jetson device with JetPack 5.1.6 username: nvidia, password: nvidia
2. On device boot, add line in (sudo visudo) : 
```
nvidia ALL=(ALL) NOPASSWD:ALL
```

3. Copy `./jetson_YOLO_setup.sh` from project on Host to Jetson device and run it.
scp ./jetson_YOLO_setup.sh nvidia@<jetson_ip>:/home/nvidia/
```bash
./jetson_YOLO_setup.sh 
```
