Jetson AGX Xavier + JetPack 5.1.6 | [https://developer.nvidia.com/embedded/jetpack-sdk-516] | 
| only available for host Ubuntu version 20.04 or less

Realsense SDK setup ref:  https://github.com/realsenseai/librealsense/blob/master/doc/installation_jetson.md


1. Register the server's public key: [https://github.com/realsenseai/librealsense/blob/master/doc/distribution_linux.md#installing-the-packages] 

2. installed pre-compiled SDK:
```
sudo apt-get install librealsense2-utils
sudo apt-get install librealsense2-dev
```

3. Sanity check SDK with sample realsense app
```
git clone https://github.com/IntelRealSense/librealsense.git
cd librealsense/example/sample
g++ -std=c++11 filename.cpp -lrealsense2
./a.out
```
4. Connect Intel RealSense Deapth camera (D435) to USB-C port on Jetson and check usb devices:
```
nvidia@ubuntu:~/librealsense/tools$ lsusb
Bus 002 Device 002: ID 8086:0b07 Intel Corp. Intel(R) RealSense(TM) Depth Camera 435   
Bus 002 Device 001: ID 1d6b:0003 Linux Foundation 3.0 root hub
Bus 001 Device 004: ID 0461:4e66 Primax Electronics, Ltd USB 2.0 Hub
Bus 001 Device 003: ID 258a:0001  
Bus 001 Device 002: ID 1a40:0101 Terminus Technology Inc. Hub
Bus 001 Device 001: ID 1d6b:0002 Linux Foundation 2.0 root hub
```

5. Start realsense-viewer app:
```
cd ~/librealsense/tools
realsense-viewer
```
https://github.com/ShonCamarlinghi/tiger_watch/issues/1#issue-4864595931

