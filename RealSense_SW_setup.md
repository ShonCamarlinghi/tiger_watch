Jetson SDK: JetPack 5.1.6 | [https://developer.nvidia.com/embedded/jetpack-sdk-516] | 
| only available for host Ubuntu version 20.04 or less
 
git clone https://github.com/IntelRealSense/librealsense.git
cd librealsense
mkdir build && cd build
cmake ..
make -j4
sudo make install
