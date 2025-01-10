git clone https://github.com/NVIDIA-AI-IOT/Lidar_AI_Solution.git && cd Lidar_AI_Solution && git checkout 8a1b2962 && cd ..
mkdir -p ./lib/cuOSD && cp -R Lidar_AI_Solution/libraries/cuOSD/src/* ./lib/cuOSD
rm -rf Lidar_AI_Solution

cd ./lib
git clone https://github.com/nothings/stb.git && cd stb && git checkout 5c205738 && cd ..

cd .. 
wget -N https://github.com/NVIDIA-AI-IOT/Lidar_AI_Solution/raw/refs/heads/master/libraries/cuOSD/data/simhei.ttf -P demo/
