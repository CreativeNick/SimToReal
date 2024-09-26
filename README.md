# SimToReal

## Installation / Setup
Note: This repo has only been tested/used on Linux Ubuntu and may have limited/broken functionality on other operating systems

### 1. Clone SimToReal GitHub repository
```
git clone https://github.com/CreativeNick/SimToReal.git
```

### 2. Clone ManiSkill GitHub repository
Visit the [ManiSkill repository](https://github.com/haosulab/ManiSkill) and follow the installation instructions. Be sure to clone/install ManiSkill inside the `SimToReal/` folder. It's recommened to do this in a virtual environment. \
In summary:
> Create virtual environment  
Run `pip install --upgrade mani_skill` to install ManiSkill package  
Run `pip install torch torchvision torchaudio` to install Torch  
Run `sudo apt-get install libvulkan1` to install Vulkan  
Run `sudo apt-get install vulkan-utils` followed by `vulkaninfo` to test Vulkan installation


## Optional Setup / Installation
There are some libraries, assets, etc you may need to install when trouble-shooting. If the above installation works for you, then you may ignore this optional setup.

### 1. Download YCB Assets
This repo uses the Yale-CMU-Berkeley (YCB) Object and Model set, which you can learn more about [here](https://www.ycbbenchmarks.com/). To download the entire asset folder, follow the instructions found in the [ycb-tools repository](https://github.com/sea-bass/ycb-tools). Be sure to install the asset folder inside the `SimToReal/` folder.

### 2. Download other libraries
There may be other libraries/programs you may need to install before running certain commands, such as [tyro](https://pypi.org/project/tyro/) in `ppo.py`.
```
pip install tyro
```
FFmpeg ImageIO are both required to capture and render videos for both training and evaluation. In case you already pip installed both of these, make sure you have the right version (Only run the second line if you do not have both installed).
```
pip uninstall "imageio-ffmpeg" "imageio"
pip install "imageio-ffmpeg"=="0.4.9" imageio=="2.34.0"
```
