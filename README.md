# SimToReal

## Installation / Setup
Note: This repo has only been tested/used on Linux Ubuntu and may have limited/broken functionality on other operating systems

### 1. Clone SimToReal GitHub repository
```
git clone https://github.com/CreativeNick/SimToReal.git
```

### 2. Clone ManiSkill GitHub repository
Visit the [ManiSkill repository](https://github.com/haosulab/ManiSkill) and follow the installation instructions. Be sure to install ManiSkill inside the `SimToReal/` folder.

### 3. Download YCB Assets
This repo uses the Yale-CMU-Berkeley (YCB) Object and Model set, which you can learn more about [here](https://www.ycbbenchmarks.com/). To download the entire asset folder, follow the instructions found in the [ycb-tools repository](https://github.com/sea-bass/ycb-tools). Be sure to install the asset folder inside the `SimToReal/` folder.

### 4. Download other libraries
There may be other libraries/programs you may need to install before running certain commands, such as [tyro](https://pypi.org/project/tyro/) in `ppo.py`.
```
pip install tyro
```
FFmpeg ImageIO are both required to capture and render videos for both training and evaluation. In case you already pip installed both of these, make sure you have the right version (Only run the second line if you do not have both installed).
```
pip uninstall "imageio-ffmpeg" "imageio"
pip install "imageio-ffmpeg"=="0.4.9" imageio=="2.34.0"
```
