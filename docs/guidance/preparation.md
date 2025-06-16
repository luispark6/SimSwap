
# Preparation

### Installation
**We highly recommand that you use Anaconda for Installation**  
**Download Below for CPU face detection model build:**
```
conda create -n simswap python=3.6
conda activate simswap
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=10.2 -c pytorch
(option): pip install --ignore-installed imageio
pip install insightface==0.2.1 onnxruntime moviepy
(option): pip install onnxruntime-gpu  (If you want to reduce the inference time)(It will be diffcult to install onnxruntime-gpu , the specify version of onnxruntime-gpu may depends on your machine and cuda version.)
pip install line_profiler
pip3 install -U timm==0.5.4
pip install tensorboard==2.19.0
```
**Download Below for GPU face detection model build:**
```
conda create -n simswap python=3.12 #Any 3.1x should work

conda activate simswap

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126 #onnxruntime-gpu requires cuda 12.x and cudnn 9.x

pip install insightface==0.2.1 onnxruntime-gpu==1.22.0 moviepy

pip install line_profiler

pip install tensorboard==2.19.0

```
**If you are using GPU for face detection, you must follow this next step:**
```
In the folder below, go the model_zoo.py file

'C:\Users\<Your Username>\.conda\envs\simswap\Lib\site-packages\insightface\model_zoo'

In this file, lines 21-22 should look like the folowing:

21  def get_model(self):
22      session = onnxruntime.InferenceSession(self.onnx_file, None)

Replace line 22 as follows:

21  def get_model(self):
22      EP_list = ['CUDAExecutionProvider', 'CPUExecutionProvider']
23      session = onnxruntime.InferenceSession(self.onnx_file, providers= EP_list)


You should be set now!!
```



- We use the face detection and alignment methods from **[insightface](https://github.com/deepinsight/insightface)** for image preprocessing. Please download the relative files and unzip them to ./insightface_func/models from [this link](https://drive.google.com/file/d/1goH5lO8BAhTpRhpBeXqWEcGkxiiLlgx9/view).
- We use the face parsing from **[face-parsing.PyTorch](https://github.com/zllrunning/face-parsing.PyTorch)** for image postprocessing. Please download the relative file and place it in ./parsing_model/checkpoint from [this link](https://drive.google.com/file/d/154JgKpzCPW82qINcVieuPH3fZ2e0P812/view).
- The pytorch and cuda versions above are most recommanded. They may vary.
- Using insightface with different versions is not recommanded. Please use this specific version.
- These settings are tested valid on both Windows and Ubuntu.

### Pretrained model
There are two archive files in the drive: **checkpoints.zip** and **arcface_checkpoint.tar**

- **Copy the arcface_checkpoint.tar into ./arcface_model**
- **Unzip checkpoints.zip, place it in the root dir ./**

[[Google Drive]](https://drive.google.com/drive/folders/1jV6_0FIMPC53FZ2HzZNJZGMe55bbu17R?usp=sharing)
[[Baidu Drive]](https://pan.baidu.com/s/1wFV11RVZMHqd-ky4YpLdcA) Password: ```jd2v```

**Simswap 512 (optional)**

The checkpoint of **Simswap 512 beta version** has been uploaded in [Github release](https://github.com/neuralchen/SimSwap/releases/download/512_beta/512.zip).If you want to experience Simswap 512, feel free to try.
- **Unzip 512.zip, place it in the root dir ./checkpoints**.


### Note
We expect users to have GPU with at least 3G memory. For those who do not, we provide [[Colab Notebook implementation]](https://colab.research.google.com/github/neuralchen/SimSwap/blob/main/SimSwap%20colab.ipynb).
