# Python环境安装及编译过程中的一些问题

### 一：Pytorch

#### 官网

[PyTorch](https://pytorch.org/)

清华源网站：

[Index of /anaconda/cloud/pytorch/win-64/ | 清华大学开源软件镜像站 | Tsinghua Open Source Mirror](https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/win-64/)

#### 检测Pytorch是否安装成功

在Python环境中输入

```python
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.backends.cudnn.is_available())
print(torch.cuda_version)
print(torch.backends.cudnn.version())
```

#### Anaconda的一些命令

创建新环境（例：pytorch，python版本为3.8）

```
conda create -n pytorch python=3.8
```

查看已有环境

```
conda env list
```

查看当前环境所有包

```
conda list
```

激活对应的环境(例：pytorch环境)

```
conda activate pytorch
```

关闭当前环境

```
conda deactivate
```

删除环境(例：pytorch环境)

```
conda remove -n pytorch --all
```

#### 给环境换个镜像源进行安装，官网的很慢

```
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/
conda config --set show_channel_urls yes
```

#### 安装Pytorch

在pytorch官网搜索对应的CUDA Version，不能比自己的CUDA Version高

在Anaconda的Pytorch环境下，输入

```
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
```

#### 卸载Pytorch

在Anaconda中，输入

```
conda uninstall pytorch
```

#### 检测显卡的Driver Version和CUDA Version

一定在下载显卡驱动才能检测！

在CMD中，输入命令：

```
powershell
```

在powershell下，输入命令：

```
nvidia-smi
```



### 二：PaddlePaddle

#### 官网

[飞桨PaddlePaddle-源于产业实践的开源深度学习平台](https://www.paddlepaddle.org.cn/)



### 三：Tensorflow

#### 环境创建与安装

在Anaconda环境下

```
conda create -n tensorflow python=3.8
```

进入tensorflow环境

```
conda activate tensorflow
```

下载tensorflow

```
conda install tensorflow
```

#### 环境测试

在tensorflow环境下，进入python环境

```
python
```

测试tensorflow（没报错表示配置成功）

```
import tensorflow
```

退出python环境

```
exit()
```

#### 环境问题





### 四、各个环境安装命令

#### 1、skimage

在Anaconda环境下

```
conda install scikit-image
```

如果不行就下面这个

```
pip install -U scikit-image
```

#### 2、dlib

方法一：

在Anaconda中搜索

```
anaconda search -t conda dlib
```

再输入下面命名

```
conda install -c https://conda.anaconda.org/conda-forge dlib
```

方法二：

在Anaconda环境下

```
pip install dlib==19.21.1
```

方法三：

在Anaconda环境下

```
conda install -c menpo dlib=18.18
```

#### 3、sklearn

原名是：scikit-learn

在Anaconda环境下

```
pip install scikit-learn
```

使用douban进行安装

```
pip install sklearn -i https://pypi.doubanio.com/simple
```



### 五、OpenCV

在Anaconda Prompt下，输入

```
pip install opencv-python
```

PS：conda中没有opencv，用pip下载





### 六：初次无法激活环境的问题

在CMD下，输入

```
activate
```

再继续输入(env：为要激活的环境名字)

```
conda activate env
```



### 七：关于Numpy的问题

#### 在CMD下，安装最新的pip

```
python -m pip install --upgrade pip
```

#### 在Anaconda Prompt下，(可以用conda，也可以用pip)

查看所有包

```
conda list
```

安装Numpy

```
conda install numpy
```

卸载所有Numpy

```
conda uninstall numpy
```

查看Numpy的版本

```
pip show numpy
```

卸载所以的Numpy

```
pip uninstall numpy
```

安装最新的Numpy

```
pip install numpy
```

安装某个版本的Numpy(例：Numpy版本为1.16.4)

```
pip install numpy==1.16.4
```

直接安装tensorflow

```
pip install --upgrade --ignore-installed tensorflow
```

#### 更改某个版本numpy

在Anaconda环境下，进入对应的环境（以tensorflow环境为例）

```
conda activate tensorflow
```

先查看numpy版本

```
conda list
```

卸载numpy

```
conda uninstall numpy
```

安装相应版本numpy

```
conda install numpy==1.23.0
```

或用pip安装

```
pip install numpy==1.23.0
```



### 八：编译过程中遇到的问题

#### **Requirement already satisfied**的问题

问题：

Requirement already satisfied: opencv-python in d:\users\nogre\appdata\local\programs\python\python38\lib\site-packages (4.6.0.66) Requirement already satisfied: numpy>=1.14.5 in d:\users\nogre\appdata\local\programs\python\python38\lib\site-packages

解决：

在CMD中重新安装就行，但要指定环境路径：

```
pip install --target=要安装的环境路径 安装包名
```

以安装opencv-python为例

```
pip install --target=D:\anaconda3\envs\tensorflow\Lib opencv-python
```



    from tensorflow.python import tf2

#### ImportError的问题

问题：

from tensorflow.python import tf2

ImportError: cannot import name 'tf2' from 'tensorflow.python' (unknown location)

解决：

1.可能版本问题，通过版本表来卸载和安装对应版本

2.也可能没安装tensorflow的环境包，直接在Anaconda的tensorflow环境下安装就行

```
conda install tensorflow
```

#### AttributeError: module ‘tensorflow‘ has no attribute ‘get_default_graph‘的问题

问题：

AttributeError: module ‘tensorflow‘ has no attribute ‘get_default_graph‘

原始代码：

```
tf.get_default_graph()
```

修改代码：

```
tf.compat.v1.get_default_graph()
```

#### RuntimeError: Found dtype Long but expected Float的问题

问题：

RuntimeError: Found dtype Long but expected Float

原始代码：

```
loss_fn(model(X), Y)
```

修改代码：（在数据后面加‘.float（）’）

```
loss_fn(model(X).float(), Y.float())
```
