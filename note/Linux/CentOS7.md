 

# CentOS7

## 一.装系统

直接把虚拟机文件复制到一个路径下，打开VMware虚拟机，扫描添加进去即可



## 二.编译Demo

#### 1.设置环境变量

```
export PATH=/opt/software/gcc-linaro-5.4.1-2017.05-x86_64_arm-linux-gnueabihf/bin:$PATH
export AIEP_HOST=172.16.208.134
```

centos 7编译出现 cmake：The CXX compiler identification is unknown

  Tell CMake where to find the compiler by setting either the environment  variable "CXX" or the CMake cache entry CMAKE_CXX_COMPILER to the full path  to the compiler, or to the compiler name if it is in the PATH.

##### Centos7升级gcc

###### 安装centos-release-scl

```
sudo yum install centos-release-scl
```

###### 安装devtoolset

```
sudo yum install devtoolset-7 bash
```

###### 激活devtoolset

```
scl enable devtoolset-7 bash
```

#### 2.编译Demo（在nna环境下）

```
sudo ./build.sdk.sh
```

#### 3.运行paddle（在nna环境下）

```
sudo ./build.paddlelite.sh
```

#### 4.运行build（在nna环境下）

```
sudo ./build.demo.sh
```

#### 5.上传Demo（在nna环境下）

```
export AIEP_HOST=172.16.208.134
./deploy2aiep.sh
```

#### 6.执行Demo程序

```
cd /opt/paddle_frame
chmod 777 run.sh
./run.sh
```

* 执行结束

* 可见检测到3个目标，推理时间约601ms
* 通过Mobax查看图片

#### 7.查看图片

```
在MobaXterm中将
/opt/paddle_frame
```



## 三.Paddle_Lite

###### Paddle-lite-Demo

[PaddlePaddle/Paddle-Lite-Demo at master (github.com)](https://github.com/PaddlePaddle/Paddle-Lite-Demo/tree/master)

###### Paddle Lite端侧部署

[Paddle Lite端侧部署 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/350505340)







## 四.装Cmake

步骤一、安装gcc等必备程序包（已安装则略过此步）

```
yum install -y gcc gcc-c++ make automake 
```

步骤二、安装wget（已安装则略过此步）

```
yum install -y wget
```

步骤三、获取CMake源码包

```
wget https://cmake.org/files/v3.10/cmake-3.10.3.tar.gz
```

步骤四、解压CMake源码包

```
tar -zxvf cmake-3.10.3.tar.gz
```

步骤五、进入目录

```
cd cmake-3.10.3
```

步骤六 编译安装

```
./bootstrap && make -j4 && sudo make install
```





## 五.装gcc

查看版本号

```
gcc --version
g++ --version
```

或者

```
gcc -v
g++ -v
```



## 六.装交叉编译器

* 例：gcc-linaro-5.4.1-2017.05-x86_64_arm-linux-gnueabihf

```
sudo wget https://releases.linaro.org/components/toolchain/binaries/5.4-2017.01/arm-linux-gnueabihf/gcc-linaro-5.4.1-2017.01-x86_64_arm-linux-gnueabihf.tar.xz
```

先打开文本框

```
sudo vim ~/.bashrc
```

再在文本框下面添加环境变量，如下，添加完保持并关闭

```
export PATH=$PATH:/opt/software/gcc-linaro-5.4.1-2017.05-x86_64_arm-linux-gnueabihf/bin
```

最后使环境变量生效

```
source ~/.bashrc
```

查看版本

```
arm-linux-gnueabihf-gcc -v
```



## 错误汇总

* 错误1

当我们使用 wget下载不安全的https 域名下的内容时会提示下面内容：

```
ERROR: cannot verify releases.linaro.org's certificate, issued by ‘/C=US/O=Let's Encrypt/CN=R3’:
  Issued certificate has expired.
To connect to releases.linaro.org insecurely, use `--no-check-certificate'.
```

* 解决

```
sudo yum install -y ca-certificates
```

