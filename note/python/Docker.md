# Docker

## ！！！安装！！！

### 1.在Centos上安装

- linux内核版本依赖

- - kernel version >= 3.8
  - 可以使用如下命令查看
  - `uname -a | awk '{split($3, arr, "-"); print arr[1]}'`

- 如果已安装过Docker, 需要移除老版本的Docker

```text
sudo yum remove docker \
                  docker-client \
                  docker-client-latest \
                  docker-common \
                  docker-latest \
                  docker-latest-logrotate \
                  docker-logrotate \
                  docker-engine
```



- 添加Docker repository yum源

```text
# 国内源, 速度更快, 推荐
sudo yum-config-manager \
    --add-repo \
    https://mirrors.ustc.edu.cn/docker-ce/linux/centos/docker-ce.repo


# 官方源, 服务器在国外, 安装速度慢
# $ sudo yum-config-manager \
#     --add-repo \
#     https://download.docker.com/linux/centos/docker-ce.repo
```



- 开始安装Docker Engine

```text
sudo yum makecache fast
sudo yum install docker-ce docker-ce-cli containerd.io
```



- 开启Docker

```text
sudo systemctl enable docker
sudo systemctl start docker
```



- 验证是否安装成功

```text
sudo docker run hello-world
```

- 如果出现"Hello from Docker.", 则代表运行成功



- 如果在每次运行docker命令是, 在前面不添加sudo, 可以执行如下命令:

```text
sudo usermod -aG docker $USER
```



- 如果嫌上面安装步骤麻烦, 可以运行如下脚本来安装

- - 不能在生产系统中使用

```text
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh --mirror Aliyun

sudo systemctl enable docker
sudo systemctl start docker

sudo groupadd docker
sudo usermod -aG docker $USER
```

### 2.在Ubuntu上安装

- linux内核版本依赖

- - kernel version >= 3.8
  - 可以使用如下命令查看
  - `uname -a | awk '{split($3, arr, "-"); print arr[1]}'`



- 操作系统依赖, 如下版本都可以

```text
Disco 19.04
Cosmic 18.10
Bionic 18.04 (LTS)
Xenial 16.04 (LTS)
```

- 如果已安装过Docker, 需要移除老版本的Docker

```text
sudo apt-get remove docker docker-engine docker.io containerd runc
```



- 使用Docker repository 来安装

```text
# 更新apt包索引
sudo apt-get update

# 为支持https
sudo apt-get install \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg-agent \
    software-properties-common

# 添加Docker GPG秘钥
# 国内源
curl -fsSL https://mirrors.ustc.edu.cn/docker-ce/linux/ubuntu/gpg | sudo apt-key add -
# 或者国外源
# curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -

# 添加安装源
# 推荐国内源
sudo add-apt-repository \
    "deb [arch=amd64] https://mirrors.ustc.edu.cn/docker-ce/linux/ubuntu \
    $(lsb_release -cs) \
    stable"
# 或者国外源
# sudo add-apt-repository \
#   "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
#   $(lsb_release -cs) \
#   stable"
```

- 安装Docker

```text
# 更新apt包索引
sudo apt-get update

# 安装docker
sudo apt-get install docker-ce docker-ce-cli containerd.io
```

- 开启Docker

```text
sudo systemctl enable docker
sudo systemctl start docker
```

- 验证是否安装成功

```text
sudo docker run hello-world
```

- 如果出现"Hello from Docker.", 则代表运行成功
- 如果在每次运行docker命令是, 在前面不添加sudo, 可以执行如下命令:

```text
sudo usermod -aG docker $USER
```



### 3、使用shell脚本安装Docker（Ubuntu、Centos）

**install_docker_.sh**

```text
#!/bin/bash
#Author: 柠檬班可优
#Date: 2019-06-06
#install docker in ubuntu and centos


function install_docker_in_ubuntu
{
    sudo  apt-get update -y
    # install some tools
    sudo apt-get install \
        apt-transport-https \
        ca-certificates \
        curl \
        gnupg-agent \
        software-properties-common \
        net-tools \
        wget -y

    # install docker
    curl -fsSL get.docker.com -o get-docker.sh
    sh get-docker.sh

    # start docker service
    sudo groupadd docker &> /dev/null
    sudo gpasswd -a "${USER}" docker
    sudo systemctl start docker

    rm -rf get-docker.sh
}


function install_docker_in_centos
{
    # install some tools
    sudo yum install -y git vim gcc glibc-static telnet bridge-utils

    # install docker
    curl -fsSL get.docker.com -o get-docker.sh
    sh get-docker.sh

    # start docker service
    sudo groupadd docker &> /dev/null
    sudo gpasswd -a "${USER}" docker
    sudo systemctl start docker

    rm -rf get-docker.sh

}


SYSTEM_NAME="$(awk -F= '/^NAME/{print $2}' /etc/os-release)"
if [[ "${SYSTEM_NAME,,}" =~ "ubuntu"  ]] ; then
    echo "Your system is ubuntu."
    echo "Installing Docker in ubuntu..."
    install_docker_in_ubuntu
elif [[ "${SYSTEM_NAME,,}" =~ "centos" ]] ; then
    echo "Your system is centos."
    echo "Installing Docker in centos..."
    install_docker_in_centos
else
    echo "This script can only run in ubuntu and centos system."
    exit 1
fi
```

**2.运行脚本**

- `bash install_docker_.sh`



## 一 镜像源

|     镜像加速器      |            镜像加速器地址            |
| :-----------------: | :----------------------------------: |
| Docker 中国官方镜像 |    https://registry.docker-cn.com    |
|   DaoCloud 镜像站   |    http://f1361db2.m.daocloud.io     |
|   Azure 中国镜像    |      https://dockerhub.azk8s.cn      |
|     科大镜像站      |  https://docker.mirrors.ustc.edu.cn  |
|       阿里云        | https://ud6340vz.mirror.aliyuncs.com |
|       七牛云        |     https://reg-mirror.qiniu.com     |
|       网易云        |     https://hub-mirror.c.163.com     |
|       腾讯云        |  https://mirror.ccs.tencentyun.com   |

## 二 命令

### 1 docker-compose常用命令-详解

查找命令

```powershell
docker-compose --help
```

docker-compose命令的基本使用格式是：

```powershell
docker-compose [-f=<arg>...] [options] [COMMAND] [ARGS...]
```

命令选项

- -f, --file FILE 指定使用的 Compose 模板文件，默认为 docker-compose.yml，可以多次指定；
- -p, --project-name NAME 指定项目名称，默认将使用所在目录名称作为项目名；

- --x-networking 使用 Docker 的可拔插网络后端特性；

- --x-network-driver DRIVER 指定网络后端的驱动，默认为 bridge；

- --verbose 输出更多调试信息；
- -v, --version 打印版本并退出；

常用命令

```powershell
docker-compose 命令 --help                     获得一个命令的帮助
docker-compose up -d nginx                     构建启动nignx容器
docker-compose exec nginx bash                 登录到nginx容器中
docker-compose down                            此命令将会停止 up 命令所启动的容器，并移除网络
docker-compose ps                              列出项目中目前的所有容器
docker-compose restart nginx                   重新启动nginx容器
docker-compose build nginx                     构建镜像 
docker-compose build --no-cache nginx          不带缓存的构建
docker-compose top                             查看各个服务容器内运行的进程 
docker-compose logs -f nginx                   查看nginx的实时日志
docker-compose images                          列出 Compose 文件包含的镜像
docker-compose config                          验证文件配置，当配置正确时，不输出任何内容，当文件配置错误，输出错误信息。 
docker-compose events --json nginx             以json的形式输出nginx的docker日志
docker-compose pause nginx                     暂停nignx容器
docker-compose unpause nginx                   恢复ningx容器
docker-compose rm nginx                        删除容器（删除前必须关闭容器，执行stop）
docker-compose stop nginx                      停止nignx容器
docker-compose start nginx                     启动nignx容器
docker-compose restart nginx                   重启项目中的nignx容器
docker-compose run --no-deps --rm php-fpm php -v   在php-fpm中不启动关联容器，并容器执行php -v 执行完成后删除容器
```

