# Ubuntu18

## 配环境

#### 1.修改root密码

- 设置root密码

```
sudo passwd root
```

成功后显示如下：

```
[sudo] password for kerwin: #输入当前用户密码
New UNIX password: #输入root新密码
Retype new UNIX password: #再次输入root密码
passwd: password updated successfully #密码更新成功
```

- 进入root

```
su root
```

- 进入用户（rog——改为自己用户名）

```
su rog
```



#### 2.安装pip

- 先更新apt，再安装

```
sudo apt update
```

- 安装python3/2

```
sudo apt install python3-pip
sudo apt install python2
```

- 安装pip

```
sudo apt install python-pip
```



#### 3.安装gcc/g++/make/wget/unzip

- 安装前先把apt更新到最新

```
sudo apt-get install gcc g++ make wget unzip libopencv-dev pkg-config
```





## 报错

1.运行apt报错如下：

```
E: Could not get lock /var/lib/dpkg/lock-frontend - open (11: Resource temporarily unavailable)
E: Unable to acquire the dpkg frontend lock (/var/lib/dpkg/lock-frontend), is another process using it?
```

解决方法：

```
sudo rm /var/lib/dpkg/lock
```

或

```
sudo rm /var/lib/dpkg/lock-frontend
sudo rm /var/cache/apt/archives/lock
```

