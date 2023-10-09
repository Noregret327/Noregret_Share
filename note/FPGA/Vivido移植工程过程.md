# Vivido移植工程过程

## 1.vivido端

1.打开工程文件

2.点击“Generate Bitstream”生成bit文件

3.点击“Open Hardware Manager”烧写“.bit”文件



## 2.Vitis端

### 打开Vitis遇到的问题：

#### 1.New Vitis IDE Available！

![image-20231009162307662](https://raw.githubusercontent.com/Noregret327/picture/master/202310091623695.png)

#### 解决方法：

1. 选择Don‘t show this again
2. 点击No

### 2.Failed to execute command platform read

![image-20231009162238915](https://raw.githubusercontent.com/Noregret327/picture/master/202310091622990.png)

#### 解决方法：

1. 点击OK即可
2. 如果再弹出下面的“Platform Invalid”，点击“Add platform to repository”（如下图1）
3. 然后会自动打开“Platform Repositories”界面（如下图2）
4. 点击➕添加当前新工程的vitis文件路径进去即可（如下图2）
5. <font color= red>如果没有“Platform Repositories”，可以自己在Vitis软件中搜索“Platform Repositories”打开继续添加路径</font>

![image-20231009162404147](https://raw.githubusercontent.com/Noregret327/picture/master/202310091624168.png)

![image-20231009162453957](https://raw.githubusercontent.com/Noregret327/picture/master/202310091624996.png)

### 3.编译工程文件

1.右键工程项目，点击“Clean Project”

2.右键工程项目，点击“Build Project”

3.点击运行即可烧写