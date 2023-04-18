# FPGA——采集与显示

## 一.VGA

### 一、原理

##### 1.简介与接口

VGA，英文全称”Video Graphics Array”，译为视频图形阵列，是一种使用**<font color=Crimson>模拟信号</font>**进行视频传输的标准协议，由IBM公司于1987年推出，因其分辨率高、显示速度快、颜色丰富等优点，广泛应用于彩色显示器领域。

![image-20230416103504078](https://raw.githubusercontent.com/Noregret327/picture/master/202304161035152.png)

##### 2.显示原理

VGA显示器采用<font color=Crimson>图像扫描的方式</font>进行图像显示，将构成图像的像素点，在行同步信号和场同步信号的同步下，按照从上到下、由左到右的顺序扫描到显示屏上。

##### 3.时序标准

![image-20230416103940878](https://raw.githubusercontent.com/Noregret327/picture/master/202304161039910.png)

![image-20230416103959795](https://raw.githubusercontent.com/Noregret327/picture/master/202304161039830.png)

![image-20230416104215063](https://raw.githubusercontent.com/Noregret327/picture/master/202304161042087.png)

![image-20230416104412790](https://raw.githubusercontent.com/Noregret327/picture/master/202304161044833.png)

##### 4.显示模式及相关参数

![image-20230416104620327](https://raw.githubusercontent.com/Noregret327/picture/master/202304161046370.png)

行扫描周期X场扫描周期X帧率=时钟频率

##### 5.VGA

VGA采集的模拟信号，FPGA为数字信号，所以FPGA输出得转换为模拟信号，转换方法有：

* AD7123芯片
* 权电阻网络：RGB565（16位宽）、RGB332（8位宽）、RGB888（24位宽）

### 二、实践

![image-20230416152610945](https://raw.githubusercontent.com/Noregret327/picture/master/202304161526971.png)