# Cyclone V SOC FPGA

## 一.SOC基本概念

##### 1.简介

SOC FPGA：同一芯片集成FPGA和处理器

##### 2.分类

* Cyclone Ⅴ SE：**Stratix V** —— 925MHz
* Xilinx：zynq——667MHz 800MHz

##### 3.优点

既拥有ARM灵活有效的数据运算和事务处理能力，又集成了FPGA的高速并行处理优势。

###### HPS（Hardware Processor System）

* MPU（存储单元）
* DDR3（高性能硬件处理系统）
* Nand FLASH

##### 4.资源

1、FPGA逻辑单元（LE）——25K

2、ARM处理器——175KB

3、DSP——36个

4、模拟锁相环（PLL）——5个

5、数字锁相环（DLL）——4个

6、HPS——速度等级：7；理论主频：800MHz；实际能到：925MHz

7、DDR3控制器——800MHz

8、千兆以太网MAC控制器——2个

9、NAND FLASH

10、QSPI FLASH

11、SD/MMC

12、SPI主机控制器——2个

13、SPI从机控制器——2个

14、USB——1个

15、UART——2个

16、I2C——4个

17、CAN——2个

**其中1-4为FPGA资源、5-17为HPS资源**



## 二.工具

##### 1.软件

1、Quartus 13.1以上——推荐最新

2、Platform Designer

3、SoC EDS——程序开发

4、DS-5 AE

5、Putty——连接开发板

6、winscp



##### 2.Linux驱动程序开发

* 有现成的程序框架
* 侧外设驱动开发



## 三.GHRD工程

GHRD：Golden Hardware Reference Design

![image-20230411145431750](https://raw.githubusercontent.com/Noregret327/picture/master/202304111454770.png)



## 四.DS5编写C程序

##### 流程

###### 1.打开SoC EDS

```
eclipse&
```

###### 2.新建工程

C Project ——> name ——> Toolchains（GCC 4.x） ——> Next —— > Finsh

###### 3.新建文件

New Source File ——> folder（hello_word） ——> file（main.c）——> Finsh

###### 4.编译文件

编译：main.c

```c
#include <stdio.h>

int main(int argc, char* argv[])
{
	printf("hello\n");
	return 0;
}
```

**记得保存之后再编译！**

###### 5.运行编译

右键工程 ——> build Project（或者快捷键：Ctrl + B）

###### 6.制作开发板Linux系统到SD卡

软件：

* win32diskimager

烧写镜像文件

###### 7.在Putty烧写

```cmd
1、连接串口com3、设置115200

2、login：root

3、fdisk -l

4、mount -t vfat /dev/mmcblkopl /mnt

5、cd /mnt

6、ls

7、./hello_world
```



## 五.GDB设置

##### 1.在eclipse platform中

* 打开——>Run——>Debug configurations——>C/C++ Application
* Debugger——>gdbsever
* Main——>GDBdebugger——>arm-linux-gnue.exe
* connection——>TCP（网口）——>address：192.168.xx.xxx

##### 2.在Putty

```cmd
gdbsever：10000 hello_world
```

Linux命令

```cmd
用户名：
root

查看当前板卡的网络地址：
ifconfig

直接设置板卡的网络地址：
ifconfig eth0 192.168.30.199

查看当前目录下的文件名：
ls

查看文件详细信息：
ls -l

添加权限给文件：
chmod 777 hello_world

执行程序：
./hello_world

#输入文件名的时候按下“tab”可以快速补全文件名称

启动调试：
gdbsever ：10000 hello_world

查看文件内容：
cat test.c

删除文件所有txt类型的文件
rm *.txt

重命名文件（test.txt）
mv test.txt test_1.txt

复制文件
cp hello_world hello_world1

```



##### 3.整体流程

1、网络访问开发板：ifconfig etho 192.168.xx.xxx（查看IP：ifconfig）

2、passwd：设置密码

3、拷贝程序到开发板

4、为程序加可执行权限：chmod 777 hello_word

5、DS-5设置调试配置

6、开发板上启动调试：gdbsever ：10000 hello_world

7、在DS-5点击debug开始调试



## 六.虚拟地址映射的Linux硬件编程

* 虚拟机映射
* 使用系统自带的驱动程序直接驱动外设IP

![image-20230411162325963](https://raw.githubusercontent.com/Noregret327/picture/master/202304111623003.png)

虚拟地址映射的Linux硬件编程：

* 虚拟机映射
* 使用系统自带的驱动程序直接驱动外设IP

**学习映射操作**



## 七.SoC系统开发流程

1.修改Platform Designer中的设计，添加IP

2.重新集成Qsys系统文件到Quartus中

3.设定信号的I/O管脚

4.得到sof或rbf文件；soc_system.sopcinfo——>hps_0.h；soc_system.sopcinfo——>soc_system.dts——>socfpga.dtb；

##### 1.添加IP

在Platform Designer中：

* 左上角——>IP Catalog——>搜索需要的IP
* 右边——>System Contents连接IP
* 验证地址——>{(1)自动——先锁定原先程序地址——>点击最上面System——>Assign Base Address}——>(2)手动

* 再复制（Generate——>Instantiation Template）的新添加的例化语句，添加到System中
* 再在Platform Designer中点击右下角的”Generate HDL“
* 在Quartus中添加管脚
* 用sof_to_rbf.bat生成rbf文件



1.用soc通过cd进入对应工程文件

2.make dts		生成“.sopcinfo”文件和dts（设备树）文件

3.make dtb		生成“.dtb”文件

4.拷贝“dtb”文件到SD卡——>并改名为“socfpga.dtb”

##### 2.问题

1.对于dtb文件：只要在Paltform Designer中进行了任何的修改，都需要soc_system.dts——>socfpga.dtb

2.对于Preloader和Uboot来说：在Platform Designer中修改了HPS组件的参数



## 八.设备树

在讲到设备树之前，先看一个具体的应用场景。对于一个 ARM 处理器般其片上都会集成了有较多的外设接口，包括I2C、SPI、UART 等。而I2C、SPI 都属于总线属性，在这些总线上又会连接其他的外部器件。例如在I2C 总线上，又会连接 EEPROM、I2C 接口的各种传感器、LCD 显示屏、RTC等。那么 Linux 系统如何能够知道 I2C 总线上连接了哪些设备，又如何知道这些设备的具体信息呢？

为了解决这个问题，Linux 内核从 3.x 开始引入设备树的概念，用于实现驱动代码与设备信息相分离。在设备树出现以前，所有关于设备的具体信息都要写在驱动里，一旦外围设备变化，驱动代码就要重写。引入了设备树之后，驱动代码只负责处理驱动的逻辑，而关于设备的具体信息存放到设备树文件中，这样，如果只是硬件接口信息的变化而没有驱动逻辑的变化,驱动开发者只需要修改设备树文件信息，不需要改写驱动代码。

##### 设备树基本格式

设备树用树状结构描述设备信息，它有以下几种特性

1.每个设备树文件都有一个根节点，每个设备都是一个节点。

2.节点间可以嵌套，形成父子关系，这样就可以方便的描述设备间的关系。

3.每个设备的属性都用一组 key-value 对(键值对)来描述

4.每个属性的描述用:结束

######  soc system.dts 文件：用于描述设备树的

“/”表示一个硬件平台

* model：描述产品型号
* compatible：兼容属性
* height、width、brightness：描述板子某些专用硬件的一些物理信息
* cpus：cpu节点，包含hps_0_arm_a9_0和hps_0_arm_a9_1两个子节点
* sopc0：有hps_0_bridges，其中它包含了i2c_0、sysid、uart...

##### 设备树目的

这些节点所代表的设备正是我们在 Platform Designer中添加的 FPGA 侧的 I。因此，如果我们在 FPGA 侧增加、删除、修改了某些IP，然后使用 SOC EDS 软件重新生成 dts 文件，这些变化也都会体现在hps_O_bridges 节点下。例如我们修改添加的 uart 1 控制器的默认波特率为9600bps，然后重新生成 dts 文件，则可以看到 dts 文件中 uart 1 节点下的 current-speed 属性值会从 115200 变为 9600。

对于一个特定的设备节点，例如 alt_vip_vfr tft，又有众多的属性描述来该节点的详细信息，用来提供给 Limux 系统用作设备驱动中需要根据硬件具体设置而修改的一些可变信息。



## 九.Linux内核

arm-linux-qnueabihf



切换到root用户下：su

内核源码位于 /home/xiaomeige/linux-socfpga

加载已有的内校配配置 : make socfpga defconfig

打开内核配置的图形界面: make menuconfig

保存配置好的内核信息: make savedefconfig && mv defconfig arch/arm/configs/hbmd defconfig





切换到root用户下 su  /su xiaomeige

添加环境变量: source /home/xiaomeige/.profile

设置目标架构: export ARCH=arm

设置具体的编译工具链: export CROSS_COMPILE=arm-linux-gnueabihf-

确保当前位于内核目录下（/home/xiaomeige/linux-socfpga）

开始编译内核：make



切换到root用户下 su  /su xiaomeige

添加环境变量: source /home/xiaomeige/.profile

设置目标架构: export ARCH=arm

设置具体的编译工具链: export CROSS_COMPILE=arm-linux-gnueabihf-

确保当前位于驱动程序目录下

开始编译内核：make



打开终端：

ctrl + alt + t



内核位置：

cd arch/arm/boot/zimage
