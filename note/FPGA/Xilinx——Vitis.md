# Xilinx——Vitis

## 1.helloworld

本章的实验任务是在领航者 ZYNQ开发板上搭建 ZYNQ嵌入式最小系统，并使用串口打印“ Hello World”信息。

**ZYNQ嵌入式最小系统**

![image-20231017145729643](https://raw.githubusercontent.com/Noregret327/picture/master/202310171457714.png)



## 2.GPIO之MIO控制LED实验

本章的实验任务是使用GPIO通过 三个 MIO引脚 控制 PS端 三个 LED的亮灭 ，实现 核心板上 LED2和底板上 PS_LED1、 PS_LED2三个 LED灯 闪烁的效果 。

### **硬件设计部分：**

![image-20231017145942304](https://raw.githubusercontent.com/Noregret327/picture/master/202310171459334.png)

### **系统框图：**

![image-20231017150031649](https://raw.githubusercontent.com/Noregret327/picture/master/202310171500699.png)

### **PL端修改：**

![image-20231017211124665](https://raw.githubusercontent.com/Noregret327/picture/master/202310172111711.png)

### **PS端程序：**

```c
#include "xparameters.h" 		//器件参数信息
#include "xstatus.h" 			//包含XST_FAILURE和XST_SUCCESS的宏定义
#include "xil_printf.h" 		//包含print()函数
#include "xgpiops.h" 			//包含PS GPIO的函数声明
#include "sleep.h" 				//包含sleep()函数

//宏定义GPIO_DEVICE_ID
#define GPIO_DEVICE_ID XPAR_XGPIOPS_0_DEVICE_ID
//连接到MIO的LED
#define MIOLED0 7 				//连接到MIO7
#define MIOLED1 8 				//连接到MIO8
#define MIOLED2 0 				//连接到MIO0

XGpioPs Gpio; 					// GPIO设备的驱动程序实例

int main()
{
	int Status;
	XGpioPs_Config *ConfigPtr;

	print("MIO Test! \n\r");
	ConfigPtr = XGpioPs_LookupConfig(GPIO_DEVICE_ID);
	Status = XGpioPs_CfgInitialize(&Gpio, ConfigPtr,ConfigPtr->BaseAddr);
	if (Status != XST_SUCCESS){
		return XST_FAILURE;
	}
	//设置指定引脚的方向：0输入，1输出
	XGpioPs_SetDirectionPin(&Gpio, MIOLED0, 1);
	XGpioPs_SetDirectionPin(&Gpio, MIOLED1, 1);
	XGpioPs_SetDirectionPin(&Gpio, MIOLED2, 1);
	//使能指定引脚输出：0禁止输出使能，1使能输出
	XGpioPs_SetOutputEnablePin(&Gpio, MIOLED0, 1);
	XGpioPs_SetOutputEnablePin(&Gpio, MIOLED1, 1);
	XGpioPs_SetOutputEnablePin(&Gpio, MIOLED2, 1);

	while(1)
	{
		XGpioPs_WritePin(&Gpio, MIOLED0, 0x0); //向指定引脚写入数据：0或1
		XGpioPs_WritePin(&Gpio, MIOLED1, 0x0);
		XGpioPs_WritePin(&Gpio, MIOLED2, 0x0);
		sleep(1); //延时1秒
		XGpioPs_WritePin(&Gpio, MIOLED0, 0x1);
		XGpioPs_WritePin(&Gpio, MIOLED1, 0x1);
		XGpioPs_WritePin(&Gpio, MIOLED2, 0x1);
		sleep(1); //延时1秒
		XGpioPs_WritePin(&Gpio, MIOLED0, 0x0); //向指定引脚写入数据：0或1
		XGpioPs_WritePin(&Gpio, MIOLED1, 0x0);
		XGpioPs_WritePin(&Gpio, MIOLED2, 0x0);
		sleep(1); //延时1秒

		XGpioPs_WritePin(&Gpio, MIOLED0, 0x0); //向指定引脚写入数据：0或1
		sleep(1); //延时1秒
		XGpioPs_WritePin(&Gpio, MIOLED0, 0x1);
		sleep(1); //延时1秒
		XGpioPs_WritePin(&Gpio, MIOLED1, 0x0);
		sleep(1); //延时1秒
		XGpioPs_WritePin(&Gpio, MIOLED1, 0x1);
		sleep(1); //延时1秒
		XGpioPs_WritePin(&Gpio, MIOLED2, 0x0);
		sleep(1); //延时1秒
		XGpioPs_WritePin(&Gpio, MIOLED2, 0x1);
		sleep(1); //延时1秒
	}
	return XST_SUCCESS;
}

```

## 3.GPIO之EMIO按键控制LED实验

本章的实验任务是使用领航者 ZYNQ底 板上的三个用户按键分别控制 PS端 三个 LED的亮灭。其中一个按键 PL_KEY0连接到了PL端需要通过EMIO进行扩展 ，另外两个按键 分别是底板上PS端的用户按键 PS_KEY0和 PS_KEY1，这两个按键控制 PS_LED0和 PS_LED1，底板上的 PL_KEY0控制核心板上的LED2。三个LED灯在按键按下的时候点亮，释放后熄灭。

PS和外部设备之间的通信主要是通过复用的输入 /输出（ Multiplexed Input/Output MIO）实现的 。 除此之外， PS还可以通过扩展的MIO Extended MIO EMIO）来实现与外部设备的连接。EMIO使用了PL的I/O资源当PS需要扩展超过54个引脚的时候可以用 EMIO 也可以用它来连接PL中实现的IP模块。

### **EMIO简介：**

ZYNQ GPIO接口信号被分成四组，分别是从 BANK0到 BANK3。其中 BANK0和 BANK1中共计 54个信号通过 MIO连接到 ZYNQ器件的引脚上，这些引脚属于 PS端； 而 BANK2和 BANK3中共计 64个信号则通过 EMIO连接到了 ZYNQ器件的 PL端 如下图所示：（GPIO框图）

![image-20231017211647057](https://raw.githubusercontent.com/Noregret327/picture/master/202310172116098.png)

在大多数情况下， PS端经由 EMIO引 出 的接口 会 直接连接到 PL端 的器件引脚上 通过 IO管脚 约束 来指定 所连接 PL引脚 的位置。 通过这 种方式， EMIO可以 为 PS端 实现额外的 64个输入引脚或 64个带有 输出使 能的输出 引脚 。 EMIO还 有一种使用方式，就是用于 连接 PL内实现的功能模块 IP核此时PL端的IP作为 PS端的一个外部设备 如下图所示：（EMIO接口 的使用方式）

![image-20231017211741335](https://raw.githubusercontent.com/Noregret327/picture/master/202310172117375.png)

### **系统框图：**

![image-20231017212021828](https://raw.githubusercontent.com/Noregret327/picture/master/202310172120882.png)

### PL端修改：

![image-20231017212252655](https://raw.githubusercontent.com/Noregret327/picture/master/202310172122692.png)

![image-20231017212406378](https://raw.githubusercontent.com/Noregret327/picture/master/202310172124408.png)

![image-20231017212414405](https://raw.githubusercontent.com/Noregret327/picture/master/202310172124443.png)

![image-20231017212420772](https://raw.githubusercontent.com/Noregret327/picture/master/202310172124797.png)

点击选中该接口 ，在左侧 External Interface Properties一 栏中将该接口的名称修改为 GPIO_EMIO_KEY

![image-20231017212431141](https://raw.githubusercontent.com/Noregret327/picture/master/202310172124164.png)

修改引脚：

![image-20231018193645542](https://raw.githubusercontent.com/Noregret327/picture/master/202310181936589.png)

改完后Ctrl+s保存：

![image-20231018193712486](https://raw.githubusercontent.com/Noregret327/picture/master/202310181937510.png)

再生成Bitstream，之后可以通过“IMPLEMENTED”的“Report Utilization”查看PL资源使用情况：

![image-20231018193932483](https://raw.githubusercontent.com/Noregret327/picture/master/202310181939506.png)

![image-20231018193940229](https://raw.githubusercontent.com/Noregret327/picture/master/202310181939254.png)



最后最后导出硬件信息：

![image-20231018194013882](https://raw.githubusercontent.com/Noregret327/picture/master/202310181940912.png)

![image-20231018194018047](https://raw.githubusercontent.com/Noregret327/picture/master/202310181940072.png)



### PS端修改：

新建vitis：

![image-20231018194047577](https://raw.githubusercontent.com/Noregret327/picture/master/202310181940605.png)

![image-20231018194054637](https://raw.githubusercontent.com/Noregret327/picture/master/202310181940668.png)

![image-20231018194105884](https://raw.githubusercontent.com/Noregret327/picture/master/202310181941908.png)

新建main.c文件：

![image-20231018194124355](https://raw.githubusercontent.com/Noregret327/picture/master/202310181941390.png)

![image-20231018194128229](https://raw.githubusercontent.com/Noregret327/picture/master/202310181941253.png)

程序：

```c
#include "stdio.h"
#include "xparameters.h"
#include "xgpiops.h"

#define GPIOPS_ID XPAR_XGPIOPS_0_DEVICE_ID   //PS端  GPIO器件 ID

#define MIO_LED0 7   //PS_LED0 连接到 MIO7
#define MIO_LED1 8   //PS_LED1 连接到 MIO8
#define MIO_LED2 0   //PS_LED2 连接到 MIO0

#define MIO_KEY0 12  //PS_KEY0 连接到 MIO7
#define MIO_KEY1 11  //PS_KEY1 连接到 MIO8

#define EMIO_KEY 54  //PL_KEY0 连接到EMIO0

int main()
{
    printf("EMIO TEST!\n");

    XGpioPs gpiops_inst;            //PS端 GPIO 驱动实例
    XGpioPs_Config *gpiops_cfg_ptr; //PS端 GPIO 配置信息

    //根据器件ID查找配置信息
    gpiops_cfg_ptr = XGpioPs_LookupConfig(GPIOPS_ID);
    //初始化器件驱动
    XGpioPs_CfgInitialize(&gpiops_inst, gpiops_cfg_ptr, gpiops_cfg_ptr->BaseAddr);

    //设置LED为输出
    XGpioPs_SetDirectionPin(&gpiops_inst, MIO_LED0, 1);
    XGpioPs_SetDirectionPin(&gpiops_inst, MIO_LED1, 1);
    XGpioPs_SetDirectionPin(&gpiops_inst, MIO_LED2, 1);
    //使能LED输出
    XGpioPs_SetOutputEnablePin(&gpiops_inst, MIO_LED0, 1);
    XGpioPs_SetOutputEnablePin(&gpiops_inst, MIO_LED1, 1);
    XGpioPs_SetOutputEnablePin(&gpiops_inst, MIO_LED2, 1);

    //设置KEY为输入
    XGpioPs_SetDirectionPin(&gpiops_inst, MIO_KEY0, 0);
    XGpioPs_SetDirectionPin(&gpiops_inst, MIO_KEY1, 0);
    XGpioPs_SetDirectionPin(&gpiops_inst, EMIO_KEY, 0);

    //读取按键状态，用于控制LED亮灭
    while(1){
        XGpioPs_WritePin(&gpiops_inst, MIO_LED0,
                ~XGpioPs_ReadPin(&gpiops_inst, MIO_KEY0));

        XGpioPs_WritePin(&gpiops_inst, MIO_LED1,
                ~XGpioPs_ReadPin(&gpiops_inst, MIO_KEY1));

        XGpioPs_WritePin(&gpiops_inst, MIO_LED2,
                ~XGpioPs_ReadPin(&gpiops_inst, EMIO_KEY));
    }

    return 0;
}

```



## 4.GPIO之MIO按键中断实验

本章的实验任务是使用底板上的 PS端的用户 按键 PS_KEY1通过 中断控制 核心板上 LED2的亮灭。

### 中断简介：

Zynq芯片的 PS部分是基于使用双核 Cortex A9处理器和 GIC pl390中断控制器的 ARM架构。中断结构与 CPU紧密链接，并接受来自 I/O外设（ IOP）和可编程逻辑 PL）的中断。中断控制器架构如下图所示：

![image-20231018194344233](https://raw.githubusercontent.com/Noregret327/picture/master/202310181943274.png)



系统中断环境：

![image-20231018194401325](https://raw.githubusercontent.com/Noregret327/picture/master/202310182003660.png)

GPIO通道：

![image-20231018194503702](C:\Users\14224\AppData\Roaming\Typora\typora-user-images\image-20231018194503702.png)

### 系统框图：

![image-20231018194624804](https://raw.githubusercontent.com/Noregret327/picture/master/202310182003732.png)

### PS端修改：

```c
/***************************** Include Files *********************************/

#include "xparameters.h"
#include "xgpiops.h"
#include "xscugic.h"
#include "xil_exception.h"
#include "xplatform_info.h"
#include <xil_printf.h>
#include "sleep.h"

/************************** Constant Definitions *****************************/

//以下常量映射到xparameters.h文件
#define GPIO_DEVICE_ID      XPAR_XGPIOPS_0_DEVICE_ID      //PS端GPIO器件ID
#define INTC_DEVICE_ID      XPAR_SCUGIC_SINGLE_DEVICE_ID  //通用中断控制器ID
#define GPIO_INTERRUPT_ID   XPAR_XGPIOPS_0_INTR           //PS端GPIO中断ID

//定义使用到的MIO引脚号
#define KEY  11         //KEY 连接到 MIO11
#define LED  0          //LED 连接到 MIO0

/************************** Function Prototypes ******************************/

static void intr_handler(void *callback_ref);
int setup_interrupt_system(XScuGic *gic_ins_ptr, XGpioPs *gpio, u16 GpioIntrId);

/**************************Global Variable Definitions ***********************/

XGpioPs gpio;   //PS端GPIO驱动实例
XScuGic intc;   //通用中断控制器驱动实例
u32 key_press;  //KEY按键按下的标志
u32 key_val;    //用于控制LED的键值

/************************** Function Definitions *****************************/

int main(void)
{
    int status;
    XGpioPs_Config *ConfigPtr;     //PS 端GPIO配置信息

    xil_printf("Gpio interrupt test \r\n");

    //根据器件ID查找配置信息
    ConfigPtr = XGpioPs_LookupConfig(GPIO_DEVICE_ID);
    if (ConfigPtr == NULL) {
        return XST_FAILURE;
    }
    //初始化Gpio driver
    XGpioPs_CfgInitialize(&gpio, ConfigPtr, ConfigPtr->BaseAddr);

    //设置KEY所连接的MIO引脚的方向为输入
    XGpioPs_SetDirectionPin(&gpio, KEY, 0);

    //设置LED所连接的MIO引脚的方向为输出并使能输出
    XGpioPs_SetDirectionPin(&gpio, LED, 1);
    XGpioPs_SetOutputEnablePin(&gpio, LED, 1);
    XGpioPs_WritePin(&gpio, LED, 0x0);

    //建立中断,出现错误则打印信息并退出
    status = setup_interrupt_system(&intc, &gpio, GPIO_INTERRUPT_ID);
    if (status != XST_SUCCESS) {
        xil_printf("Setup interrupt system failed\r\n");
        return XST_FAILURE;
    }

    //中断触发时，key_press为TURE，延时一段时间后判断按键是否按下，是则反转LED
    while (1) {
        if (key_press) {
            usleep(20000);
            if (XGpioPs_ReadPin(&gpio, KEY) == 0) {
                key_val = ~key_val;
                XGpioPs_WritePin(&gpio, LED, key_val);
            }
            key_press = FALSE;
            XGpioPs_IntrClearPin(&gpio, KEY);      //清除按键KEY中断
            XGpioPs_IntrEnablePin(&gpio, KEY);     //使能按键KEY中断
        }
    }
    return XST_SUCCESS;
}

//中断处理函数
//  @param   CallBackRef是指向上层回调引用的指针
static void intr_handler(void *callback_ref)
{
    XGpioPs *gpio = (XGpioPs *) callback_ref;

    //读取KEY按键引脚的中断状态，判断是否发生中断
    if (XGpioPs_IntrGetStatusPin(gpio, KEY)){
        key_press = TRUE;
        XGpioPs_IntrDisablePin(gpio, KEY);         //屏蔽按键KEY中断
    }
}

//建立中断系统，使能KEY按键的下降沿中断
//  @param   GicInstancePtr是一个指向XScuGic驱动实例的指针
//  @param   gpio是一个指向连接到中断的GPIO组件实例的指针
//  @param   GpioIntrId是Gpio中断ID
//  @return  如果成功返回XST_SUCCESS, 否则返回XST_FAILURE
int setup_interrupt_system(XScuGic *gic_ins_ptr, XGpioPs *gpio, u16 GpioIntrId)
{
    int status;
    XScuGic_Config *IntcConfig;     //中断控制器配置信息

    //查找中断控制器配置信息并初始化中断控制器驱动
    IntcConfig = XScuGic_LookupConfig(INTC_DEVICE_ID);
    if (NULL == IntcConfig) {
        return XST_FAILURE;
    }

    status = XScuGic_CfgInitialize(gic_ins_ptr, IntcConfig,
            IntcConfig->CpuBaseAddress);
    if (status != XST_SUCCESS) {
        return XST_FAILURE;
    }

    //设置并使能中断异常
    Xil_ExceptionRegisterHandler(XIL_EXCEPTION_ID_INT,
            (Xil_ExceptionHandler) XScuGic_InterruptHandler, gic_ins_ptr);
    Xil_ExceptionEnable();

    //为中断设置中断处理函数
    status = XScuGic_Connect(gic_ins_ptr, GpioIntrId,
            (Xil_ExceptionHandler) intr_handler, (void *) gpio);
    if (status != XST_SUCCESS) {
        return status;
    }

    //使能来自于Gpio器件的中断
    XScuGic_Enable(gic_ins_ptr, GpioIntrId);

    //设置KEY按键的中断类型为下降沿中断
    XGpioPs_SetIntrTypePin(gpio, KEY, XGPIOPS_IRQ_TYPE_EDGE_FALLING);

    //使能按键KEY中断
    XGpioPs_IntrEnablePin(gpio, KEY);

    return XST_SUCCESS;
}

```



## 5.AXI GPIO按键控制LED实验

本章的实验任务是通过调用AXI GPIO IP核，使用中 断机制，实现 领航 者 底 板上 PL端 按键 PL_KEY0控制核心 板上 PS端 LED2亮灭 的功能。

在“EMIO按键控制 LED实验 ”中 ，我们通过 EMIO实现了 PS端与 PL端 的交互 ，而 PS与 PL最 主要的连接方式 则是一 组 AXI接口。 AXI互联 接口 作 为 ZYNQ PS和 PL之间 的桥梁， 能够使两者协同 工 作，进而形成一 个完整的、 高度集成的系统 。本章我们将在PL端调用 AXI GPIO IP核 并通过 AXI4 Lite接口 实现 PS与PL中 AXI GPIO模块的通信 。

### AXI简介：

AXI GPIO可以 配置成单通道或者双通道， 每个 通道的位宽可 以 单独设置 。 另外 通过打开或者关闭三态缓冲器， AXI GPIO的端口还 可以被动态地配置成输入或者输出接口 。其 顶层模块的框图如下所示：

![image-20231018200323339](https://raw.githubusercontent.com/Noregret327/picture/master/202310182003965.png)

### 系统框图：

![image-20231018200520208](https://raw.githubusercontent.com/Noregret327/picture/master/202310182005815.png)

### PL端设计：

勾选时钟：

![image-20231018203045196](https://raw.githubusercontent.com/Noregret327/picture/master/202310182030228.png)

勾选RESET和AXI Interface：

![image-20231018203136694](https://raw.githubusercontent.com/Noregret327/picture/master/202310182031670.png)

中断IRQ：

![image-20231018203223018](https://raw.githubusercontent.com/Noregret327/picture/master/202310182032053.png)

MIO配置：

![image-20231018203243625](https://raw.githubusercontent.com/Noregret327/picture/master/202310182032656.png)

添加“AXI GPIO”的IP核：

![image-20231018203335520](https://raw.githubusercontent.com/Noregret327/picture/master/202310182033551.png)

再自动运行添加：

![image-20231018203352483](https://raw.githubusercontent.com/Noregret327/picture/master/202310182033543.png)

![image-20231018203417691](https://raw.githubusercontent.com/Noregret327/picture/master/202310182034150.png)

修改“AXI GPIO”IP核的配置：

![image-20231018203527810](https://raw.githubusercontent.com/Noregret327/picture/master/202310182035366.png)

连接“ip2intc_irpt”和“IRQ_F2P”：

![image-20231018203605865](https://raw.githubusercontent.com/Noregret327/picture/master/202310182036364.png)

修改“AXI GPIO”IP核的输出引脚：

![image-20231018203723898](https://raw.githubusercontent.com/Noregret327/picture/master/202310182037021.png)

最后设置“I/O”口：

![image-20231018203810536](https://raw.githubusercontent.com/Noregret327/picture/master/202310182038669.png)

### PS端设计：

```c
 #include "xparameters.h"
 #include "xgpiops.h"
 #include "xgpio.h"
 #include "xscugic.h"
 #include "xil_exception.h"
 #include "xplatform_info.h"
 #include <xil_printf.h>
 #include "sleep.h"

 /************************** Constant Definitions *****************************/

 //以下常量映射到xparameters.h文件
 #define GPIOPS_DEVICE_ID    XPAR_XGPIOPS_0_DEVICE_ID     //PS端GPIO器件ID
 #define AXI_GPIO_DEVICE_ID  XPAR_AXI_GPIO_0_DEVICE_ID    //PL端AXI GPIO器件ID
 #define SCUGIC_ID           XPAR_SCUGIC_SINGLE_DEVICE_ID //通用中断控制器ID
 #define AXI_GPIO_INT_ID     XPAR_FABRIC_GPIO_0_VEC_ID    //PL端AXI GPIO中断ID

 #define MIO_LED       0                                  //LED 连接到 MIO0
 #define KEY_CHANNEL1  1                                  //PL 按键使用 AXI GPIO 通道 1
 #define KEY_CH1_MASK  XGPIO_IR_CH1_MASK                  //通道 1的中断位定义

 /************************** Function Prototypes ******************************/
 void instance_init();
 int setup_interrupt_system(XScuGic *gic_inst_ptr, XGpio *axi_gpio_inst_ptr,
     u16 AXI_GpioIntrId);
 static void intr_handler(void *callback_ref);

 /**************************Global Variable Definitions ***********************/
 XGpioPs gpiops_inst;        //PS端GPIO驱动实例
 XGpio   axi_gpio_inst;      //PL端AXI GPIO驱动实例
 XScuGic scugic_inst;        //通用中断控制器驱动实例
 int led_value=1;            //ps端LED0的显示状态

 /************************** Function Definitions *****************************/

 int main(void)
 {
     int status;

     //初始化各器件驱动
     instance_init();

     xil_printf("AXI_Gpio interrupt test \r\n");

     //设置LED所连接的MIO引脚的方向为输出并使能输出
     XGpioPs_SetDirectionPin(&gpiops_inst, MIO_LED, 1);
     XGpioPs_SetOutputEnablePin(&gpiops_inst, MIO_LED, 1);
     XGpioPs_WritePin(&gpiops_inst, MIO_LED, led_value);

     //建立中断,出现错误则打印信息并退出
     status = setup_interrupt_system(&scugic_inst, &axi_gpio_inst, AXI_GPIO_INT_ID);
     if (status != XST_SUCCESS) {
         xil_printf("Setup interrupt system failed\r\n");
         return XST_FAILURE;
     }

     return XST_SUCCESS;
 }

 //初始化各器件驱动
 void instance_init()
 {
     XScuGic_Config *scugic_cfg_ptr;
     XGpioPs_Config *gpiops_cfg_ptr;

     //初始化中断控制器驱动
 	scugic_cfg_ptr = XScuGic_LookupConfig(SCUGIC_ID);
     XScuGic_CfgInitialize(&scugic_inst, scugic_cfg_ptr, scugic_cfg_ptr->CpuBaseAddress);

     //初始化PS端  GPIO驱动
     gpiops_cfg_ptr = XGpioPs_LookupConfig(GPIOPS_DEVICE_ID );
     XGpioPs_CfgInitialize(&gpiops_inst, gpiops_cfg_ptr, gpiops_cfg_ptr->BaseAddr);

     //初始化PL端  AXI GPIO驱动
     XGpio_Initialize(&axi_gpio_inst, AXI_GPIO_DEVICE_ID);
 }

 //建立中断系统，使能KEY按键的下降沿中断
 //  @param   GicInstancePtr是一个指向XScuGic驱动实例的指针
 //  @param   gpio是一个指向连接到中断的GPIO组件实例的指针
 //  @param   GpioIntrId是Gpio中断ID
 //  @return  如果成功返回XST_SUCCESS, 否则返回XST_FAILURE
 int setup_interrupt_system(XScuGic *gic_inst_ptr, XGpio *axi_gpio_inst_ptr, u16 AXI_GpioIntrId)
 {
     //设置并使能中断异常
     Xil_ExceptionInit();
     Xil_ExceptionRegisterHandler(XIL_EXCEPTION_ID_INT,
             (Xil_ExceptionHandler) XScuGic_InterruptHandler, gic_inst_ptr);
     Xil_ExceptionEnable();

     //设置中断源的优先级和触发类型(高电平触发)
     XScuGic_SetPriorityTriggerType(gic_inst_ptr, AXI_GpioIntrId, 0xA0, 0x01);
     //为中断设置中断处理函数
     XScuGic_Connect(gic_inst_ptr, AXI_GpioIntrId,
             (Xil_ExceptionHandler) intr_handler, (void *) axi_gpio_inst_ptr);

     //使能来自于axi_Gpio器件的中断
     XScuGic_Enable(gic_inst_ptr, AXI_GpioIntrId);

     //配置PL端 AXI GPIO
	 //设置 AXI GPIO 通道 1方向为输入
     XGpio_SetDataDirection(axi_gpio_inst_ptr, KEY_CHANNEL1, 1);
     XGpio_InterruptEnable(axi_gpio_inst_ptr, KEY_CH1_MASK);  //使能通道1的中断
     XGpio_InterruptGlobalEnable(axi_gpio_inst_ptr);          //使能axi gpio全局中断

     return XST_SUCCESS;
 }

 //中断处理函数
 //  @param   CallBackRef是指向上层回调引用的指针
 static void intr_handler(void *callback_ref)
 {
     XGpio *axi_gpio_inst_ptr = (XGpio *)callback_ref;
     usleep(20000);                                               //延时20ms，按键消抖
   if (XGpio_DiscreteRead(axi_gpio_inst_ptr, KEY_CHANNEL1) == 0) {//按键有效按下
 	  print("Interrupt Detected!\n");
 	  led_value = ~led_value;
 	  XGpioPs_WritePin(&gpiops_inst, MIO_LED, led_value);	  //改变LED显示状态
 	  XGpio_InterruptDisable(axi_gpio_inst_ptr, KEY_CH1_MASK);//关闭 AXI GPIO中断使能
    }
 	  XGpio_InterruptClear(axi_gpio_inst_ptr, KEY_CH1_MASK);  //清除中断
 	  XGpio_InterruptEnable(axi_gpio_inst_ptr, KEY_CH1_MASK); //使能AXI GPIO中断
 }


```



## 6.自定义IP核-呼吸灯实验

本章的实验任务是通过自定义一个LED IP 核，通过PS 端的程序来控制核心板上PL 端 LED1 灯呈现呼吸灯的效果，并且PS 可以通过AXI 接口来控制呼吸灯的开关和呼吸的频率。

### 系统框图：

本次实验选择常用的方式，即创建一个带有AXI 接口的IP 核，该IP 核通过AXI 协议实现PS 和PL 的数据通信。AXI 协议是一种高性能、高带宽、低延迟的片内总线，关于该协议的详细内容，我们会在后面的例程中向大家做详细的介绍。本次实验的系统框图如图所示：

![image-20231018203958914](https://raw.githubusercontent.com/Noregret327/picture/master/202310182040660.png)

























## ###使用###

### 1.打开“Terminal”

![image-20231017172218149](C:\Users\14224\AppData\Roaming\Typora\typora-user-images\image-20231017172218149.png)

![image-20231017172208483](https://raw.githubusercontent.com/Noregret327/picture/master/202310171722502.png)

![image-20231017205210492](https://raw.githubusercontent.com/Noregret327/picture/master/202310172052538.png)

### 2.修改PS端后

#### 1、生成顶层 HDL

依次执行 Generate Output Products”和 Create HDL Wrapper”。

![image-20231017205431310](https://raw.githubusercontent.com/Noregret327/picture/master/202310172054333.png)

![image-20231017205438954](https://raw.githubusercontent.com/Noregret327/picture/master/202310172054981.png)

#### 2、生成 Bitstream文件并导出到 Vitis

![image-20231017205916895](https://raw.githubusercontent.com/Noregret327/picture/master/202310172059925.png)

![image-20231017205934579](C:\Users\14224\AppData\Roaming\Typora\typora-user-images\image-20231017205934579.png)

![image-20231017205947610](https://raw.githubusercontent.com/Noregret327/picture/master/202310172059633.png)