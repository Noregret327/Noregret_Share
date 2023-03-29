# FPGA——IP核

### 一、什么是IP核

#### 1、IP核是什么

IP（intellectual Property）即知识产权，IP即电路功能模块。

在数字电路中，将常用的且比较复杂的功能模块设计成参数可修改的模块，让其他用户可以直接调用这些模块，这就是IP核。

如：FIFO、RAM、SDRAM...

#### 2、为什么要使用IP核

提高开发效率，减少设计和调试时间，加速开发进程，降低开发成本。

#### 3、IP核的存在形式

分类依据：产品交付方式

* HDL语言形式——软核

  硬件描述语言；可进行参数调整、复用性强；布局、布线灵活；设计周期短、设计投入少；

* 网表形式——固核

  完成了综合的功能块；可预布线特定信号或分配特定的布线资源；

* 版图形式——硬核

  硬核是完成提供设计的最终阶段产品-掩膜（Mask）；缺乏灵活性、可移植性差；更易于实现IP核保护；

#### 4、IP核的缺点

* IP核往往不能跨平台使用
* IP核不透明，看不到内部核心代码
* 定制IP需要额外收费

#### 5、Quartus II软件下IP核的调用

* Mega Wizard插件管理器
* SOPC构造器
* DSP构造器
* Qsys设计系统例化

##### Mega Wizard插件管理器：

1.打开Quartus II，选择‘Tool’，选择‘MegaWizard Plug-In Manager [page 1]’

2.选择“Create a new custom megafunction variation”

3.选“芯片型号”，输出”语言类型“，搜索“IP核”

#### 6、Altera IP核的分类

1.Arithmetic——数学运算类（LPM、ALTMULT、ALTFP）

2.Communications——输出通讯类（CRC、8B10B）

3.DSP——数字运算类（CIC、FIR、NCO、FFT）/视频核图像处理类

4.Gates——数学运算类（LPM）

5.I/O

6.Interfaces

7.JTAG-accessible Extensions

8.Memory Compiler——存储类（FIFO、RAM、ROM...）

9.PLL



### 二、IP核的调用——PLL

#### 1、PLL——IP核简介

PLL (Phase Locked Loop，即锁相环) 是最常用的IP核之一，其性能强大，可以对输入到FPGA的时钟信号进行任意分频、倍频、相位调整、占空比调整，从而输出一个期望时钟。

##### PLL的基本工作原理

![image-20230326144320665](https://raw.githubusercontent.com/Noregret327/picture/master/202303261443714.png)

* FD/PD：鉴频鉴相器：比较ref_clk（参考时钟）和com_clk（比较时钟）的频率和相位的差异。
* LF：环路滤波器：用于滤到高频噪声。还会根据鉴频器的值输出不同电压值。
* VCO：压控振荡器：通过前面的环路滤波器的电压值控制输出频率，输入电压值越高输出频率越高。
* 工作原理：根据参考时钟与比较时钟经鉴频鉴相器对比输出到环路滤波器，环路滤波器会输出对应的电压值到压控振荡器后输出pll_out，然后反馈到鉴频鉴相器反复调节到与参考时钟相同的频率。

##### PLL的倍频

![image-20230326144213206](https://raw.githubusercontent.com/Noregret327/picture/master/202303261442238.png)

* PLL的倍频原理：它比前面的基本原理多个分频器，当输入参考频率为50MHz时，假如分频器为二分频，输入到分频器的频率则要为100MHz输出的频率才为50MHz，所以pll_out的输出频率为100MHz。

##### PLL的分频

![image-20230326144807872](https://raw.githubusercontent.com/Noregret327/picture/master/202303261448907.png)

* PLL的分频原理：它比前面的基本原理多个分频器，和前面的倍频一样，但分频器的位置不一样，当输入参考频率为50MHz时，假如分频器为五分频，输出到鉴频鉴相器的频率为10MHz，所以pll_out的输出频率为10MHz。

##### 分频与倍频——除法与乘法

![image-20230326154518003](https://raw.githubusercontent.com/Noregret327/picture/master/202303261545043.png)

* 根据需要的输出，与参考频率一起求解最大公约数，用公约数做对应参考频率的除法因子和比较频率的乘法因子的值，从而设置输出频率。

#### 2、PLL——IP核配置

1.打开Quartus II的Mega Wizard插件管理器

2.设置“芯片”、“语言”、“输出文件位置”——\ip_core\pll_ip\pll_ip，最后选择ip核

3.IP核等级越高速度越慢，等级越低速度越快，对于赛灵思的相反，等级越高速度越快。

4.PLL的模式：

* 正常模式：PLL反馈路径源是全局或局域时钟网络，可以最小化时钟类型和指定 PLL寄存器的时钟延时，还可以指定补偿的 PLL输出。
* 源同步模式：数据和时钟信号同时到达输入引脚。从引脚到 I/0 输入存器之间的时钟延时与从引脚到 I/0 输入寄存器之间的数据延时匹配，信号可以确保在任何输入/输出使能(IOE)寄存器的时钟和数据端口有相同的相位关系。——**数据与时钟的相位关系不变**
* 零延时缓存模式：PLL 反馈路径限制在专用 PLL 外部输出引脚上。片外驱动的 PLL反馈路径与时钟输入是相位对齐的，且时钟输入和外部时钟输出之间的延时最小。——**PLL内部时钟有相位偏移，外部时钟无相位偏移**
* 无补偿模式：PLL反馈路径限制在 PLL环中，没有时钟网络或其他外部源。处于无补偿模式的 PLL 没有时钟网络补偿，但生成的**时钟抖动是最小化**的。——**PLL内部以及外部时钟都有相位偏移**
* 外部反馈模式：PLL补偿到 PLL的反馈输入，这样可以最小化输入时钟引脚和反馈时钟引脚之间的延时。

5.异步复位（先取消勾选）

6.输出时钟配置

* 频率设置——除法因子和乘法因子的设置/直接设置频率
* 相位偏移设置（phase shift）——角度的设置
* 占空比设置（duty cyde）——一般50%
* 设置：C0——100MHz、50%；C1——25MHz、50%；C2——50MHz、90°、50%；C3——50MHz、20%



#### 3、PLL——IP核调用

##### IP核的调用

```verilog
module  pll
(
    input   wire    sys_clk     ,
    output  wire    clk_mul_2   ,
    output  wire    clk_div     ,
    output  wire    clk_pha_90  ,
    output  wire    clk_duc_20  ,
    output  wire    locked
);

pll_ip	pll_ip_inst 
(
	.inclk0 ( sys_clk   ),
	.c0     ( clk_mul_2 ),
	.c1     ( clk_div   ),
	.c2     ( clk_pha_90),
	.c3     ( clk_duc_20),
	.locked ( locked    )
);

endmodule
```

##### IP核的修改

* 通过添加文件旁的“IP Components”，双击对应的IP核进行修改
* 通过创建IP核的方式，打开“Tool”，选择“MegaWizard Plug-In Manager”，再点击edit IP核

##### IP核的添加

* 通过添加文件的方式，添加对应IP核文件夹所在的”.qip“文件夹

##### IP核的复制

* 先复制ip_core整个文件夹到prj目录下，再通过IP核的添加方式添加IP核
* 通过“MegaWizard Plug-In Manager”，再点击copy IP核，选择IP核（.v文件），新建文件夹“ip_core/pll_ip”，设置输出路径

#### 4、PLL——IP核仿真



### 三、IP核的调用——ROM

#### 1、ROM——IP核简介

ROM是只读存储器 (Read-Only Memory) 的简称是一种只能读出事先所存数据的固态半导体存储器。其特性是一旦储存资料就无法再将之改变或删除，且资料不会因为电源关闭而消失。

ROM生成的文件是“.hex”、“.mif”

* 分类：
* 单端口

![image-20230326204332228](https://raw.githubusercontent.com/Noregret327/picture/master/202303262043260.png)

* 双端口

![image-20230326204355008](https://raw.githubusercontent.com/Noregret327/picture/master/202303262043041.png)



#### 2、ROM——IP核配置

1.新建选择“Memory File”中的“hex”文件类型或者“mif”文件类型

2.设置“Number of words”——数据大小和“Word size”——位宽

3.设置rom数据

4.保存数据（保存在“ip_core”下的"rom_8_256"，名字为“rom_8_256”）——8表示位宽，256表示数据大小



* **Cyclone IV——支持M9K（9Kbits）**



* 单端口配置

1.打开Quartus II的Mega Wizard插件管理器

2.设置“芯片”、“语言”、“输出文件位置”——\ip_core\rom_8x256\rom_8x256，最后选择单端口ROM的ip核

3.设置输出的位宽、深度、存储类型、最大深度、时钟模式选择（单时钟——地址写入和读取都使用同一个时钟、双时钟——地址的写入使用一个时钟读取使用另外一个时钟）

4.设置reg/clk_en/adress——（读写操作都是使用上升沿）——（选择寄存器输出会延时一个时钟周期，不选择寄存器输出会延时两个时钟周期）

5.ROM的IP核初始化——添加“.mif”文件

6.设置EDA

7.设置Summary——选择“inst”文件，取消“bb”文件

* 双端口设置

1.打开Quartus II的Mega Wizard插件管理器

2.设置“芯片”、“语言”、“输出文件位置”——\ip_core\rom_8x256_double\rom_8x256_double，最后选择双端口ROM的ip核

3.选择ROM大小方式——“words”或“Bits”

4.设置不同输出的位宽、存储器的类型选择、时钟选择

5.设置reg/clk_en/adress——（读写操作都是使用上升沿）——（选择寄存器输出会延时一个时钟周期，不选择寄存器输出会延时两个时钟周期）

6.ROM的IP核初始化——添加“.mif”文件

7.设置EDA

8.设置Summary——选择“inst”文件，取消“bb”文件

#### 3、ROM——IP核调用

* ds：表示串行数据
* oe：表示使能信号
* shcp：移位寄存器时钟
* stcp：存储寄存器时钟

```verilog
module rom
(
    input   wire            sys_clk     ,
    input   wire            sys_rst_n   ,
    input   wire            key1        ,
    input   wire            key2        ,
    
    output  wire            ds          ,
    output  wire            oe          ,
    output  wire            shcp        ,
    output  wire            stcp
);

wire                key1_flag   ;
wire                key2_flag   ;
wire    [7:0]       addr        ;
wire    [7:0]       data        ;

key_filter
#(
    .CNT_MAX (20'd999_999)
)
key_filter_inst1
(
    .sys_clk     (sys_clk),
    .sys_rst_n   (sys_rst_n),
    .key_in      (key1),

    .key_flag    (key1_flag)
);

key_filter
#(
    .CNT_MAX (20'd999_999)
)
key_filter_inst2
(
    .sys_clk     (sys_clk),
    .sys_rst_n   (sys_rst_n),
    .key_in      (key2),

    .key_flag    (key2_flag)
);

rom_ctrl
#(
    .CNT_MAX (24'd9_999_999)
)
rom_ctrl_inst
(
    .sys_clk     (sys_clk),
    .sys_rst_n   (sys_rst_n),
    .key1        (key1_flag),
    .key2        (key2_flag),

    .addr        (addr)
);

rom_8x256	rom_8x256_inst 
(
	.address    (addr   ),
	.clock      (sys_clk),
	.q          (data   )
);

seg_595_dynamic seg_595_dynamic_inst
(
    .sys_clk     (sys_clk),     //系统时钟，频率50MHz
    .sys_rst_n   (sys_rst_n),   //复位信号，低有效
    .data        ({12'b0,data}),//数码管要显示的值
    .point       (6'b000_000),  //小数点显示,高电平有效
    .seg_en      (1'b1),        //数码管使能信号，高电平有效
    .sign        (1'b0),        //符号位，高电平显示负号
        
    .stcp        (stcp),        //数据存储器时钟
    .shcp        (shcp),        //移位寄存器时钟
    .ds          (ds  ),        //串行数据输入
    .oe          (oe  )         //使能信号

);

endmodule
```

#### 4、ROM——IP核仿真



### 四、IP核的调用——RAM

#### 1、RAM——IP核简介

RAM是随机存取存储器(Random Access Memory)的简称，是一个易失性存储器；其工作时可以随时对任何一个指定的地址写入或读出数据。这是ROM所并不具备的功能。

* Altera分类：
* 单端口RAM

![image-20230327105424780](https://raw.githubusercontent.com/Noregret327/picture/master/202303271054816.png)

* 双端口RAM——又分为：简单双端口RAM和真正双端口RAM

![image-20230327111527231](https://raw.githubusercontent.com/Noregret327/picture/master/202303271115255.png)

![image-20230327111538442](https://raw.githubusercontent.com/Noregret327/picture/master/202303271115463.png)

#### 2、RAM——IP核配置

* 单端口选择

通过IP核管理器选择单端口RAM

* 选择位宽、存储类型、最大深度、时钟选择
* Regs/Clken/Byte Enable/Adrs——选择异步清零和读使能信号
* Read During Write Option和Mem Init——默认就行
* EDA——默认就行
* Summary——只选择inst



* 双端口选择_简单型

通过IP核管理器选择双端口RAM

* 选择双端口为简单还是真正类型、选择数据存储格式——简单类型
* Widths/Blk Type——256x8，其他默认
* Clks/Rd, Byte En——选择读写使能
* Regs/Clkens/Adrs——默认值
* Output1——默认值
* Mem Init——默认值
* EDA——默认就行
* Summary——只选择inst



* 双端口选择_真正型

通过IP核管理器选择双端口RAM

* 选择双端口为简单还是真正类型、选择数据存储格式——真正类型
* Widths/Blk Type——256x8，其他默认
* Clks/Rd, Byte En——选择读写使能
* Regs/Clkens/Adrs——默认值
* Output1——默认值
* Mem Init——默认值
* EDA——默认就行
* Summary——只选择inst

#### 3、RAM——IP核调用

* 

#### 4、RAM——IP核仿真

* 仿真报错： Module 'ram_8x256_one' is not defined.

解决方法1：

* 打开之前添加仿真的文件
* 把“ram_8x256_one”这个文件也添加上去



### 五、IP核的调用——FIFO

#### 1、FIFO——IP核简介

FIFO ( First In First Out，即**先入先出**)，是一种数据缓冲器，用来实现数据先入先出的读写方式。

FIFO 存储器主要是作为**缓存**，应用在同步时钟系统和异步时钟系统中，在很多的设计中都会使用;如：多比特数据做跨时钟域处理、前后带宽不同步等都用到了FIFO。

* 写操作

![image-20230328145551766](https://raw.githubusercontent.com/Noregret327/picture/master/202303281455783.png)

* 读操作

![image-20230328145602129](https://raw.githubusercontent.com/Noregret327/picture/master/202303281456149.png)

* 跨时域处理

![image-20230328145615185](https://raw.githubusercontent.com/Noregret327/picture/master/202303281456228.png)

* 带宽不同步

![image-20230328145622995](https://raw.githubusercontent.com/Noregret327/picture/master/202303281456037.png)

##### FIFO根据读写时钟不同分类

* 同步FIFO——SCFIFO
* 异步FIFO——DCFIFO

#### 2、SCFIFO——IP核的配置、调用与仿真

##### 创建SCFIFO

* 选择位宽、数据大小、选择同步FIFO
* SCFIFO Options——选择FIFO的Full和Empty的设置
* Rdreq Option, Blk Type——FIFO读取模式（正常）、数据类型（Auto）
* Optimization, Circuitry Protection——优化选择（最大速度/最小面积✔）
* EDA——默认就行
* Summary——只选择inst

```verilog
module  fifo
(
    input   wire            sys_clk     ,
    input   wire    [7:0]   pi_data     ,
    input   wire            rd_req      ,
    input   wire            wr_req      ,
    
    output  wire            empty       ,
    output  wire            full        ,
    output  wire    [7:0]   po_data     ,
    output  wire    [7:0]   usedw       
    
);

scfifo_8x256	scfifo_8x256_inst 
(
	.clock  (sys_clk),
	.data   (pi_data),
	.rdreq  (rd_req ),
	.wrreq  (wr_req ),
	.empty  (empty  ),
	.full   (full   ),
	.q      (po_data),
	.usedw  (usedw  )
);

endmodule


//仿真
`timescale  1ns/1ns
module  tb_fifo();

reg             sys_clk     ;
reg             sys_rst_n   ;
reg     [7:0]   pi_data     ;
reg             rd_req      ;
reg             wr_req      ;
reg     [1:0]   cnt         ;

wire            empty       ;
wire            full        ;
wire    [7:0]   po_data     ;
wire            usedw       ;

initial
    begin
        sys_clk = 1'b1;
        sys_rst_n   <=    1'b0;
        #20
        sys_rst_n   <=  1'b1;
    end

always #10 sys_clk = ~sys_clk;

always@(posedge sys_clk or negedge sys_rst_n)
    if(sys_rst_n    ==  1'b0)
        cnt <=  2'd0;
    else    if(cnt  ==  2'd3)
        cnt <=  2'd0;
    else
        cnt <=  cnt +   1'b1;

always@(posedge sys_clk or negedge sys_rst_n)
    if(sys_rst_n    ==  1'b0)
        wr_req  <=  1'b0;
    else    if(cnt  ==  2'd0 && rd_req  ==  1'b0)
        wr_req  <=  1'b1;
    else
        wr_req  <=  1'b0;

always@(posedge sys_clk or negedge sys_rst_n)
    if(sys_rst_n    ==  1'b0)    
        pi_data <=  8'd0;
    else    if(pi_data  ==  8'd255 && wr_req == 1'b1)
        pi_data <=  8'd0;
    else    if(wr_req == 1'b1)
        pi_data <=  pi_data + 1'b1;
    else
        pi_data <=  pi_data;
        
always@(posedge sys_clk or negedge sys_rst_n)
    if(sys_rst_n    ==  1'b0) 
        rd_req  <=  1'b0;
    else    if(full ==  1'b1)
        rd_req  <=  1'b1;
    else    if(empty    ==  1'b1)
        rd_req  <=  1'b0;

fifo    fifo_inst
(
    .sys_clk     (sys_clk),
    .pi_data     (pi_data),
    .rd_req      (rd_req ),
    .wr_req      (wr_req ),

    .empty       (empty  ),
    .full        (full   ),
    .po_data     (po_data),
    .usedw       (usedw  )
    
);

endmodule

```



#### 3、DCFIFO——IP核的配置、调用与仿真

##### 创建DCFIFO

* 选择位宽、数据大小、选择异步FIFO——输入与输出位宽（8bit和16bit）、异步FIFO
* DCFIFO 1和DCFIFO 2——设置optimization（资源速度-中）、输出管脚选择（全选-full、empty、usedw、extra MSB）
* Rdreq Option, Blk Type——FIFO读取模式（正常）、数据类型（Auto）
* Optimization, Circuitry Protection——优化选择（最大速度/最小面积✔）
* EDA——默认就行
* Summary——只选择inst

```Verilog
module  fifo_dcfifo
(
    input   wire                wr_clk      ,
    input   wire                wr_req      ,
    input   wire    [7:0]       wr_data     ,
    input   wire                rd_clk      ,
    input   wire                rd_req      ,
    
    output  wire    [15:0]      rd_data     ,
    output  wire                wr_empty    ,
    output  wire                wr_full     ,
    output  wire    [8:0]       wr_usedw    ,
    output  wire                rd_empty    ,
    output  wire                rd_full     ,
    output  wire    [7:0]       rd_usedw
);

dcfifo_8x256_to_16x128	dcfifo_8x256_to_16x128_inst
(
	.data       (wr_data    ),
	.rdclk      (rd_clk     ),
	.rdreq      (rd_req     ),
	.wrclk      (wr_clk     ),
	.wrreq      (wr_req     ),
    
	.q          (rd_data    ),
	.rdempty    (rd_empty   ),
	.rdfull     (rd_full    ),
	.rdusedw    (rd_usedw   ),
	.wrempty    (wr_empty   ),
	.wrfull     (wr_full    ),
	.wrusedw    (wr_usedw   )
);

endmodule


//仿真
`timescale 1ns/1ns
module  tb_fifo_dcfifo();

reg                 wr_clk      ;
reg                 wr_req      ;
reg     [7:0]       wr_data     ;
reg                 rd_clk      ;
reg                 rd_req      ;
reg                 sys_rst_n   ;
reg     [1:0]       cnt         ;
reg                 wr_full_reg0;
reg                 wr_full_reg1;    

wire    [15:0]      rd_data     ;
wire                wr_empty    ;
wire                wr_full     ;
wire    [8:0]       wr_usedw    ;
wire                rd_empty    ;
wire                rd_full     ;
wire    [7:0]       rd_usedw    ;

initial
    begin
        wr_clk  =   1'b1;
        rd_clk  =   1'b1;
        sys_rst_n   <=  1'b0;
        #20
        sys_rst_n   <=  1'b1;
    end

always #10 wr_clk   =   ~wr_clk;    //周期20ns——50MHz

always #20 rd_clk   =   ~rd_clk;    //周期40ns——25MHz

always@(posedge wr_clk or negedge sys_rst_n)
    if(sys_rst_n == 1'b0)
        cnt  <=  2'd0;
    else    if(cnt == 2'd3)
        cnt  <=  2'd0;
    else
        cnt  <=  cnt    +   1'b1;

always@(posedge wr_clk or negedge sys_rst_n)
    if(sys_rst_n == 1'b0)
        wr_req  <=  1'b0;
    else    if((cnt == 2'd0) && (rd_req == 1'b0))
        wr_req  <=  1'b1;
    else
        wr_req  <=  1'b0;

always@(posedge wr_clk or negedge sys_rst_n)
    if(sys_rst_n == 1'b0)
        wr_data <=  8'd0;
    else    if((wr_data == 8'd255) && (wr_req == 1'b1))
        wr_data <=  8'd0;
    else    if(wr_req == 1'b1)
        wr_data <=  wr_data + 1'b1;
    else
        wr_data <=  wr_data;

always@(posedge rd_clk or negedge sys_rst_n)
    if(sys_rst_n == 1'b0)
        begin
            wr_full_reg0    <=  1'b0;
            wr_full_reg1    <=  1'b0;
        end
    else
        begin
            wr_full_reg0    <=  wr_full;
            wr_full_reg1    <=  wr_full_reg0;
        end

always@(posedge rd_clk or negedge sys_rst_n)
    if(sys_rst_n == 1'b0)
        rd_req  <=  1'b0;
    else    if(wr_full_reg1 == 1'b1)
        rd_req  <=  1'b1;
    else    if(rd_empty == 1'b1)
        rd_req  <=  1'b0;

fifo_dcfifo fifo_dcfifo_inst
(
    .wr_clk      (wr_clk    ),
    .wr_req      (wr_req    ),
    .wr_data     (wr_data   ),
    .rd_clk      (rd_clk    ),
    .rd_req      (rd_req    ),

    .rd_data     (rd_data   ),
    .wr_empty    (wr_empty  ),
    .wr_full     (wr_full   ),
    .wr_usedw    (wr_usedw  ),
    .rd_empty    (rd_empty  ),
    .rd_full     (rd_full   ),
    .rd_usedw    (rd_usedw  )
);

endmodule
```



### !!!仿真问题

#### 1、出现蓝色线

* Modelsim仿真出现蓝色或者红色的线错误原因：数据的宽度定义不不对，有时候负数时符号位溢出了。
* 可能IP核设置不对——关于出现问题的端口的位宽设置。

