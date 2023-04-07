# FPGA—Verilog

## 学习过程

1.点灯——led

2.多路选择器——mux2_1

3.译码器——decoder

4.半加器——half_adder

5.全加器——full_adder

6.触发器——flip_flop（锁存器——Latch）

7.阻塞与非阻塞——blocking_and_no_blocking

8.计数器——counter

9.分频器（偶分频）——divider_six

10.分频器（奇分频）——divider_five

11.按键消抖——key_filter

12.触摸按键——touch_ctrl_led

13.流水灯——water_led

14.呼吸灯——breath_led

15.状态机——simple_fsm





```
1s
f = 50MHz = 5*10^4KHz = 5*10^7Hz
t = 1/f = (1/5*10^7)s = 2*10^(-8)s = 20ns
M = 1s/20ns = 5*10^7
计数从0开始，所以计数为：M - 1
```



## ！！！模板！！！

#### 2选1选择器

```verilog
//程序
module  mux2_1
(
    input   wire    [0:0]   in_1,
    input   wire            in_2,
    input   wire            sel,
    
    output  reg             out			//always语句中要用reg型
);

always@(*)
    if(sel == 1'b1)
        out = in_1;
    else
        out = in_2;

endmodule
```



```verilog
//testbatch
`timescale 1ns/1ns

module  tb_mux2_1();
//定义变量
reg     in_1;
reg     in_2;
reg     sel ;

wire    out ;
//设置初始值
initial
    begin
        in_1    <=  1'b0;
        in_2    <=  1'b0;
        sel     <=  1'b0; 
    end
//模拟时钟
always #10 in_1 <=  {$random} % 2;      //对随机数进行求余数——余数结果只有0/1
always #10 in_2 <=  {$random} % 2;      //对随机数进行求余数——余数结果只有0/1
always #10 sel  <=  {$random} % 2;      //对随机数进行求余数——余数结果只有0/1

//打印信息
initial
    begin
        $timeformat(-9,0,"ns",6);       //-9:表示小数点几位；0：表示位数；6：表示打印最小数字字符是6
        $monitor("@time %t:in_1=%b in_2=%b sel=%b out=%b",$time,in_1,in_2,sel,out);	//监视器
    end
//实例化
mux2_1  mux2_1_inst
(
    .in_1(in_1),
    .in_2(in_2),
    .sel (sel ),

    .out (out )							//括号里面的是自己定义的输入输出，.后面的是芯片网络
);
endmodule
```



### 一、符号&操作

#### 信号定义

寄存器型变量/输入定义为：reg型

线网型变量/输出定义为：wire型

#### 参数

```verilog
parameter CNT_MAX = 100;		//可以在模块实例化中进行使用和修改
localparam CNT_MAX = 100;		//只能在模块内部中使用，不能进行实例化
```



##### 1、时钟信号

时钟信号属重复序列。

1、always语句

```verilog
reg clk;
initial
    clk = 0;
always
    #10 clk = ~clk;
```

2、forever语句

```verilog
reg clk
initial
    begin
        clk = 0;
        forever
            #10 clk = ~clk;
    end
```

##### 2、复位信号

复位信号属确定值序列。

1、直接给定延时单位

```verilog
// 阻塞型赋值
initial
    begin
        rst = 0;
        #100 ret = 1;
        #80 rst = 0;
        #30 rst = 1;
    end
```

或

```Verilog
// 阻塞型赋值
initial
    begin
        rst = 0;
        rst = #100 1;		//等同于 #100 rst = 1;
        rst = #80 0;
        rst = #30 1;
    end
```

或

```Verilog
//非阻塞型
initial
    begin
        rst <= 0;
        rst <= #100 1;
        rst <= #180 0;
        rst <= #210 1;
    end
```

2、任务调用方法

```Verilog
//任务定义
task sys_time;
    input [10:0]rst_time;	//调用task时，将参数100赋值给rst_time
    begin
        rst = 0;
        #rst_time rst=1;
    end
    
//任务调用
initial
    begin 
        sys_time(100);		//100个时间单位延时之后复位
    end
```

#### 常用系统函数

##### 1)$time

作用：返回所在模块的仿真时间，可以查看信号的出现时间，用来把握信号的时序。

```Verilog
$display("the time is %t", $time);		//显示当时的时间
```

##### 2)$display

作用：将需要显示的内容在命令栏显示。

```verilog
$display("the signal is %d", ad);		//将ad信号以十进制的方式显示出来
```

##### 3)$monitor

作用：监视变量的变化，一旦变量变化，则将变量显示出来。

```Verilog
$monitor("at time is %t and the signal is %b\n", $time, signal);
```

##### 3)$stop或$finsh

作用：暂停或结束仿真。

#### 赋值

阻塞式赋值：=				一般搭配assign使用，立即执行赋值

非阻塞式赋值：<=			在整条语句/块语句结束之后才执行赋值

#### 操作

1、创建Verilog HDL File——主意用类似C语言来编写电路

```Verilog
module DFF1(CLK,D,Q);
	output Q;
	input CLK,D;
	reg Q;
	always @(posedge CLK)
		Q <= D;
endmodule
```

2、创建Block Diagram/Schematic File——主要用与或非门等电路元件画电路

![image-20230318142407938](https://raw.githubusercontent.com/Noregret327/picture/master/img202303181424029.png)

3、Create Symbol Files for Current File——主要是把电路封装起来便于下次调用

![image-20230318142420693](https://raw.githubusercontent.com/Noregret327/picture/master/img202303181424761.png)





### ！！！例子！！！

```verilog
//DUT
module nand2_1(a,b,y);
    input a,b;
    output y;
    reg y;
    always @(a,b)
        begin
            case({a,b})
                2'b00 : y = 1;
                2'b01 : y = 1;
                2'b10 : y = 1;
                2'b11 : y = 0;
                default : y = 'bx;
            endcase
        end
endmodule

//Testbench
`timescale 1ns/100ps						//时间尺度
module nand2_1_tb;							//测试模块名称
    reg a,b;
    wire y;									
    initial
        begin 
            a = 0; b = 0;					//赋初值
            #100 a = 0; b = 1;
            #100 a = 1; b = 0;
            #100 a = 1; b = 1;
            #100 a = 0; b = 0;
            #200 $stop;						//停止仿真
        end
    nand2_1 mynand2(.a(a),.b(b),.y(y));		//例化
endmodule
```



### 二、4选1多路选择器

#### 1、case语句表达式

```verilog
module MUX41a(a,b,c,d,s1,s0,y);
	input a,b,c,d;
	input s1,s0;
	output y;
	reg y;
	always @(a or b or c or d or s1 or s0)
	begin : MUX41
		case({s1,s0})
			2'b00 : y<=a;
			2'b01 : y<=b;
			2'b10 : y<=c;
			2'b11 : y<=d;
			default : y<=a;
		endcase
	end
endmodule
```

#### 2、assign语句表达式

```verilog
module MUX41a(a,b,c,d,s1,s0,y);
	input a,b,c,d,s1,s0;
	output y;
	wire [1:0] SEL;									//定义2元素位矢量SEL为网线型变量wire
	wire AT,BT,CT,DT;									//定义中间变量，以作连线或信号节点
	assign SEL={s1,s0};								//对s1，s0进行并位操作，即SEL[]]=s1;SEL[0]=s0
	assign AT={SEL==2'D0};
	assign BT={SEL==2'D1};
	assign CT={SEL==2'D2};
	assign DT={SEL==2'D3};
	assign y=(a&AT)|(b&BT)|(c&CT)|(d&DT);		//4个逻辑信号相或
endmodule
```

#### 3、条件赋值语句表达式

```verilog
module MUX41a(a,b,c,d,S1,S0,Y);
	input A,B,C,D,S1,S0;
	output Y;
	wire AT = S0 ? D : C;
	wire BT = S0 ? B : A;
	wire Y = (S1 ? AT : BT);
endmodule
```

#### 4、条件语句表述方式

```verilog
module MUX41a (A,B,C,D,S1,S0,Y);
	input A,B,C,D,S1,S0;
	output Y;
	reg [1:0] SEL;
	reg Y;
	always @(A,B,C,D,SEL)
	begin
		SEL = {S1,S0};
		if (SEL==0) Y = A;
		else if (SEL==1) Y = B;
		else if (SEL==2) Y = C;
		else Y = D;
	end
endmodule
```



### 三、D触发器

#### 原理

保持型D触发器

边沿触发型D触发器

只有在时钟在上升沿时候才随着D的改变而改变。

#### 1、程序——上升沿触发

```verilog
module DFF1(CLK,D,Q);
	output Q;
	input CLK,D;
	reg Q;
	always @(posedge CLK)
		Q <= D;
endmodule
```

posedge：上升沿触发语句——positive edge

negedge：下降沿触发语句——negative edge

#### 2、程序——UDP表述

```verilog
primitive EDGE_UDP(Q,D,CLK,RST);
	input D,CLK,RST;
	output Q;
	reg Q;
	table // D  CLK   : Q : Q+
			 0 (01) 0 : ? : 0;
			 1 (01) 0 : ? : 1;
			 ? (1?) 0 : ? : -;
			 ? (?0) 0 : ? : -;
			 1   0  1 : ? : 0;
        	 1   1  1 : ? : 0;
        	 0   0  1 : ? : 0;
        	 0   1  1 : ? : 0;
    endtable
endprimitive

module DFF_UDP(Q,D,CLK,RST);
    input D,CLK,RST;
    output Q;
    EDGE_UDP U1(Q,D,CLK,RST);
endmodule
```

#### 3、程序——异步复位和时钟使能

```verilog
module DFF2(CLK, D,Q,RST,EN);
    output Q;
    input CLK,D,RST,EN;
    reg Q;
    always @(posedge CLK or negedge RST)
        begin
            if (!RST) Q <= 0;
            else if (EN) Q <= D;
        end
endmodule
```

#### 4、程序——同步复位

```verilog
module DFF1(CLK,D,Q,RST);
    output Q;
c
    always @(RST)
        if (RST==1) Q1 = 0;
    else Q1 = D;
    always @(posedge CLK)
        Q <= Q1;
endmodule

```

```verilog
module DFF2(input CLK, input D, input RST, output reg Q,);
    always @(posedge CLK)
        Q <= RST ? 1'b0 : D;
endmodule
        
```

```verilog
module DFF3(CLK,D,Q,RST);
    output Q;
    input CLK,D,RST;
    reg Q;
    always @(posedge CLK)
        if (RST==1) Q = 0;
    else if (RST==0) Q = D;
endmodule

```

#### 5、程序——基本锁存器（保持状态）

```verilog
module LATCH1(CLK,D,Q);
    output Q;
    input CLK,D;
    reg Q;
    always @(D or CLK)
        if (CLK) Q <= D;
endmodule
```

```verilog
//含清0控制
module LATCH2(CLK,D,Q,RST);
    output Q;
    input CLK,D,RST;
    assign Q = (!RST) ? 0 : (CLK ? D : Q);
endmodule
```

```verilog
//含清0控制
module LATCH3(CLK,D,Q,RST);
    output Q;
    input CLK,D,RST;
    reg Q;
    always @(D or CLK or RST)
        if (!RST) Q <= 0;
    else if (CLK) Q <= D;
endmodule
```

#### 6、程序——异步时序电路

```verilog
module AMOD(D,A,CLK,Q);
    output Q;
    input A,D,CLK;
    reg Q,Q1;
    always @(posedge CLK) Q1 = ~ (A|Q);
    always @(posedge Q1) Q = D;
endmodule
```



### 四、半加器电路

#### 1、原理

半加器是由一个异或门和一个与门组成，半加器共有两个输入端和两个输出端，详细电路结构如下图所示：

![image-20230224200433692](https://raw.githubusercontent.com/Noregret327/picture/master/img202303181421975.png)

![image-20230318142350366](https://raw.githubusercontent.com/Noregret327/picture/master/img202303181423469.png)

#### 2、程序

```verilog
module h_adder(A,B,SO,CO);
    input A,B;
    output SO,CO;
    assign SO = A ^ B;
    assign CO = A & B;
endmodule
```

#### 3、半加器的UDP结构建模描述方式

1.用户自定义原语——User-Defined Primitive

2.库元件及其调用

```Verilog
//UDP——用户自定义原语
primitive XOR2(DOUT,X1,X2);
    input X1,X2;	output DOUT;
    table // X1 X2 : DOUT
        	  0  0 :   0;
        	  0  1 :   1;
        	  1  0 :   1;
              1  1 :   0;
    endtable
endprimitive

//库元件及其调用
module H_ADDER(A,B,SO,CO);
    input A,B;	output SO,CO;
    XOR2 U1(SO,A,B);	//调用元件XOR2——使用例化语句的位置关联法
    and U2(CO,A,B);		//调用元件and
endmodule
```

```verilog
//UDP——用户自定义原语
primitive MUX41_UDP(Y,D3,D2,D1,D0,S1,S0);
    input D3,D2,D1,D0,S1,S0;	output Y;
    table //D3 D2 D1 D0 S1 S0 : Y
        	?  ?  ?  1  0  0  : 1;
        	?  ?  ?  0  0  0  : 0;
        	?  ?  1  0  0  1  : 1;
        	?  ?  0  0  0  1  : 0;
        	?  1  ?  0  1  0  : 1;
        	?  0  ?  0  1  0  : 0;
        	1  ?  ?  0  1  1  : 1;
        	0  ?  ?  0  1  1  : 0;
    endtable
endprimitive

//库元件及其调用
module MUX41UPD(D,S,DOUT);
    input [3:0] D;
    itput [1:0] S;
    output DOUT;
    MUX41_UDP (DOUT,D[3],D[2],D[1],D[0],S[1],S[0]);   //调用MUX41_UDP——使用例化语句的位置关联法 	
endmodule
```



### 五、 全加器设计及例化语句应用

#### 1、原理

全加器是由两个半加器和一个或门组成，全加器共有三个输入端和两个输出端，详细电路结构如下图所示：

![image-20230318142305072](https://raw.githubusercontent.com/Noregret327/picture/master/img202303181423197.png)

#### 2、例化语句

下面是例化语句结构：

![image-20230318142321797](https://raw.githubusercontent.com/Noregret327/picture/master/img202303181423920.png)

例化语句分为：

1.端口关联法；如U2的两个语句

2.位置关联法：如U1的语句

#### 3、程序

```Verilog
module f_adder(ain,bin,cin,cout,sum);
    output cout,sum; input ain,bin,cin;
    wire net1,net2,net3;
    h_adder U1(ain,bin,net1,net2);
    h_adder U2(.A(net1),.SO(sum),.B(cin),.CO(net3));
endmodule
```



### 六、8位加法器设计及算术操作符应用

#### 程序

```verilog
module ADDER8B(A,B,CIN,COUT,DOUT);
    output [7:0] DOUT;
    output COUT;
    input [7:0] A,B;
    input CIN;
    wire [8:0] DATA;		//加操作的进位自动进入DATA[8]
    assign DATA = A + B + CIN;
    assign COUT = DATA[8];
    assign DOUT = DATA[7:0];
endmodule

module ADDER8B(A,B,CIN,COUT,DOUT);
    output [7:0] DOUT;
    output COUT;
    input [7:0] A,B;
    input CIN;				//加操作的进位进入并位COUT
    assign {COUT,DOUT} = A + B + CIN;  //{COUT,DOUT}=[8:0]
endmodule
```



### 七、层次化设计

##### 两种设计方法

电子元件设计者——自底向上

逻辑设计者——自上向下

![image-20230318142235095](https://raw.githubusercontent.com/Noregret327/picture/master/img202303181422227.png)

![image-20230301212834668](https://raw.githubusercontent.com/Noregret327/picture/master/img202303181421976.png)





### 八、Quartus使用

#### ！！！注意！！！

##### if else和case区别：

if：它有优先级，会生成优先编码器

case：它没有优先级

##### 避免Latch的产生

Latch其实就是锁存器，是一种在异步电路系统中，对输入信号电平敏感的单元，用来存储信息。

锁存器在数据未锁存时，输出端的信号随输入信号变化，就像信号通过一个缓冲器，一旦锁存信号有效，则数据被锁存，输入信号不起作用。因此，锁存器也被称为透明锁存器，指的是不锁存时输出对于输入是透明的。

！！！在同步电路中避免产生！！！

Latch的危害：对毛刺敏感、占用更多逻辑资源、不能异步复位、额外的延时、复杂的静态时序分析

Latch产生的情况有：

​	1.组合逻辑中if-else条件分支语句缺少else语句

​	2.组合逻辑中case条件分支语句条件未完全列举，且缺少default语句

​	3.组合逻辑中输出变量赋值给自己

##### 异步电路与同步电路：

异步电路：异步电路主要是组合逻辑电路，用于产生FIFO或RAM的读写控制信号脉冲，但它同时也用在时序电路中，此时它没有统一的时钟，状态变化的时刻是不稳定，通常输入信号只在电路处于稳定状态时才发生变化。

同步电路：同步电路是由时序电路（寄存器和各种触发器）和组合逻辑电路构成的电路，其所有操作都是在严格的时序控制下完成的。这些时序电路共享一个时钟CLK，二所有的状态变化都是在时钟的上升沿（或下降沿）完成的。

##### 寄存器

寄存器具有存储功能，一般是由D触发器构成，由时钟脉冲控制，每个D触发器（D Flip Flop，DFF）能够存储一位二进制码。

D触发器的工作原理：在一个脉冲信号（一般为晶振产生的时钟脉冲）上升沿或下降沿的作用下，将信号从输入端D送到输出端Q，如果时钟脉冲的边沿信号未出现，即使输入信号改变，输出信号仍然保持原值，且寄存器拥有复位清零功能，其复位又分为同步复位和异步复位。

##### 竞争冒险

竞争（Race Condition）指的是多个信号在同一时间影响同一个电路元件的状态，而由于它们到达时间的微小差异，导致该电路元件的输出结果不确定。例如，在一个由两个输入端口和一个输出端口组成的电路中，两个输入端口同时向输出端口发送信号，但是由于两个信号到达的时间不同，可能导致输出端口输出的信号并不是我们期望的结果。

冒险（Hazard）指的是在电路的输入信号发生变化时，在输出信号的某个瞬间发生的短暂错误。例如，在一个由与门和非门组成的电路中，当输入信号 A 和输入信号 B 同时改变时，可能会出现短暂的错误输出。

为了避免竞争和冒险的出现，我们可以采用以下措施：

1. 添加时钟边沿：在输入信号发生改变时，电路只有在时钟边沿处才能够进行运算。这样可以确保所有的输入信号都稳定之后再进行运算，从而消除竞争和冒险的出现。
2. 添加缓冲器：在电路的输入信号发生改变时，我们可以通过添加缓冲器的方式，使信号的变化尽可能的平滑，从而减少竞争和冒险的出现。
3. 添加延时电路：为了确保输入信号在电路内部的传输时间是相同的，我们可以在输入端口添加适当的延时电路，以保证输入信号的到达时间相同。这样可以避免因为信号到达时间不同而导致的竞争和冒险。

##### 阻塞赋值与非阻塞赋值

阻塞赋值：符号“=”，对应电路往往与触发沿没关系，只与输入电平的变化有关系。（直到赋值完成才允许下一条赋值语句执行）

非阻塞赋值：符号“<=”，对应电路往往与触发边沿有关系，只有在触发沿的时刻才能进行非阻塞赋值。（在计算非阻塞语句赋值号右边的语句和更新赋值左边的语句期间，允许其他的Verilog语句同时进行操作）

！！！非阻塞操作只能用于对寄存器类型变量进行赋值，因此只能用于“initial”和“always”块中，不允许用于连续赋值“assign”。

编写组合逻辑时使用阻塞赋值，编写时序逻辑时使用非阻塞赋值！！！

##### 分频器

分频，就是把输入信号的频率变成成倍数地低于输入频率的输出信号。

分频器分为偶数分频器和奇数分频器，和计数器非常类似，有时候甚至可以说就是一个东西。

！！！有分频和降频方法，建议使用降频方法

##### 



#### 1、创建工程项目

1.“File”——创建工程项目

2.设置文件路径，设置工程项目名

3.添加文件

4.选择设备，选择设备参数“FPGA”-“256”-“8”，选择设备“EP4CE6F17C8”

5.选择仿真软件

#### 2、使用软件打开RTL

1.先编译好文件再运行

2.打开“Task”——“Comple Design”——“Analysis & Synthesis”——“Netlist Viewers”——“RTL Viewer”

![image-20230318142214681](https://raw.githubusercontent.com/Noregret327/picture/master/img202303181422780.png)

#### 3、仿真设置

1.编写并添加“tb”文件——在“sim”文件夹中编写保存

2.打开设置——“Assignments”——“setting”

3.选择“Simulation”——选择工具——添加“testbench”

4.“new”——设置test bench名字——设置仿真时间——添加test bench文件

5.点击“RTL Simulation”

