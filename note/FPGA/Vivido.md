# Vivido

## 简介：

##### 调试

调试FPGA设计是一个不断迭代的过程：

![image-20230421143845615](https://raw.githubusercontent.com/Noregret327/picture/master/202304211438696.png)

Xilinx硬件调试解决方案：

- Vivado工具集成了逻辑分析仪，用于替换外部的逻辑分析仪
- 添加ILA核和VIO核实现硬件调试
- 通过JTAG接口和PC连接

1.ILA ( Integrated Logic Analyzer）

- 监控逻辑内部信号和端口信号

2.VIO ( Virtual Input/Output )

- 实时监控和驱动逻辑内部信号和端口信号

##### 设计流程

![image-20230421151243815](https://raw.githubusercontent.com/Noregret327/picture/master/202304211512871.png)

仿真分类：

- 功能仿真

功能仿真也称为行为仿真，主旨在于验证申路的功能是否符合设计要求其特点是不考虑电路门延迟与线延迟主要是验证电路与理想情况是否一致。

- 时序仿真

时序仿真也称为布局布线后仿真，是指电路已经映射到特定的工艺环境以后，综合考虑电路的路径延迟与门延迟的影响，验证电路能否在一定时序条件下满足设计构想的过程，能较好地反映芯片的实际工作情况。