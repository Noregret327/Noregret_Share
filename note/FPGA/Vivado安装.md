# Vivado安装

## 1.官网地址

https://china.xilinx.com/support/download.html

## 2.安装选择

安装的时候都要选上：（只需要安装一次即可。）

Vitis：主要用于PS端

Vivido：主要用于PL端

### 3.License破解

1）将下面代码保存为：vivado_lic2037.lic文件格式

```
# ----- REMOVE LINES ABOVE HERE --------------------------
#
INCREMENT VIVADO_HLS xilinxd 2037.05 permanent uncounted AF3E86892AA2 \
	VENDOR_STRING=License_Type:Bought HOSTID=ANY ISSUER="Xilinx \
	Inc" START=19-May-2016 TS_OK
INCREMENT Vivado_System_Edition xilinxd 2037.05 permanent uncounted \
	A1074C37F742 VENDOR_STRING=License_Type:Bought HOSTID=ANY \
	ISSUER="Xilinx Inc" START=19-May-2016 TS_OK
PACKAGE Vivado_System_Edition xilinxd 2037.05 DFF4A65E0A68 \
	COMPONENTS="ISIM ChipScopePro_SIOTK PlanAhead ChipscopePro XPS \
	ISE HLS_Synthesis AccelDSP Vivado Rodin_Synthesis \
	Rodin_Implementation Rodin_SystemBuilder \
	PartialReconfiguration AUTOESL_FLOW AUTOESL_CC AUTOESL_OPT \
	AUTOESL_SC AUTOESL_XILINX petalinux_arch_ppc \
	petalinux_arch_microblaze petalinux_arch_zynq ap_sdsoc SDK \
	SysGen Simulation Implementation Analyzer HLS Synthesis \
	VIVADO_HLS" OPTIONS=SUITE
#
# ----- REMOVE LINES BELOW HERE --------------------------
```

2）Vivado 点击 Help->Manage License... 启动Vivado License Manager如图：

![image-20231008143056899](https://raw.githubusercontent.com/Noregret327/picture/master/202310081430948.png)

3）添加之前保存的vivado_lic2037.lic文件即可，激活成功为下图所示：

![image-20231008143246035](https://raw.githubusercontent.com/Noregret327/picture/master/202310081432089.png)