# 神经网络NN

## T

**1）RetinaNet**

**[pytorch-retinanet](https://github.com/yhenon/pytorch-retinanet)**

**2）TensorRT量化**

**[mmyolo_tensorrt](https://github.com/thb1314/mmyolo_tensorrt)**

本项目系统介绍了YOLO系列模型在TensorRT上的量化方案，工程型较强，我们给出的工具可以实现不同量化方案在Yolo系列模型的量化部署

**3）FastestDet**

**[FastestDet](https://github.com/dog-qiuqiu/FastestDet)**







## 01目标检测

目标检测主要是将图像中的物体检测出来并进行定位，是用于图像处理的神经网络，目标检测适用于需要物体位置和形状信息的场景，如自动驾驶、视频监控、物体识别等。

- **DenseNet：**是一种<font color=ffa500>**密集连接的卷积神经网络**</font>，它的主要特点是在网络中的每个层之间建立了直接的连接。这种连接方式使得网络中的每个层都可以直接访问前面所有层的特征图，从而使得网络可以更好地利用前面层的特征信息。DenseNet的网络结构如下图所示，其中每个密集块（dense block）由多个卷积层组成，每个卷积层的输入都是前面所有层的特征图的拼接。DenseNet的优点是可以有效地缓解梯度消失问题，同时还可以减少参数量，提高模型的训练效率。【DenseNet在一些小数据集和需要更好的特征传递的任务中表现出色。在人脸识别、行为分析等方面可以帮助公安机关快速准确地锁定犯罪嫌疑人；在车辆检测、交通拥堵预测等方面可以提高交通运营效率；在医学影像分析、疾病预测等方面可以为医生提供更加准确的诊断依据。】
- **ResNet：**是一<font color=ffa500>**种残差网络**</font>，它的主要特点是在网络中引入了残差块（residual block），这种块可以让网络更加深，同时还可以缓解梯度消失问题。ResNet的网络结构如下图所示，其中每个残差块由两个卷积层和一个跨层连接组成。这种跨层连接可以让网络直接学习残差，从而使得网络更加容易训练。ResNet在很多图像识别任务中表现出色，特别是在大规模数据集上。【在很多图像识别任务中表现出色，特别是在大规模数据集上，同时也被广泛应用于目标检测领域】
- **R-FCN：**是一种基于<font color=ffa500>**全卷积网络的目标检测算法**</font>，它使用位置敏感的RoI池化层来实现位置不变性，从而提高了检测精度和速度。



<font color=red>**SSD和YOLO都是one-stage方法，相比传统的two-stage方法具有更快的检测速度和更高的检测精度。**</font>

- **SSD：**是一种one-stage方法，可以在单个卷积神经网络中完成目标检测，而不需要使用多个网络。使用卷积层来提取特征，然后使用卷积层来进行检测。采用多尺度的特征图用于检测，采用卷积进行检测，设置先验框。
- **YOLO：**是一种one-stage方法，可以在单个卷积神经网络中完成目标检测，而不需要使用多个网络。使用卷积层来提取特征，然后使用卷积层来进行检测。将输入图像划分为SxS个网格，并为每个网格预测B个边界框，以及每个边界框的物体类别和置信度。它使用一个单一的卷积神经网络同时预测所有网格的边界框和分类置信度。
- **DarkNet19：**DarkNet19是一个深度为19层的卷积神经网络，用于图像分类和检测任务。它是YOLOv2目标检测算法的主干网络，它的优点在于具有较少的参数和计算量，在计算速度和精度之间取得了良好的平衡，同时在训练过程中也具有较高的准确率和收敛速度。



- **AlexNet：**<font color=green>**2012年**</font>ImageNet比赛的冠军，它的主要特点是网络结构更深，包含5层卷积和3层全连接，同时使用了数据增强、Dropout方法、ReLU激活函数和Local Response Normalization（LRN）等技术来改进模型的训练过程。

- **Inception**：<font color=green>**2014年**</font>Inception是由Google提出的一种深度卷积神经网络，其主要特点是采用了多个Inception模块，通过1x1、3x3和5x5的卷积核进行特征提取，同时使用了全局平均池化层，减少了参数数量，提高了模型的泛化能力。Inception在ImageNet数据集上的表现也非常优秀。

- **GoogLeNet**：GoogLeNet在ImageNet数据集上的表现优异，获得了<font color=green>**2014年**</font>ImageNet图像分类挑战赛的冠军。<font color=green>**2015年**</font>GoogLeNet是一种基于Inception模块的深度卷积神经网络，它的网络不仅有深度，还在横向上具有“宽度”。其主要特点是采用了22层网络，使用了多个Inception模块，通过1x1、3x3和5x5的卷积核进行特征提取，同时使用了全局平均池化层，减少了参数数量，提高了模型的泛化能力。使用多通路的设计形式，每个支路使用不同大小的卷积核，最终输出特征图的通道数是每个支路输出通道数的总和。这将会导致输出通道数变得很大，尤其是使用多个Inception模块串联操作的时候，模型参数量会变得非常大。为了减小参数量，Inception模块使用了1x1卷积层来控制输出通道数。
- **VGG：**VGG通过使用一系列大小为3x3的小尺寸卷积核和pooling层构造深度卷积神经网络，并取得了较好的效果。VGG网络的设计严格使用3×3的卷积层和池化层来提取特征，并在网络的最后面使用三层全连接层，将最后一层全连接层的输出作为分类的预测。在VGG中每层卷积将使用ReLU作为激活函数，在全连接层之后添加dropout来抑制过拟合。
- **VGG-19**：VGG-19是由牛津大学计算机视觉组提出的一种深度卷积神经网络，主要用于图像分类和识别。，其主要特点是采用了19层网络，所有卷积层都采用了3x3的卷积核，同时使用了池化层和全连接层。VGG-19的网络结构非常简单，但是参数数量较多，训练时间较长。VGG-19在ImageNet数据集上的表现也非常优秀。





## 02语义分割的神经网络

语义分割是对图像进行像素级分类，预测每个像素属于的类别，不区分个体。语义分割适用于对图像进行精细分割和分类，如无人驾驶、人像分割、智能遥感、医疗影像分析等。

<font color=red>**采用two-stage方法（传统方法），如Faster R-CNN，需要使用两个网络来完成目标检测。**</font>

- **CNN：**用于<font color=ffa500>**图像分类和识别**</font>的神经网络，使用卷积层和池化层来提取图像的特征，然后使用全连接层来进行分类。主要结构包括卷积层、池化层和全连接层。

- **R-CNN：**是<font color=red>**区域卷积神经网络**</font>，用于<font color=ffa500>**目标检测和语义分割**</font>。使用卷积神经网络来提取特征，然后使用支持向量机（SVM）来进行分类，以实现目标检测和语义分割。
- **Fast R-CNN：**主要用于<font color=ffa500>**目标检测和语义分割**</font>，Fast R-CNN是一种改进的R-CNN，它使用共享卷积层来处理整个图像和所有感兴趣区域，而不是独立地处理每个感兴趣区域，从而提高了效率。
- **Mask R-CNN：**是一种用于<font color=ffa500>**目标检测和语义分割的卷积神经网络**</font>，它使用共享卷积层来处理整个图像和所有感兴趣区域，而不是独立地处理每个感兴趣区域，从而提高了效率。Mask R-CNN扩展了Faster R-CNN，通过添加一个用于预测对象掩码的分支，与现有的用于边界框识别的分支并行，从而实现了目标检测和语义分割。



## 03YOLO发展

- **YOLOv1：**于**2016年**提出，是one-stage检测的开创者，将目标检测问题转化为回归问题，使用一个卷积神经网络直接输出目标的位置和类别，达到了45FPS的速度，但存在难以检测小目标和定位精度不高的问题。
- **YOLOv2：**于**2017年**提出，对YOLOv1进行了多方面的改进，包括更换骨干网络为Darknet-19，引入anchor box，使用passthrough层融合高低层特征，使用全局平均池化代替全连接层，使用批量归一化和多尺度训练等，提升了检测的精度和速度，达到了67FPS。
- **YOLOv3：**于**2018年**提出，是YOLO之父Joseph Redmon的最后一作，引入了残差网络模块，构建了Darknet-53骨干网络，使用特征金字塔网络在三个不同的尺度上进行检测，使用多标签分类和逻辑斯蒂回归代替softmax分类，使用GIOU损失函数优化边界框，实现了较好的检测效果，但速度有所下降，为22FPS。
- **YOLOv4：**于**2020年**提出，由Alexey Bochkovskiy等人接力完成，综合了多种目标检测的最新技术，包括CSPDarknet53骨干网络，SPP空间金字塔池化，PAN特征金字塔网络，Mish激活函数，Mosaic数据增强，CIOU损失函数，DIOU非极大值抑制等，达到了63FPS的速度和43.5%的mAP。
- **YOLOv5：**于**2020年**提出，由Glenn Jocher等人开发，基于PyTorch框架，使用CSPResNet50骨干网络，引入了Focus和SPP模块，使用PANet和FPN进行特征融合，使用CIOU损失函数，使用WBF非极大值抑制，支持多种模型尺寸和自动模型压缩，具有较高的灵活性和可扩展性，速度和精度均有提升，达到了140FPS的速度和50.4%的mAP。
- **YOLOv6：**于**2021年**提出，由Alexey Bochkovskiy等人开发，基于YOLOv4的改进，使用CSPResNeXt50骨干网络，引入了SAM和ECO模块，使用PANet和BiFPN进行特征融合，使用Mish激活函数，使用Mosaic和MixUp数据增强，使用CIOU损失函数，使用DIOU非极大值抑制，速度和精度均有提升，达到了96FPS的速度和48.1%的mAP。
- **YOLOv7：**于**2021年**提出，由Alexey Bochkovskiy等人开发，基于YOLOv6的改进，使用CSPResNeXt101骨干网络，引入了SAM和ECO模块，使用PANet和BiFPN进行特征融合，使用Mish激活函数，使用Mosaic和MixUp数据增强，使用CIOU损失函数，使用DIOU非极大值抑制，速度和精度均有提升，达到了76FPS的速度和50.9%的mAP。
- **YOLOv8：**于**2023年**提出，Ultralytics公司开发和维护的。使用CSPResNeXt101作为骨干网络，使用SAM和ECO模块，增强了特征的选择性和复杂性；使用PANet和BiFPN进行特征融合，提高了多尺度检测的性能；使用Mish激活函数，优化了网络的非线性表达；使用Mosaic和MixUp数据增强，提升了模型的泛化能力和鲁棒性；使用CIOU损失函数，改善了边界框的定位精度；使用DIOU非极大值抑制，减少了重叠边界框的数量。YOLOv8在COCO数据集上，在目标检测任务上，YOLOv8达到了76FPS的速度和50.9%的mAP；在图像分割任务上，YOLOv8达到了63FPS的速度和43.5%的mAP。

[YOLO家族进化史（v1-v7） - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/539932517)

[了解YOLO的发展历史和不同版本之间的差异-CSDN博客](https://blog.csdn.net/jiong9412/article/details/132875120)

[YOLO系列目标检测算法研究进展 (ceaj.org)](http://cea.ceaj.org/CN/10.3778/j.issn.1002-8331.2301-0081)

[《YOLO全面回顾：从YOLOV1到现在及未来》 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/641297736)

[YOLO对象检测算法又又又更新了，YOLOv8横空出世 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/599191661)

[YOLO系列详解：YOLOv1、YOLOv2、YOLOv3、YOLOv4、YOLOv5、YOLOv6、YOLOv7-CSDN博客](https://blog.csdn.net/qq_40716944/article/details/114822515)



## 04轻量化模型资源对比

### ZYNQ 7020资源

**PL端**

- 逻辑单元Logic Cells： 85K
- 查找表 LUTs: 53,200
- 触发器(flip-flops): 106,400
- 乘法器 18x25MACCs： 220
- Block RAM： 4.9 Mb
- 两个 AD 转换器,可以测量片上电压、温度感应和高达 17 外部差分输入通道， 1MBPS

### 1）基于宇航级FPGA的YOLOv5s 网络模型硬件加速（YOLOv5）

![image-20231210143339151](https://raw.githubusercontent.com/Noregret327/picture/master/202312101433192.png)

### 2）基于FPGA的卷积神经网络硬件加速器设计（YOLOv3）

![image-20231210143356081](https://raw.githubusercontent.com/Noregret327/picture/master/202312101433122.png)

### 3）基于FPGA的卷积神经网络和视觉Transformer通用加速器

![image-20231210143418392](https://raw.githubusercontent.com/Noregret327/picture/master/202312101434444.png)

### 4）面向FPGA部署的改进YOLO铝片表面缺陷检测系统（YOLOv2）

![image-20231210143442909](https://raw.githubusercontent.com/Noregret327/picture/master/202312101434953.png)

### 5）基于FPGA的快速带钢表面缺陷检测系统设计（VGG-19）

![image-20231210143500648](https://raw.githubusercontent.com/Noregret327/picture/master/202312101435687.png)

### 6）基于FPGA的卷积神经网络硬件加速器设计（YOLOv2）

![image-20231210143517343](https://raw.githubusercontent.com/Noregret327/picture/master/202312101435387.png)

### 7）基于ZYNQ平台的通用卷积加速器设计（YOLOv3）

![image-20231210143536693](https://raw.githubusercontent.com/Noregret327/picture/master/202312101435731.png)

### 8）基于ZYNQ的卷积神经网络加速器设计（YOLOv2）

![image-20231210143553218](https://raw.githubusercontent.com/Noregret327/picture/master/202312101435252.png)

![image-20231210143629759](https://raw.githubusercontent.com/Noregret327/picture/master/202312101436787.png)

### 9）基于改进YOLOv4-Tiny的FPGA加速方法（YOLOv4）

![image-20231210143618843](https://raw.githubusercontent.com/Noregret327/picture/master/202312101436899.png)

### 10）YOLOv3-tiny的硬件加速设计及FPGA实现（YOLOv3）

![image-20231210143651062](https://raw.githubusercontent.com/Noregret327/picture/master/202312101436108.png)

### 11）基于Winograd的YOLO节能优化框架FPGA加速器（YOLOv2）

《A Power-Efficient Optimizing Framework FPGA Accelerator Based on Winograd for YOLO》-2020年

针对PYNQ架构下的深度学习目标检测模型YOLO，提出了一种基于Winograd算法的加速器设计方法。

![image-20231211151921888](https://raw.githubusercontent.com/Noregret327/picture/master/202312111519035.png)

![image-20231211152121963](https://raw.githubusercontent.com/Noregret327/picture/master/202312111521004.png)

![image-20231211152044756](https://raw.githubusercontent.com/Noregret327/picture/master/202312111520796.png)

![image-20231211152017665](https://raw.githubusercontent.com/Noregret327/picture/master/202312111520707.png)



### 12）基于Xilinx ZYNQ FPGA的深度学习YOLO网络改进算法（YOLOv2）

《An improved algorithm for deep learning YOLO network based on Xilinx ZYNQ FPGA》-2020年

![image-20231211153415381](https://raw.githubusercontent.com/Noregret327/picture/master/202312111534417.png)

![image-20231211153513114](https://raw.githubusercontent.com/Noregret327/picture/master/202312111535155.png)

![image-20231211153615354](https://raw.githubusercontent.com/Noregret327/picture/master/202312111536392.png)

### 13)基于可重构硬件加速器的YOLO v3微型FPGA体系结构用于实时感兴趣区域检测（YOLOv3）

《A YOLO v3-tiny FPGA architecture using a reconfigurable hardware accelerator for real-time region of interest detection》-2023年

海上救援的

![image-20231211190119061](https://raw.githubusercontent.com/Noregret327/picture/master/202312111901107.png)









## 05YOLO国内外应用

### 1）E-YOLO:基于改进YOLOv8n模型的发情期奶牛识别（Estrus-YOLO）

《E-YOLO: Recognition of estrus cow based on improved YOLOv8n model》-2024年

![image-20231210143232611](https://raw.githubusercontent.com/Noregret327/picture/master/202312101432365.png)

### 2）基于YOLO v5算法的太阳能电池板缺陷检测设计（YOLO v5）

《Solar panel defect detection design based on YOLO v5 algorithm》-2023年

![image-20231210162229473](https://raw.githubusercontent.com/Noregret327/picture/master/202312101622563.png)

![image-20231210162251907](https://raw.githubusercontent.com/Noregret327/picture/master/202312101622960.png)

![image-20231210161659690](https://raw.githubusercontent.com/Noregret327/picture/master/202312101616724.png)

### 3）用FPGA实现两层感知器级联神经网络，实现高效的实时手势跟踪（手势识别）

《FPGA implementation of two multilayer perceptron neural network in cascade for efficient real time hand gestures tracking》-2023年

![image-20231211193819893](https://raw.githubusercontent.com/Noregret327/picture/master/202312111938963.png)

![image-20231211193631561](https://raw.githubusercontent.com/Noregret327/picture/master/202312111936774.png)

![image-20231211193902434](https://raw.githubusercontent.com/Noregret327/picture/master/202312111939567.png)

![image-20231211194025065](https://raw.githubusercontent.com/Noregret327/picture/master/202312111940229.png)

![image-20231211194112267](https://raw.githubusercontent.com/Noregret327/picture/master/202312111941323.png)









## 06计算机视觉领域CV

### 原理

**原理：**计算机视觉需要大量数据。 它一遍又一遍地运行数据分析，直到能够辨别差异并最终识别图像为止。

计算机视觉问题是指在这个领域中需要解决的各种问题。这些问题包括但不限于<font color=green>**图像分类、物体检测、语义分割、目标跟踪、人脸识别、姿态估计、三维重建、深度估计、光流估计、图像增强、图像去噪、图像修复、图像生成、图像超分辨率、视频分析、视频摘要、视频生成等**</font>

这个过程会用到两种关键技术：一种是机器学习，叫做深度学习，另一种是卷积神经网络 (CNN)。

**机器学习:**使用算法模型，让计算机能够自行学习视觉数据的上下文。 如果通过模型馈入足够多的数据，计算机就能"查看"数据并通过自学掌握分辨图像的能力。 算法赋予机器自学的能力，而无需人类编程来使计算机能够识别图像。

**CNN:** 将图像分解为像素，并为像素指定标记或标签，从而使机器学习或深度学习模型能够"看"到物体。 它使用标签来执行卷积运算（用两个函数产生第三个函数的数学运算）并预测它"看到"的东西。 该神经网络运行卷积运算，并通过一系列迭代检验预测准确度，直到预测开始接近事实。 然后它以类似于人类的方式识别或查看图像。

图片来源：[一文读懂图像分类、目标定位、语义分割与实例分割的区别_语义分割和普通分类的区别-CSDN博客](https://blog.csdn.net/dxh0907070012/article/details/108631745)

![img](https://img-blog.csdnimg.cn/20200916220912791.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R4aDA5MDcwNzAwMTI=,size_16,color_FFFFFF,t_70)

### 分类

- **目标检测：**图像中的物体检测出来并进行定位
- **语义分割：**对图像进行像素级分类，预测每个像素属于的类别，不区分个体
- **图像分类：**是进行深度学习研究与学习的基本任务，其主要是在已知类别数量的情况下，通过输入一张图片，来判断图片所属类别。【只需要判断图像中包含<font color=red>**物体的类别**</font>】
- **目标定位：**则是在图像分类的基础上，进一步判断图像中的目标具体在图像的什么位置，通常是以包围盒的(bounding box)形式进行定位。在目标定位中，通常只有一个或固定数目的目标，而目标**检测**更一般化，其图像中出现的目标种类和数目都不定。【<font color=red>**精确地定位出图像**</font>中某一物体类别信息和所在位置】
- **目标跟踪：**是指在视频序列中跟踪特定目标的位置和运动状态
- **姿态估计：**是指从图像或视频中估计出人体或物体的姿态
- **三维重建：**是指从图像或视频中重建出三维场景
- **深度估计：**是指从单张图像或图像序列中估计出场景的深度信息
- **语义分割：**是目标检测更进阶的任务，目标检测只需要框出每个目标的包围盒，语义分割需要进一步判断图像中哪些像素属于哪个目标。但是，语义分割不区分属于相同类别的不同实例。例如，当图像中有多只猫时，语义分割会将两只猫整体的所有像素预测为“猫”这个类别。【将图像中所有<font color=red>**像素进行分类**</font>】
- **实例分割**：需要区分出哪些像素属于第一只猫、哪些像素属于第二只猫。【区分<font color=red>**相同类别中不同个体**</font>】
- **图像增强：**是指通过对图像进行处理来改善其质量
- **图像去噪：**是指通过对图像进行处理来去除噪声
- **图像修复：**是指通过对图像进行处理来修复损坏的部分
- **图像生成：**是指使用计算机生成新的图像
- **人脸识别：**是指从图像或视频中识别出人脸并进行身份验证或识别
- **光流估计：**是指从图像序列中估计出相邻帧之间的像素运动



### 论文综述

以下论文来源：[25篇最新CV领域综述性论文速递！涵盖15个方向：目标检测/图像处理/姿态估计/医学影像/人脸识别等方向-腾讯云开发者社区-腾讯云 (tencent.com)](https://cloud.tencent.com/developer/article/1629264)

###### CNN

[A Survey of Convolutional Neural Networks: Analysis, Applications, and Prospects](https://arxiv.org/abs/2004.02806)

本文对**119篇**相关文献进行了梳理，由**华东理工大学**学者发布。本文旨在在卷积神经网络这个快速增长的领域中尽可能提供新颖的想法和前景，不仅涉及二维卷积，而且涉及一维和多维卷积。首先，本文简要介绍了CNN的历史并概述了CNN发展，介绍经典CNN模型，重点论述使它们达到SOTA的关键因素，并通过实验分析提供了一些经验法则，最后对一维，二维和多维卷积的应用进行了概述。



###### 目标检测

[Deep Domain Adaptive Object Detection: a Survey - arXiv.org](https://arxiv.org/abs/2002.06797)

本文共梳理了**40篇**相关文献，由**中科院自动化所**学者发布。基于深度学习(DL)的目标检测已经取得了很大的进展，这些方法通常假设有大量的带标签的训练数据可用，并且训练和测试数据从相同的分布中提取。然而，这两个假设在实践中并不总是成立的。深域自适应目标检测(DDAOD)作为一种新的学习范式应运而生。本文综述了深域自适应目标检测方法的研究进展。

[Anomalous Example Detection in Deep Learning: A Survey](https://arxiv.org/abs/2003.06979)

本文共梳理了**119篇**相关文献，由**雪城大学**学者发布。讨论多种异常实例检测方法，并分析了各种方法的相对优势和劣势。

[Moving objects detection with a moving camera: A comprehensive review](https://www.sciencedirect.com/science/article/pii/S157401372030410X)

本文共梳理了**347篇**相关文献。随着移动传感器的兴起，研究移动相机逐渐变为热门方向。本文对不同现有方法进行了识别，并将其分为一个平面或多个两类。在这两个类别中，将各类方法分为**8组**：全景背景减法，双摄像头，运动补偿，子空间分割，运动分割，平面+视差，多平面和按块分割图像。本文还对公开可用的数据集和评估指标进行了研究。



###### 视觉/其他

[On Information Plane Analyses of Neural Network Classifiers -- A Review](https://arxiv.org/abs/2003.09671)

[A Survey of Methods for Low-Power Deep Learning and Computer Vision](https://ieeexplore.ieee.org/document/9221198/)

[When Deep Learning Meets Data Alignment: A Review on Deep Registration Networks (DRNs)](https://arxiv.org/abs/2003.03167)

[Toward Unconstrained Palmprint Recognition on Consumer Devices: A Literature Review](https://ieeexplore.ieee.org/document/9085344/)

[Features for Ground Texture Based Localization -- A Survey](https://arxiv.org/abs/2002.11948)

[From Seeing to Moving: A Survey on Learning for Visual Indoor Navigation (VIN)](https://arxiv.org/abs/2002.11310)



###### 图像分类

[A survey on Semi-, Self- and Unsupervised Learning for Image Classification](https://arxiv.org/abs/2002.08721)

本文共梳理了**51篇**相关文献。综述了标签较少的图像分类中常用的**21种**技术和方法。我们比较方法，并确定了三个主要趋势。



###### 深度相关（Deep）

[Monocular Depth Estimation Based On Deep Learning: An Overview](https://arxiv.org/abs/2003.06620)

本文对**119篇**相关文献进行了梳理，由**华东理工大学**学者发布。随着深度神经网络的迅速发展，基于深度学习的单眼深度估计已得到广泛研究。为了提高深度估计的准确性，提出了各种网络框架，损失函数和训练策略。因此，本文综述了当前基于深度学习的单眼深度估计方法，总结了几种基于深度学习的深度估计中广泛使用的数据集和评价指标，同时根据不同的训练方式回顾了一些有代表性的现有方法：有监督，无监督和半监督。



###### 图像去噪

[Deep learning on image denoising: An overview](https://www.sciencedirect.com/science/article/pii/S0893608020302665)

本文梳理了**238篇**相关文献，由**哈尔滨工业大学、广东工业大学、清华大学**学者共同发布。不同类型的处理噪声深度学习方法存在巨大差异，而目前很少有相关研究来进行相关总结。本文对图像去噪中不同深度学习技术进行了比较研究，分析不同方法的动机和原理，并在公共去噪数据集进行比较。研究包括：(1). 加白噪声图像的CNN；(2)用于真实噪声图像的CNN；(3)用于盲噪声去噪的CNN；(4)用于混合噪声图像的CNN。



###### 图像分割

[Image Segmentation Using Deep Learning: A Survey](https://ieeexplore.ieee.org/document/9356353)

本文梳理了**172篇**相关文献，对语义和实例分割文献进行了全面回顾，涵盖了的各种开创性作品，包括全卷积像素标记网络，编码器-解码器体系结构，多尺度以及基于金字塔的方法，递归网络，视觉注意模型以及对抗中的生成模型。



###### 人脸识别

[Deepfakes and beyond: A Survey of face manipulation and fake detection](https://www.sciencedirect.com/science/article/pii/S1566253520303110)

本文梳理了**105篇**相关文献，本文对操纵人脸的图像技术（包括DeepFake方法）以及检测此类技术的方法进行了全面综述。论述了四种类型的面部操作：全脸合成、面部身份交换（DeepFakes）、面部属性操作以及面部表情操作。



###### 姿态估计

[A review on object pose recovery: From 3D bounding box detectors to full 6D pose estimators](https://www.sciencedirect.com/science/article/pii/S0262885620300305)

本文梳理了**206篇**相关文献，由**伦敦帝国理工学院**学者发布。本文对3D边界框检测器到完整的6D姿态估计器的物体姿态恢复方法的进行了首次全面的综述。基于数学模型，将各类方法分为分类，回归，分类与回归，模板匹配和点对特征匹配任务。



###### 行为/动作识别

[A Survey on 3D Skeleton-Based Action Recognition Using Learning Method](https://arxiv.org/abs/2002.05907)

本文梳理了**81篇**相关文献，由**北京大学**学者发布。本文强调了动作识别的必要性和3D骨架数据的重要性，然后以数据驱动的方式对基于递归神经网络，基于卷积神经网络和基于图卷积网络的主流动作识别技术进行了全面介绍，这也是第一次对使用3D骨架数据进行基于深度学习的动作识别的全面研究。



###### 人群计数

[CNN-based Density Estimation and Crowd Counting: A Survey](https://arxiv.org/abs/2003.12783)

本文梳理了**222篇**相关文献，由**北京航空航天大学**学者发布，基于CNN的密度图估计方法，调研了**220+**工作，对人群计数进行了全面系统的研究。同时根据评估指标，在人群统计数据集上选择表现最好的三名，并分析其优缺点。



###### 医学影像

[A Comprehensive Review for Breast Histopathology Image Analysis Using ...](https://arxiv.org/abs/2003.12255)

本文梳理了**180篇**相关文献，由**东北大学**学者发布。对基于人工神经网络的BHIA技术进行了全面概述，将BHIA系统分为经典和深度神经网络以进行深入研究，分析现有模型以发现最合适的算法，并提供可公开访问的数据集。

[Medical Image Registration Using Deep Neural Networks: A Comprehensive Review](https://arxiv.org/abs/2002.03401)

本文梳理了**117篇**相关文献，对使用深度神经网络进行医学图像配准的最新文献进行了全面回顾，系统地涵盖了该领域的相关作品，包括关键概念，统计分析，关键技术，主要贡献，挑战和未来方向。

[Towards automatic threat detection: A survey of advances of deep learning within X-ray security imaging](https://www.sciencedirect.com/science/article/pii/S0031320321004258)

本文梳理了**151篇**相关文献，由**英国杜伦大学**学者发布。本文分常规机器学习和当代深度学习两类来回顾X射线安全成像算法。将深度学习方法分为有监督，半监督和无监督学习，着重论述分类，检测，分割和异常检测任务，同时包含有完善的X射线数据集。

[Deep neural network models for computational histopathology: A survey](https://www.sciencedirect.com/science/article/pii/S1361841520301778)

本文梳理了**130篇**相关文献，由**多伦多大学**学者发布。本文对组织病理学图像分析中使用的最新深度学习方法进行了全面回顾，包括有监督，弱监督，无监督，迁移学习等领域，并总结了几个现有的开放数据集。



###### 三维重建

[A Survey On 3D Inner Structure Prediction from its Outer Shape](https://arxiv.org/abs/2002.04571)

本文梳理了**81篇**相关文献，由**北京大学**学者发布。由于过去与骨架数据相关内容很少，本文是第一篇针对使用3D骨架数据进行基于深度学习的动作识别进行全面讨论的研究。本文突出了动作识别和3D骨架数据的重要性，以数据驱动的方式对基于递归神经网络、卷积神经网络和图卷积网络的主流动作识别技术进行了全面介绍。并介绍了最大的3D骨架数据集NTU-RGB+D及其新版本NTU-RGB+D 120，并论述了几种现有的顶级算法。



###### 三维点云

[Target-less registration of point clouds: A review - arXiv.org](https://arxiv.org/abs/1912.12756)

本文对**48篇**文献进行了梳理，总结了无目标点云配准的基本工作，回顾了三种常用的配准方法，即基于特征匹配的方法，迭代最近点算法和随机假设，并分析了这些方法的优缺点，介绍它们的常见应用场景。



###### OCR

[Handwritten Optical Character Recognition (OCR): A Comprehensive ...](https://arxiv.org/abs/2001.00139)

本文对**142篇**相关文献进行了梳理，总结了有关OCR的研究，综述了2000年至2018年之间发布的研究文章，介绍OCR的最新结果和技术，并分析研究差距，以总结研究方向。



