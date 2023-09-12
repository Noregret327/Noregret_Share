# 机器学习

## 1.入门实例 收入数据集

### 1.1 完整例子

```python
import torch

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


// 导入数据
data = pd.read_csv('dataset/Income1.csv')

// 数据处理
from torch import nn

X = torch.from_numpy(data.Education.values.reshape(-1, 1).astype(np.float32))
Y = torch.from_numpy(data.Income.values.reshape(-1, 1).astype(np.float32))

// 导入模型
model = nn.Linear(1, 1)  # out = w*input + b 等价于model(input)
// 损失函数
loss_fn = nn.MSELoss()   # 损失函数
// 优化参数
opt = torch.optim.SGD(model.parameters(), lr=0.0001)     # 优化参数

// 训练
for epoch in range(5000):
    for x, y in zip(X, Y):
        y_pred = model(x)              # 使用模型预测
        loss = loss_fn(y, y_pred)      # 根据预测结果计算损失
        opt.zero_grad()                # 把变量梯度清 0
        loss.backward()                # 求解梯度
        opt.step()                     # 优化模型参数
        
        
// 模型权重和模型偏置
model.weight
model.bias

```

```python
// 绘图
plt.scatter(data.Education, data.Income)
plt.xlabel('Education')
plt.ylabel('Income')
plt.plot(X.numpy(), model(X).data.numpy(), c='r')

```



### 1.2 分解写法

```python
import torch

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


// 导入数据
data = pd.read_csv('dataset/Income1.csv')
X = torch.from_numpy(data.Education.values.reshape(-1, 1).astype(np.float32))
Y = torch.from_numpy(data.Income.values.reshape(-1, 1).astype(np.float32))

/////////////分解写法///////////////////
# 新建变量
w = torch.randn(1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)

# 模型公式——	w@x + b
learning_rate = 0.0001

# 训练
for epoch in range(5000):
    for x, y in zip(X, Y):
        y_pred = torch.matmul(x, w) +b 			# w*x + b
        loss = (y - y_pred).pow(2).mean()		# 根据预测结果计算损失
		if not w.grad is None:					# 如果开始梯度没有值，那么我们把w的值的梯度置为0
            w.grad.data.zero_()
        if not b.grad is None:					# 如果开始梯度没有值，那么我们把b的值的梯度置为0
            b.grad.data.zero_()
        loss.backward()							# 损失值进行反向传播
        with torch.no_grad():					# 优化参数
            w.data -= w.grad.data*learning_rate
            b.data -= b.grad.data*learning_rate
        

///////////////////////////////////////
        
// 模型权重和模型偏置
w
b

// 绘图
plt.scatter(data.Education, data.Income)
plt.plot(X.numpy(), (X*w + b).data.numpy(), c='r')


```

### 1.3 模型创建过程

1. <font color=red size=5>创建模型——公式</font>
2. <font color=red size=5>计算损失值——loss</font>
3. <font color=red size=5>梯度清 0 ——zero_grad（）</font>
4. <font color=red size=5>反向传播——backward（）</font>
5. <font color=red size=5>优化参数——opt</font>



## 2.张量与数据类型

<font color=blue size=4>什么是张量？</font>

- 张量就是多维数组
- 维度：在具体的Python代码中，可以使用ndim来查看张量的维度
- 形状：在具体的Python代码中，可以使用shape来查看张量的形状
- 数据类型：在具体的Python代码中，可以使用dtype来查看张量的数据类型

<font color=blue size=4>张量：</font>

1. 0维张量：标量（scalar）——纯量就是一个实数，既可以是正数，也可以是负数，例如质量、体积、温度等。
2. 1维张量：向量（vector）——向量是由于数字组成的数组，可以包含多个数字，但只有一个维度。
3. 2维张量：矩阵（matrix）——矩阵具备两个维度，即row与column。
4. 3维张量与更高维张量——将多个矩阵组成一个新的数组，即可获得3维张量；将多个3维张量组成一个新的数组，即可获得4维张量，以此类推。

<font color=blue size=4>例子：</font>

- 矩阵数据（2维张量）：形状通常为(samples，features)，此处的samples代表样本，features代表特征。
- 图像（4维张量）：形状通常为（samples，height，width，channels），其中channels代表颜色通道，通常为RGB。
- 视频（5维张量）：形状通常为（samples，frames，height，width，channels），其中frames代表帧，每一帧都是一张彩色图像。

### 2.1 张量

```python
import torch

// torch的随机数、零、一
x = torch.rand(2, 3)    # 随机生成两行三列的随机数
x = torch.zeros(2, 3)   # 生成两行三列的全是 0 的数
x = torch.ones(2, 3, 4) # 生成两个三行四列全是 1 的数

// size（）和shape的区别
x.size()       			# 输出：torch.Size([2, 3, 4])
x.shape                 # 输出：torch.Size([2, 3, 4])
x.size(1)               # 返回对应位置的值      输出：3

// 修改张量类型
x = torch.tensor([6, 2], dtype=torch.float32)  # 修改张量的数据类型    输出：tensor([6., 2.])


```

### 2.2 tensor与ndarray数据类型的转换

```python
////////////////////////////////////////////////////////////////////////////////////////////////////////
// tensor与ndarray数据类型的转换
////////////////////////////////////////////////////////////////////////////////////////////////////////

import numpy as np

// 生成2行3列的随机数（array数据类型）
a = np.random.randn(2, 3)		# 输出：array类型

/*************  array——>tensor **********************/
b = torch.from_numpy(a)			# 输出：tensor类型

/*************  tensor——>array **********************/
b.numpy()						# 输出：array类型

```

### 2.3 张量相加

```python
////////////////////////////////////////////////////////////////////////////////////////////////////////
// 张量相加
////////////////////////////////////////////////////////////////////////////////////////////////////////

// 随机数组生成
x1 = np.random.randn(2, 3)		# 生成随机数组——X1				
x1 = torch.from_numpy(x1)		# 转换为张量

x2 = np.random.randn(2, 3)		# 生成随机数组——X2
x2 = torch.from_numpy(x2)		# 转为张量

// 张量相加方法
x1 + x2                    # +号相加
x1.add(x2)                 # 用add相加
x1.add_(x2)                 # 加下划线是覆盖原来的

```

### 2.4 数组的行列转换与排列

```python
x1 = np.random.randn(2, 3)		# 生成随机数组——X1	

x1.view(2, 3)           # 转为两行三列
x1.view(3, 2)           # 转为三行两列

x1.view(-1, 1)          # 重新排列（-1：排列； 1：表示一列）

x1.mean()              	# 求均值

x1.sum()               	# 求和
```

### 2.5 从张量求标量值 item()

```python
x1 = np.random.randn(2, 3)		# 生成随机数组——X1	
x = x1.sum()					# 输出：tensor(-10.7382, dtype=torch.float64)
x.item()  						# 输出：-10.738246420338335
```

### 2.6 张量的自动微分

将torch.tensor属性.requires_grad设置为True， pytorch将开始跟踪对此张量的所有操作。 完成计算后，可以调用.backward()并自动计算所有梯度。 该张量的梯度将累计加到.grad属性中。

```python
# requires_grad: 如果需要为张量计算梯度，则为True，否则为False。我们使用pytorch创建tensor时，可以指定requires_grad为True（默认为False）

x = torch.ones(2, 2, requires_grad=True)		# 输出：tensor([[1., 1.], [1., 1.]], requires_grad=True)
```

### 2.7 tensor的数据结构

```python
// tensor的结构：data、grad、grad_fn
# data：张量的数据/值

# grad_fn： grad_fn用来记录变量是怎么来的，方便计算梯度，y = x*3,grad_fn记录了y由x计算的过程。

# grad：当执行完了backward()之后，通过x.grad查看x的梯度值。

// data
x.data
// grad
x.grad
// grad_fn
x.grad_fn
```



## 3.逻辑回归/二分类问题

<font color=blue size=4>线性回归：</font>

线性回归预测的是一个<font color=red size=4>连续值</font>

<font color=blue size=4>逻辑回归：</font>

逻辑回归给出的<font color=red size=4>“是”和“否”</font>>的回答——逻辑回归模型是单个神经元：计算输入特征的加权和，然后使用一个激活函数（或传递函数）计算输出

### 3.1 sigmoid函数——激活层

sigmoid函数是一个概率分布函数，给定某个输入，它将输出为一个概率值。

![](https://raw.githubusercontent.com/Noregret327/picture/master/202309112100206.png)

### 3.2 逻辑回归损失函数：

平方差所惩罚的是与损失为同一数量级的情形

对于<font color=#00cc00 size=5>分类问题</font>，我们最好的使用交叉熵损失函数会更有效

<font color=red size=5>交叉熵</font><font color=0066cc size=5>会输出一个更大的“损失”</font>

### 3.3 交叉熵——损失函数

假设概率分布p为期望输出，概率分布q为实际输出，H（p，q）为交叉熵，则：
$$
H(p,q) = - \sum{p(x)\log{q(x)}}
$$
在pytorch里，我们使用<font color=red size=6>“nn.BCELoss()”</font>来计算二元交叉熵

### 3.4 数据预处理

```python
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 导入数据，并不要表头
data = pd.read_csv('dataset/credit-a.csv', header=None)

# 查看数据
data
# 查看数据信息
data.info()

# 提取数据
X = data.iloc[:, :-1]             				# 取出所有行、除了最后一列不要
Y = data.iloc[:, -1].replace(-1, 0)             # 取出所有行、最后一列，并将 -1 替换为 0
Y.unique()                        				# 查看最后一列的值（原：-1，1）（现：0，1）

# 数据预处理
X = torch.from_numpy(X.values).type(torch.float32)           		# 把X的值转为tensor，再把tensor转为浮点数
X.shape											# torch.Size([653, 15])		——>		635组数据，15个特征
Y = torch.from_numpy(Y.values.reshape(-1, 1)).type(torch.float32)	# 先把Y的值转换为1列，再把Y的值转为tensor，最后把tensor转为浮点数
Y.size()										# torch.Size([653, 1])		——>		635组数据，1个输出


##########################################
# 输入有15个特征，输出只有一个
##########################################


```

### 3.5 模型的创建

```python
# 新建模型
model = nn.Sequential(
                nn.Linear(15, 1),                # 线性层
                nn.Sigmoid()                     # 激活——将输出转为概率值
)

# 模型
model

# 损失函数和优化函数
loss_fn = nn.BCELoss()
opt = torch.optim.Adam(model.parameters(), lr=0.0001)

# 迭代次数
batches = 16
no_of_batch = 653//16
epoches = 1000

# 训练
for epoch in range(epoches):
    for i in range(no_of_batch):
        start = i*batches
        end = start + batches
        x = X[start: end]
        y = Y[start: end]
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
     
    
model.state_dict()                        # sigmoid(w1*w1 + w2*w2 + ... + w15*w15 + b)

# 求出输出
((model(X).data.numpy() > 0.5).astype('int') == Y.numpy()).mean()
```



## 4.多层感知器

与上一节学习的逻辑回归模型是单个神经元：计算输入的特征的加权和，然后使用一个激活函数（或传递函数）计算输出

### 4.1 单个神经元（二分类）

![image-20230912104048481](https://raw.githubusercontent.com/Noregret327/picture/master/202309121040587.png)

