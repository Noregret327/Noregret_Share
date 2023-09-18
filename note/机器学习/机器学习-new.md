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

### 4.2 多个神经元（多分类）

![image-20230912104624545](https://raw.githubusercontent.com/Noregret327/picture/master/202309121046620.png)

### 4.3 单层神经元的缺陷

1. 无法拟合<font color=red size=4>“异或”运算</font>（异或问题看似简单，使用单层的神经元确实没有办法解决）
2. 神经元要求数据必须是线性可分的（异或问题无法找到一条直线分割两个类）



### 4.4 多层感知器

![image-20230912105205109](https://raw.githubusercontent.com/Noregret327/picture/master/202309121052179.png)

### 4.5 激活函数

激活函数带有非线性，使模型拟合大大增强，可以处理更加复杂的多分类问题

![image-20230912105259742](https://raw.githubusercontent.com/Noregret327/picture/master/202309121052899.png)

![image-20230912105514682](https://raw.githubusercontent.com/Noregret327/picture/master/202309121055864.png)

![image-20230912105533976](C:/Users/14224/AppData/Roaming/Typora/typora-user-images/image-20230912105533976.png)

![image-20230912105550177](https://raw.githubusercontent.com/Noregret327/picture/master/202309121055251.png)

### 4.6 模型的创建

```python
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

# 导入数据
data = pd.read_csv('dataset/HR.csv')		# 导入HR的数据
data.info()									# 查看数据信息
data.part.unique()							# 查看”part“类下有什么信息			输出：array(['sales', 'accounting', 'hr', ...)
data.salary.unique()						# 查看”salary“类下有什么信息			输出：array(['low', 'medium', 'high'], dtype=object)
data.groupby(['salary', 'part']).size()		# 将数据根据”salary“和”part”字段划分为不同的群体（group）进行分析

# 数据处理
data = data.join(pd.get_dummies(data.salary))	# 将“salary”数据离散化，再加入到数据后面
del data['salary']								# 删除“salary”数据
data = data.join(pd.get_dummies(data.part))		# 将“part”数据离散化，再加入到数据后面
del data['part']								# 删除“part”数据
data.head()										# 查看数据的表头，检查是否离散化并加入和删除“salary”和“part”数据
data.left.value_counts()						# 对“left”数据的进行排序和计算数量

# 获取数据
# 获取Y数据
Y_data = data.left.values.reshape(-1, 1)						# 获取“left”数据并重新排列存放在Y_data
Y_data.shape													# 查看“left”数据数量				输出：(14999, 1)
Y = torch.from_numpy(Y_data).type(torch.FloatTensor)			# 再把Y_data转换为张量“Tensor”
# 获取X数据
X_data = data[[c for c in data.columns if c !='left']].values	# 获取“left”数据的X_data值
X_data = X_data.astype(np.float32)								# 转换数据类型为float32
X = torch.from_numpy(X_data).type(torch.float32)				# 再把X_data转换为张量“Tensor”
# 查看数据大小
X.size()														# 查看X的张量大小					输出：torch.Size([14999, 20])
Y.shape															# 查看Y的张量大小					输出：torch.Size([14999, 1])



/**************************************************************************************************************/
# 创建模型
from torch import nn
# 自定义模型： 
# nn.Module：继承这个类 
# init：初始化所有的层 
# forward：定义模型的运算过程（前向传播的过程）
//////////////////////////////////////////////////////////////////
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.liner_1 = nn.Linear(20, 64)
        self.liner_2 = nn.Linear(64, 64)
        self.liner_3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    def forward(self, input):
        x = self.liner_1(input)
        x = self.relu(x)
        x = self.liner_2(x)
        x = self.relu(x)
        x = self.liner_3(x)
        x = self.sigmoid(x)
        return x	
///////////////////////////////////////////////////////////////////

model = Model()                   	# 初始化类
model								# 查看模型
/*************************************************************************************************************/
# 模型输出为：
# Model(
#     (liner_1): Linear(in_features=20, out_features=64, bias=True)
#     (liner_2): Linear(in_features=64, out_features=64, bias=True)
#     (liner_3): Linear(in_features=64, out_features=1, bias=True)
#     (relu): ReLU()
#     (sigmoid): Sigmoid()
# )
/*************************************************************************************************************/
```

### 4.7 改下模型

```python
import torch.nn.functional as F

///////////////////////////////////////////////////////////////////
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.liner_1 = nn.Linear(20, 64)
        self.liner_2 = nn.Linear(64, 64)
        self.liner_3 = nn.Linear(64, 1)
    def forward(self, input):
        x = F.relu(self.liner_1(input))
        x = F.relu(self.liner_2(x))
        x = F.sigmoid(self.liner_3(x))
        return x

///////////////////////////////////////////////////////////////////

model = Model()                   	# 初始化类
model								# 查看模型
/*************************************************************************************************************/
# 模型输出为：
# Model(
#      (liner_1): Linear(in_features=20, out_features=64, bias=True)
#      (liner_2): Linear(in_features=64, out_features=64, bias=True)
#      (liner_3): Linear(in_features=64, out_features=1, bias=True)
# )
/*************************************************************************************************************/


lr = 0.0001							# 定义学习率
def get_model():					# 重新定义模型和优化函数
    model = Model()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    return model, opt
model, optim = get_model()			# 获取模型和优化函数

```

### 4.8 定义损失函数

```python
# 定义损失函数
loss_fn = nn.BCELoss()

# 设置训练批次
batch = 64
no_of_batches = len(data)//batch
epochs = 100

///////////////////////////////////////////////////////////////////
# 开始训练
for epoch in range(epochs):
    for i in range(no_of_batches):
        start = i*batch
        end = start + batch
        x = X[start: end]
        y = Y[start: end]
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        optim.zero_grad()
        loss.backward()
        optim.step()
    with torch.no_grad():
        print('epoch:', epoch, 'loss:', loss_fn(model(X), Y).data.item())
///////////////////////////////////////////////////////////////////
        
        
# 查看模型的loss
loss_fn(model(X), Y)					# 输出：tensor(0.3745, grad_fn=<BinaryCrossEntropyBackward0>)

```

### 4.9 使用dataset类进行重构

```python
# 导入dataset类
from torch.utils.data import TensorDataset

# 使用dataset类进行重构
HRdataset = TensorDataset(X, Y)
len(HRdataset)							# 查看“HRdataset”长度
HRdataset[2: 5]							# 对“HRdataset”数据进行切片

# 获取模型和优化函数
model, optim = get_model()

///////////////////////////////////////////////////////////////////
# 重构后的训练
for epoch in range(epochs):
    for i in range(no_of_batches):
        x, y = HRdataset[i*batch: i*batch+batch]
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        optim.zero_grad()
        loss.backward()
        optim.step()
    with torch.no_grad():
        print('epoch:', epoch, 'loss:', loss_fn(model(X), Y).data.item())
///////////////////////////////////////////////////////////////////

```

### 4.10 使用dataloader类

```python
# 导入dataloader类
from torch.utils.data import DataLoader
 
# 使用dataloader类
HR_ds = TensorDataset(X, Y)
HR_dl = DataLoader(HR_ds, batch_size=batch, shuffle=True)
 
# 获取模型和优化函数
model, optim = get_model()

///////////////////////////////////////////////////////////////////
# 重构后的训练
for epoch in range(epochs):
   for x, y in HR_dl:
       y_pred = model(x)
       loss = loss_fn(y_pred, y)
       optim.zero_grad()
       loss.backward()
       optim.step()
   with torch.no_grad():
       print('epoch:', epoch, 'loss:', loss_fn(model(X), Y).data.item())
///////////////////////////////////////////////////////////////////
```

### 4.11 过拟合与欠拟合

- <font color=red size=4>**过拟合：对于训练数据过度拟合，对于未知数据预测很差**</font>
- <font color=red size=4>**欠拟合：对于训练数据拟合不够，对于未知数据预测很差**</font>

#### sklearn环境安装：

```cmd
conda install scikit-learn
```

### 4.12 划分数据集

```python
# 导入环境
from sklearn.model_selection import train_test_split

# 划分数据集（3：1）
train_x, test_x, train_y, test_y = train_test_split(X_data, Y_data)

# 查看数据大小
X_data.shape						 # 查看原始数据大小				输出：(14999, 20)
train_x.shape，test_x.shape			# 查看训练集和测试集数据大小		输出：(11249, 20)、(3750, 20)

# 转换为张量
train_x = torch.from_numpy(train_x).type(torch.float32)
train_y = torch.from_numpy(train_y).type(torch.float32)
test_x = torch.from_numpy(test_x).type(torch.float32)
test_y = torch.from_numpy(test_y).type(torch.float32)

# 加载dataset和dataloader
train_ds = TensorDataset(train_x, train_y)
train_dl = DataLoader(train_ds, batch_size=batch, shuffle=True)
test_ds = TensorDataset(test_x, test_y)
test_dl = DataLoader(test_ds, batch_size=batch)

```



## 5.多分类问题

### 5.1 Softmax分类

对数几率回归解决的是二分类的问题，对于<font color=red size=4>多个选项的问题</font>，我们可以使用softmax函数，它是对数几率回归在N个可能不同的值上的推广。

Softmax层的作用：

<font color=0x000fff size=4>神经网络的原始输出不是一个概率值，实质上只是输入的数值做了复杂的加权和与非线性处理之后的一个值而已</font>，那么如何将这个输出变为概率分布？

- Softmax要求每个样本必须属于某个类别，且所有可能的样本均被覆盖
- Softmax个样本分量之和为1——当只有两个类别时，与对数几率回归完全相同



### 5.2 pytorch交叉熵

在pytorch里，对于多分类问题，我们使用nn.CrossEntropyLoss()和nn.NLLLoss等来计算softmax交叉熵



### 5.3 完整代码

```python
import torch
# 导入环境
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# 导入划分数据集环境
from sklearn.model_selection import train_test_split
# 导入dataset和dataloader环境
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
# 导入nn和F环境
from torch import nn
import torch.nn.functional as F

# 导入数据
data = pd.read_csv('./dataset/iris.csv')
# 了解数据
data.head()                         				# 查看表头信息——确认输出（Species）
data.Species.unique()             					# 查看“Species”有多少类——确认属于二分类还是多分类问题
data['Species'] = pd.factorize(data.Species)[0]    	# “factorize”把“Species”中的所有类进行编号（0，1，2），再把编号后的覆盖掉“Species”列

# 提取数据——X，Y
X = data.iloc[:, 1: -1].values                     	# 取出所有行与第二列到倒数第二列（1：表示第二列，-1：表示倒数第二列）
Y = data.Species.values
X.shape												# 输出：(150, 4)
Y.shape												# 输出：(150,)


# 划分数据集
train_x, test_x, train_y, test_y = train_test_split(X, Y)
# 将数据转换为张量
train_x = torch.from_numpy(train_x).type(torch.float32)
train_y = torch.from_numpy(train_y).type(torch.int64)
test_x = torch.from_numpy(test_x).type(torch.float32)
test_y = torch.from_numpy(test_y).type(torch.LongTensor)       # LongTensor = int64
# 设置batch
len(data)
batch = 8
# 加载dataset和dataloader
train_ds = TensorDataset(train_x, train_y)
train_dl = DataLoader(train_ds, batch_size=batch, shuffle=True)
test_ds = TensorDataset(test_x, test_y)
test_dl = DataLoader(test_ds, batch_size=batch)

# 创建模型
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.liner_1 = nn.Linear(4, 32)
        self.liner_2 = nn.Linear(32, 32)
        self.liner_3 = nn.Linear(32, 3)
    def forward(self, input):
        x = F.relu(self.liner_1(input))
        x = F.relu(self.liner_2(x))
        x = self.liner_3(x)
        return x
# 导入模型
model = Model()
# 查看模型
model

# 创建损失函数
loss_fn = nn.CrossEntropyLoss()
# 获取一个批次张量
input_batch, label_batch = next(iter(train_dl))	
input_batch.shape, label_batch.shape				# 查看一个批次的张量
y_pred = model(input_batch)							# 查看一个批次输入的预测结果	
y_pred.shape										# 查看预测结果的张量				输出：torch.Size（[8, 3]）
y_pred												# 查看预测结果--共有8个预测，每个预测有三个，其中最大对应的为预测结果					
torch.argmax(y_pred, dim=1)							# 通过“argmax”来查看最大值所在位置			


# 定义acc——输入预测与真实进行对比，求出准确率
def accuracy(y_pred, y_true):
    y_pred = torch.argmax(y_pred, dim=1)
    acc = (y_pred == y_true).float().mean()
    return acc

# 创建空的列表
train_loss = []
train_acc = []
test_loss = []
test_acc = []

# 迭代次数
epochs = 20
# 优化函数
optim = torch.optim.Adam(model.parameters(), lr=0.0001)

# 训练
for epoch in range(epochs):
    for  x, y in train_dl:
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        optim.zero_grad()
        loss.backward()
        optim.step()
        
        # 查看训练结果 
    with torch.no_grad():
        epoch_accuracy = accuracy(model(train_x), train_y)
        epoch_loss = loss_fn(model(train_x), train_y).data
        epoch_test_accuracy = accuracy(model(test_x), test_y)
        epoch_test_loss = loss_fn(model(test_x), test_y).data
        print('epoch:', epoch, 'loss:', round(epoch_loss.item(), 3),
                               'accuracy:', round(epoch_accuracy.item(), 3),
                               'test_loss:', round(epoch_test_loss.item(), 3),
                               'test_accuracy:', round(epoch_test_accuracy.item(), 3)
             )
        # 将训练结果存在空的列表里
        train_loss.append(epoch_loss)
        train_acc.append(epoch_accuracy)
        test_loss.append(epoch_test_loss)
        test_acc.append(epoch_test_accuracy)


# 画出loss        
plt.plot(range(1, epochs+1), train_loss, label='train_loss')
plt.plot(range(1, epochs+1), test_loss, label='test_loss')
plt.legend()

# 画出acc
plt.plot(range(1, epochs+1), train_acc, label='train_acc')
plt.plot(range(1, epochs+1), test_acc, label='test_acc')
plt.legend()
        
```

### 5.4 通用训练函数

编写一个fit，输入模型、输入数据（train_dl,test_dl），对数据输入在模型上训练，并且返回loss和acc变化。

```python
def fit(epoch, model, trainloader, testloader):
    # 训练模型
    correct = 0													# 新建正确个数变量
    total = 0													# 新建总的个数变量
    running_loss = 0											# 新建loss个数变量
    for x, y in trainloader:
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        optim.zero_grad()
        loss.backward()
        optim.step()
        with torch.no_grad():
            y_pred = torch.argmax(y_pred, dim=1)				# 记录预测值
            correct += (y_pred == y).sum().item()				# 计算正确个数
            total += y.size(0)									# 计算总的个数（y.size(0)：y真实值，size(0):运行样本个数）
            running_loss += loss.item()							# 计算loss个数
    # 求解每个样本的loss和acc        
    epoch_loss = running_loss / len(trainloader.dataset)		# 求解每个样本的loss
    epoch_acc = correct / total									# 求解每个样本的正确率
    
    
    # 验证测试
    test_correct = 0
    test_total = 0
    test_running_loss = 0
    with torch.no_grad():
        for  x, y in testloader:
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            y_pred = torch.argmax(y_pred, dim=1)
            test_correct += (y_pred == y).sum().item()
            test_total += y.size(0)
            test_running_loss += loss.item()
    # 求解test的loss和acc         
    epoch_test_loss = test_running_loss / len(testloader.dataset)
    epoch_test_acc = test_correct / test_total
        
    print('epoch:', epoch, 
          'loss:', round(epoch_loss, 3),
          'accuracy:', round(epoch_acc, 3),
          'test_loss:', round(epoch_test_loss, 3),
          'test_accuracy:', round(epoch_test_acc, 3)
         )
        
    return epoch_loss, epoch_acc, epoch_test_loss, epoch_test_acc
```

```python
model = Model()												# 创建模型
optim = torch.optim.Adam(model.parameters(), lr=0.0001)		# 创建optim

# 创建训练次数
epochs = 20													
# 记录训练集和测试集的loss和acc
train_loss = []
train_acc = []
test_loss = []
test_acc = []

# 开始训练
for epoch in range(epochs):
    epoch_loss, epoch_acc, epoch_test_loss, epoch_test_acc = fit(epoch,
                                                                 model,
                                                                 train_dl,
                                                                 test_dl)
    train_loss.append(epoch_loss)
    train_acc.append(epoch_acc)
    test_loss.append(epoch_test_loss)
    test_acc.append(epoch_test_acc)
```

## 6.手写数字分类（全链接模型）

### 6.1 导入数据集

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from torchvision import datasets, transforms
# torchvision内置了常用数据集和最常见的模型
# datasets:包含常用的数据集 
# transforms：包含常用的转换方法
# transforms.ToTensor
#                    1.转化为一个 tensor
#                    2.转换到 0-1 之间
#                    3.会将channel放在第一个维度



# 下载训练集和测试集
train_ds = datasets.MNIST(
    'data/',
    train = True,
    transform = transformmation,
    download = True
)
test_ds = datasets.MNIST(
    'data/',
    train = False,
    transform = transformmation,
    download = True
)


# 导入训练集和测试集
train_dl = torch.utils.data.DataLoader(train_ds, batch_size=64, shuffle=True)
test_dl = torch.utils.data.DataLoader(test_ds, batch_size=256)
```

### 6.2 查看数据集

```python
# 取出一个批次的图片标签看看
imgs, labels = next(iter(train_dl))		# 取出一个批次的训练集的图片和标签            
imgs.shape								# 查看这个批次图片的张量				  输出：torch.Size([64, 1, 28, 28])
img = imgs[0]							# 取出第一张图片						
img.shape								# 查看这张图片的张量				       输出：torch.Size([1, 28, 28])
img = img.numpy()						# 将图片转换为numpy
img = np.squeeze(img)					# 再转换为【high， width】，不需要channel
img.shape								# 查看这张图片的张量				       输出：(28, 28)
plt.imshow(img)							# 画出该图片
labels[0]								# 查看这张图片的标签
labels[:10]								# 查看前十张图片的标签


# 定义一个查看图片函数（不需要自己手动一个个转换）
def imshow(img):
    npimg = img.numpy()
    npimg = np.squeeze(npimg)
    plt.imshow(npimg)

    
# 绘画十张手写数字
plt.figure(figsize=(10, 1))
for i, img in enumerate(imgs[:10]):
    plt.subplot(1, 10, i+1)               # 一行、十列、逐个取出
    imshow(img)
# 查看这十张手写数字标签
labels[:10]
```

### 6.3 创建模型

将图片展平一个长的tensor——全链接模型（一张图片像素为28 x 28，展平后为28 x 28的tensor输入） 隐藏层的超参数选择与定义，即隐藏层的输入和输出定义和选择

```python
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.liner_1 = nn.Linear(28*28, 120)            # 输入为图片展平像素大小，输出为下一个隐藏层的输入
        self.liner_2 = nn.Linear(120, 84)               # 该层为隐藏层，输入和输出涉及超参数的选择
        self.liner_3 = nn.Linear(84, 10)                # 输入为隐藏层的输出，输出为多分类问题的特征结果，数字标签只有10个数字，所以为10
    def forward(self, input):
        x = input.view(-1, 28*28)                       # 将图片展平
        x = F.relu(self.liner_1(x))
        x = F.relu(self.liner_2(x))
        x = self.liner_3(x)
        return x

# 加载模型
model = Model()
# 查看模型
model
//////////////////////////////////////////////////////////////////////////////////
# 该模型输出为：
# Model(
#   (liner_1): Linear(in_features=784, out_features=120, bias=True)
#   (liner_2): Linear(in_features=120, out_features=84, bias=True)
#   (liner_3): Linear(in_features=84, out_features=10, bias=True)
# )
/////////////////////////////////////////////////////////////////////////////////
```

### 6.4 导入训练代码

```python
def fit(epoch, model, trainloader, testloader):
    # 训练模型
    correct = 0													# 新建正确个数变量
    total = 0													# 新建总的个数变量
    running_loss = 0											# 新建loss个数变量
    for x, y in trainloader:
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        optim.zero_grad()
        loss.backward()
        optim.step()
        with torch.no_grad():
            y_pred = torch.argmax(y_pred, dim=1)				# 记录预测值
            correct += (y_pred == y).sum().item()				# 计算正确个数
            total += y.size(0)									# 计算总的个数（y.size(0)：y真实值，size(0):运行样本个数）
            running_loss += loss.item()							# 计算loss个数
    # 求解每个样本的loss和acc        
    epoch_loss = running_loss / len(trainloader.dataset)		# 求解每个样本的loss
    epoch_acc = correct / total									# 求解每个样本的正确率
    
    
    # 验证测试
    test_correct = 0
    test_total = 0
    test_running_loss = 0
    with torch.no_grad():
        for  x, y in testloader:
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            y_pred = torch.argmax(y_pred, dim=1)
            test_correct += (y_pred == y).sum().item()
            test_total += y.size(0)
            test_running_loss += loss.item()
    # 求解test的loss和acc         
    epoch_test_loss = test_running_loss / len(testloader.dataset)
    epoch_test_acc = test_correct / test_total
        
    print('epoch:', epoch, 
          'loss:', round(epoch_loss, 3),
          'accuracy:', round(epoch_acc, 3),
          'test_loss:', round(epoch_test_loss, 3),
          'test_accuracy:', round(epoch_test_acc, 3)
         )
        
    return epoch_loss, epoch_acc, epoch_test_loss, epoch_test_acc
```

### 6.5 定义训练次数、损失函数和优化函数

```python
# 定义训练次数
epochs = 20
# 定义损失函数
loss_fn = torch.nn.CrossEntropyLoss()
# 定义优化函数
optim = torch.optim.Adam(model.parameters(), lr=0.0001)

# 定义训练集和测试集的loss和acc变量
train_loss = []
train_acc = []
test_loss = []
test_acc = []

# 开始训练
for epoch in range(epochs):
    epoch_loss, epoch_acc, epoch_test_loss, epoch_test_acc = fit(epoch,
                                                                 model,
                                                                 train_dl,
                                                                 test_dl)
    # 存放数据
    train_loss.append(epoch_loss)
    train_acc.append(epoch_acc)
    test_loss.append(epoch_test_loss)
    test_acc.append(epoch_test_acc)
```

### 6.6 绘制loss和acc的变化图

```python
# 绘制loss变化
plt.plot(range(1, epochs+1), train_loss, label='train_loss')
plt.plot(range(1, epochs+1), test_loss, label='test_loss')
plt.legend()

# 绘制acc变化
plt.plot(range(1, epochs+1), train_acc, label='train_acc')
plt.plot(range(1, epochs+1), test_acc, label='test_acc')
plt.legend()
```



### 6.1 手写数字分类 全连接模型

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from torchvision import datasets, transforms

# 定义一个转换方法
transformmation = transforms.Compose([
    transforms.ToTensor()
])

# 导入数据集
train_ds = datasets.MNIST(
    'data/',
    train = True,
    transform = transformmation,
    download = True
)
test_ds = datasets.MNIST(
    'data/',
    train = False,
    transform = transformmation,
    download = True
)

# 加载（dataloader）训练集和测试集
train_dl = torch.utils.data.DataLoader(train_ds, batch_size=64, shuffle=True)
test_dl = torch.utils.data.DataLoader(test_ds, batch_size=256)

# 创建训练模型
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.liner_1 = nn.Linear(28*28, 120)            # 输入为图片展平像素大小，输出为下一个隐藏层的输入
        self.liner_2 = nn.Linear(120, 84)               # 该层为隐藏层，输入和输出涉及超参数的选择
        self.liner_3 = nn.Linear(84, 10)                # 输入为隐藏层的输出，输出为多分类问题的特征结果，数字标签只有10个数字，所以为10
    def forward(self, input):
        x = input.view(-1, 28*28)                       # 将图片展平
        x = F.relu(self.liner_1(x))
        x = F.relu(self.liner_2(x))
        x = self.liner_3(x)
        return x

# 加载模型
model = Model()

# 定义训练代码
def fit(epoch, model, trainloader, testloader):
    # 训练模型
    correct = 0													# 新建正确个数变量
    total = 0													# 新建总的个数变量
    running_loss = 0											# 新建loss个数变量
    for x, y in trainloader:
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        optim.zero_grad()
        loss.backward()
        optim.step()
        with torch.no_grad():
            y_pred = torch.argmax(y_pred, dim=1)				# 记录预测值
            correct += (y_pred == y).sum().item()				# 计算正确个数
            total += y.size(0)									# 计算总的个数（y.size(0)：y真实值，size(0):运行样本个数）
            running_loss += loss.item()							# 计算loss个数
    # 求解每个样本的loss和acc        
    epoch_loss = running_loss / len(trainloader.dataset)		# 求解每个样本的loss
    epoch_acc = correct / total									# 求解每个样本的正确率
    
    
    # 验证测试
    test_correct = 0
    test_total = 0
    test_running_loss = 0
    with torch.no_grad():
        for  x, y in testloader:
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            y_pred = torch.argmax(y_pred, dim=1)
            test_correct += (y_pred == y).sum().item()
            test_total += y.size(0)
            test_running_loss += loss.item()
    # 求解test的loss和acc         
    epoch_test_loss = test_running_loss / len(testloader.dataset)
    epoch_test_acc = test_correct / test_total
        
    print('epoch:', epoch, 
          'loss:', round(epoch_loss, 3),
          'accuracy:', round(epoch_acc, 3),
          'test_loss:', round(epoch_test_loss, 3),
          'test_accuracy:', round(epoch_test_acc, 3)
         )
        
    return epoch_loss, epoch_acc, epoch_test_loss, epoch_test_acc

# 定义训练次数
epochs = 20
# 定义损失函数
loss_fn = torch.nn.CrossEntropyLoss()
# 定义优化函数
optim = torch.optim.Adam(model.parameters(), lr=0.0001)

# 定义训练集和测试集的loss和acc变量
train_loss = []
train_acc = []
test_loss = []
test_acc = []

# 开始训练
for epoch in range(epochs):
    epoch_loss, epoch_acc, epoch_test_loss, epoch_test_acc = fit(epoch,
                                                                 model,
                                                                 train_dl,
                                                                 test_dl)
    # 存放数据
    train_loss.append(epoch_loss)
    train_acc.append(epoch_acc)
    test_loss.append(epoch_test_loss)
    test_acc.append(epoch_test_acc)

# 绘制loss变化
plt.plot(range(1, epochs+1), train_loss, label='train_loss')
plt.plot(range(1, epochs+1), test_loss, label='test_loss')
plt.legend()
plt.show()

# 绘制acc变化
plt.plot(range(1, epochs+1), train_acc, label='train_acc')
plt.plot(range(1, epochs+1), test_acc, label='test_acc')
plt.legend()
plt.show()
```



## 7.基础部分总结

### 7.1 梯度下降法

#### 7.1.1 梯度下降法

- 梯度下降法是一种致力于<font color=0x0ff size=4>**找到函数极值点的算法。**</font>
- 所谓<font color=red size=5>**“学习”**</font>便是<font color=0x0ff size=4>**改进模型参数，以便通过大量训练步骤将损失最小化。**</font>

- 将<font color=red size=5>**梯度下降法**</font>应用于<font color=0x0ff size=4>**寻找损失函数的极值点**</font>便构成了依据输入数据的模型学习。

#### 7.1.2 梯度

- <font color=red size=5>梯度的输出</font>是<font color=0x0ff size=4>一个由若干偏导数构成的向量</font>，它的每个分量对应于函数对输入向量的相应分量的偏导。
- <font color=red size=5>梯度的输出向量</font>表明了在<font color=0x0ff size=4>每个位置损失函数增长最快的方向</font>，可将它视为表示了在函数的每个位置向哪个方向移动函数值可以增长。
- <font color=red size=5>曲线</font>对应于损失函数；<font color=red size=5>点</font>表示<font color=0x0ff size=4>权值的当前值</font>，即现在所在的位置；<font color=red size=5>梯度</font>用<font color=0x0ff size=4>箭头表示，表明为了增加损失，需要向右移动</font>；箭头的长度概念化地表示了如果在对应的方向移动，函数值能够增长多少。如果向着梯度的反方向移动，则损失函数的值会相应减少。沿着损失函数减少的方向移动，并再次计算梯度值，并重复上述过程，直至梯度的模为0，将到达损失函数的极小值点。这正是我们的目标。
- <font color=red size=5>**梯度**</font>就是表明<font color=0x0ff size=4>**损失函数相对参数的变化率**</font>。



### 7.2 学习速率

#### 7.2.1 学习速率

- <font color=0x0ff size=4>对梯度进行缩放的参数</font>被称为<font color=red size=5>学习速率（learning rate）</font>
- 如果学习速率太小，则找到损失函数极小值点时可能需要许多轮迭代；如果太大，则算法可能会“跳过”极小值点并且因周期性的“跳跃”而永远无法找到极小值点。
- 在具体实践中，可通过查看损失函数值随时间的变化曲线，来判断学习速率的选取是合适的。
- 合适的学习速率，损失函数随时间下降，直到一个底部；不合适的学习速率，损失函数可能会发生震荡。
- 在调整学习速率时，即需要使其足够小，保证不至于发生超调，也要保证它足够大，以使损失函数能够尽快下降，从而可通过较少次数的迭代更快地完成学习

#### 7.2.2 局部极值点

- 可通过将权值随机初始化来改善局部极值的问题。
- 权值的初值使用随机值，可以增加从靠近全局最优点附件开始下降的机会。



### 7.3 反向传播算法

- 反向传播算法是一种高效计算数据流图中梯度的技术
- **每一层的导数**都是**最后一层的导数与前一层输出之积**，这正是<font color=red size=5>链式法则</font>的奇妙之处，误差反向传播算法利用的正是这一特点。
- 前馈时，从输入开始，**逐一计算每个隐含层的输出，直到输出层**。然后开始**计算导数**，并<font color=0x0ff size=4>从输出层经各隐含层逐一反向传播。</font>为了减少计算量，还需对所有已完成计算的元素进行复用。这便是反向传播算法的由来。



### 7.4 优化函数

优化器（optimizer）是根据反向传播计算出的梯度，优化模型参数的内置方法。

#### SGD：随机梯度下降优化器

- 随机梯度下降优化器SGD和min-batch是同一个意思，抽取m个小批量（独立同分布）样本，通过计算他们平均梯度均值。

- lr：float >= 0.学习率
- momentum：float >= 0.参数，用于加速SGD在相关方向上前进，并抑制震荡。

#### RMSprop：

- 经验上，它被证明有效且实用的深度学习网络优化算法
- 它增加了一个衰减系数来控制历史信息的获取多少。
- 它会对学习率进行衰减。
- 特别适合优化序列的问题

#### Adam：

- 它可以看做是修正后的Momentum+RMSprop算法
- 它通常被认为对超参数的选择相当鲁棒性
- 学习率建议：0.001
- 它是一种可以替代传统随机梯度下降过程的一阶优化算法，它能基于训练数据迭代地更新神经网络权重。
- 它通过计算梯度的一阶矩估计和二阶矩估计而为不同的参数设计独立的自适应学习率。



### 7.5 基础总结

#### 7.5.1 创建模型的三种方法

1. 单层创建	nn.Linear
2. torch.nn.Sequential
3. 自定义类（继承自nn.Module）



#### 7.5.2 数据输入方式

1. 从ndarray创建Tensor直接切片输入
2. 使用torch.utils.data.TensorDataset创建dataset
3. 使用torchvision.dataset的内置数据集
4. 使用torch.utils.data.Dataloader封装

 

#### 7.5.3 模型训练的步骤

1. 预处理数据
2. 创建模型和优化方法
3. 模型调用
4. 计算损失
5. 梯度归零
6. 计算梯度
7. 优化模型
8. 打印指标



### 7.6 不同问题使用的损失函数和输出设置

#### 一、回归问题

预测连续的值叫做回归问题

- 损失函数：均方误差 mse
- 输出层激活方式：无

#### 二、二分类问题

回答是和否的问题

- 损失函数：BCEloss
- 输出激活层激活方式：sigmoid

或

- 损失函数：CrossEntropyLoss
- 输出层激活方式：无

#### 三、多分类问题

多个分类的问题，输出与分类个数相同长度的张量

- 损失函数：CrossEntropyLoss
- 输出层激活方式：无

或

- 损失函数：nn.NLLLoss——要求labels必须是独热编码方式
- 输出层激活方式：torch.log_softmax



## 8.计算机视觉

### 8.1 CNN

#### 8.1.1 卷积神经网络

主要应用于计算机视觉相关任务，但它能处理的任务并不局限于图像，其实语音识别也是可以使用卷积神经网络。

- 当计算机看到一张图像（输入一张图像）时，它看到的是一大堆像素值。
- 当我们人类对图像进行分类时，这些数字毫无用处，可它们却是计算机可获取的唯一输入。
- 现在的问题是：当你提供给计算机这一数组后，它将输出描述该图像属于某一特定分类的概率的数字（比如：80%是猫、15%是狗、5%是鸟）
- CNN工作方式：计算机通过寻找诸如边缘和曲线之类的低级特点来分类图片，继而通过一系列卷积层级构建出更为抽象的概念。

#### 8.1.2 CNN工作：

1. <font color=red size=4>卷积层</font>：用于检测输入数据中的特征，例如图像中的边缘、纹理和形状等。——<font color=red size=4>conv2d</font>
2. <font color=red size=4>非线性层（激活层）</font>：将卷积层的输出经过激活函数，将负数置零，从而引入非线性，使网络能够学习更复杂的特征。——<font color=red size=4>relu/sigmoid/tanh</font>
3. <font color=red size=4>池化层</font>：用于减小特征图的空间维度，降低计算复杂性，并且使模型对位置的变化更加稳定。——<font color=red size=4>pooling2d</font>
4. <font color=red size=4>全链接层</font>：将前面层的所有特征连接到每个神经元，用于进行最终的分类或回归操作。——<font color=red size=4>w*x+b</font>

（如果没有这些层，模型很难与复杂模型匹配，因为网络将有过多的信息填充，也就是其他那些层作用就是突出重要信息，降低噪声）

#### 8.1.3 什么是卷积？

- <font color=blue size=4>卷积</font>是指将卷积核应用到某个张量的所有点上，通过将卷积核在输入的张量上滑动而生成经过滤波处理的张量。
- 总结：<font color=blue size=4>**卷积就是完成对图像特征的提取或者说信息匹配**</font>

#### 8.1.4 卷积层

三个参数：

- ksize：卷积核大小
- strides：卷积核移动的跨度
- padding：边缘填充

#### 8.1.5 非线性变化层

- relu
- sigmoid
- tanh

#### 8.1.6 池化层

- layers.MaxPooling2D：最大池化
- nn.AvgPool2d(2,stride=2)：平均池化
- nn.AdaptiveAvgPool2d(output_size=(100,100))：自适应平均池化

#### 8.1.7 全连接层

将最后的输出与全部特征连接，我们要使用全部的特征，为最后的分类的做出决策。

最后配合softmax进行分类。



### 8.2 手写数字分类 卷积模型

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

import torchvision
from torchvision import datasets, transforms

transformmation = transforms.Compose([
    transforms.ToTensor()
])

# 导入数据集
train_ds = datasets.MNIST(
    'data/',
    train = True,
    transform = transformmation,
    download = False
)
test_ds = datasets.MNIST(
    'data/',
    train = False,
    transform = transformmation,
    download = False
)
train_dl = torch.utils.data.DataLoader(train_ds, batch_size=64, shuffle=True)
test_dl = torch.utils.data.DataLoader(test_ds, batch_size=256)

# 创建卷积模型
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d((2, 2))
        self.conv2 = nn.Conv2d(6, 16, 5)  
        self.liner_1 = nn.Linear(16*4*4, 256)            
        self.liner_2 = nn.Linear(256, 10)               
    def forward(self, input):
        x = F.relu(self.conv1(input))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x) 
#         print(x.self())              # torch.size([64, 16, 4, 4])
        x = x.view(x.size(0), -1)                       # 将图片展平
        x = F.relu(self.liner_1(x))
        x = self.liner_2(x)
        return x
        
# 通用训练模型定义
def fit(epoch, model, trainloader, testloader):
    # 训练模型
    correct = 0													# 新建正确个数变量
    total = 0													# 新建总的个数变量
    running_loss = 0											# 新建loss个数变量
    for x, y in trainloader:
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        optim.zero_grad()
        loss.backward()
        optim.step()
        with torch.no_grad():
            y_pred = torch.argmax(y_pred, dim=1)				# 记录预测值
            correct += (y_pred == y).sum().item()				# 计算正确个数
            total += y.size(0)									# 计算总的个数（y.size(0)：y真实值，size(0):运行样本个数）
            running_loss += loss.item()							# 计算loss个数
    # 求解每个样本的loss和acc        
    epoch_loss = running_loss / len(trainloader.dataset)		# 求解每个样本的loss
    epoch_acc = correct / total									# 求解每个样本的正确率
    
    
    # 验证测试
    test_correct = 0
    test_total = 0
    test_running_loss = 0
    with torch.no_grad():
        for  x, y in testloader:
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            y_pred = torch.argmax(y_pred, dim=1)
            test_correct += (y_pred == y).sum().item()
            test_total += y.size(0)
            test_running_loss += loss.item()
    # 求解test的loss和acc         
    epoch_test_loss = test_running_loss / len(testloader.dataset)
    epoch_test_acc = test_correct / test_total
        
    print('epoch:', epoch, 
          'loss:', round(epoch_loss, 3),
          'accuracy:', round(epoch_acc, 3),
          'test_loss:', round(epoch_test_loss, 3),
          'test_accuracy:', round(epoch_test_acc, 3)
         )
        
    return epoch_loss, epoch_acc, epoch_test_loss, epoch_test_acc

# 训练输入定义
epochs = 20
model = Model()
loss_fn = torch.nn.CrossEntropyLoss()           # 定义损失函数
optim = torch.optim.Adam(model.parameters(), lr=0.0001)        # 定义优化函数

# 记录训练集和测试集的loss和acc
train_loss = []
train_acc = []
test_loss = []
test_acc = []

# 开始训练
for epoch in range(epochs):
    epoch_loss, epoch_acc, epoch_test_loss, epoch_test_acc = fit(epoch,
                                                                 model,
                                                                 train_dl,
                                                                 test_dl)
    train_loss.append(epoch_loss)
    train_acc.append(epoch_acc)
    test_loss.append(epoch_test_loss)
    test_acc.append(epoch_test_acc)
```

### 8.3 四种天气图片分类

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import torchvision
import os
import shutil

from torchvision import datasets, transforms
torchvision.datasets.ImageFolder        			# 从分类的文件夹中创建数据集

# 创建数据集，并分类
base_dir = r'./dataset/4weather'					# 创建数据集基本目录
train_dir = os.path.join(base_dir, 'train')			# 在数据集基本目录创建训练集目录
test_dir = os.path.join(base_dir, 'test')			# 在数据集基本目录创建测试集目录
species = ['cloudy', 'rain', 'shine', 'sunrise']	# 定义类型集合
image_dir = r'./data/dataset2/'						# 数据集图片原始位置


if not os.path.isdir(base_dir):						# 判断是否创建了数据集基本目录，如果没有就执行下面运算
    os.make(base_dir)								# 创建数据集基本目录
    os.mkdir(train_dir)								# 创建训练集目录
    os.mkdir(test_dir)								# 创建测试集目录
    

# 创建新的四种天气文件夹——从train和test文件夹中分别创建
# for train_or_test in ['train', 'test']:
#     for spec in species:
#         os.mkdir(os.path.join(base_dir, train_or_test, spec))


# 从数据集中提取图片——并分类到各个train和test文件夹中（1/5的测试集划分）
# for i, img in enumerate(os.listdir(image_dir)):
#     for spec in species:
#         if spec in img:
#             s = os.path.join(image_dir, img)
#             if i%5 == 0:
#                 d = os.path.join(base_dir, 'test', spec, img)
#             else:
#                 d = os.path.join(base_dir, 'train', spec, img)
#             shutil.copy(s, d )

# 查看文件夹中各个文件夹里面的文件数
for train_or_test in ['train', 'test']:
    for spec in species:
        print(train_or_test, spec, len(os.listdir(os.path.join(base_dir, train_or_test, spec))))
//////////////////////////
# train cloudy 240
# train rain 172
# train shine 202
# train sunrise 286
# test cloudy 60
# test rain 43
# test shine 51
# test sunrise 71
/////////////////////////

# 创建数据集对象——tranform、train_ds、test_ds
transform = transforms.Compose([
    transforms.Resize((96, 96)),					# 将输入的图像调整为大小为 96x96	
    transforms.ToTensor(),							# 将调整尺寸后的图像转换为 PyTorch 的张量（tensor）
    transforms.Normalize(mean=[0.5, 0.5, 0.5],		# 对图像进行标准化处理，以便将像素值范围缩放到 -1 到 1 之间。
                         std=[0.5, 0.5, 0.5])		# 它从每个通道（红色、绿色和蓝色）中减去均值（0.5）并将结果除以标准差（0.5）
])													# 助于提高模型的训练稳定性和性能，因为它有助于确保输入数据的分布类似于标准正态分布。
train_ds = torchvision.datasets.ImageFolder(		# 使用 PyTorch 的 ImageFolder 数据集类来创建一个数据集对象
    train_dir, 										# 包含训练图像数据的文件夹路径
    transform=transform								# 使用了之前定义的 transform 对象
)
test_ds = torchvision.datasets.ImageFolder(
    test_dir, 										# 包含测试图像数据的文件夹路径
    transform=transform
)

# 查看数据的类型和索引标号
train_ds.classes									# 输出：['cloudy', 'rain', 'shine', 'sunrise']
train_ds.class_to_idx								# 输出：{'cloudy': 0, 'rain': 1, 'shine': 2, 'sunrise': 3}
len(train_ds), len(test_ds)							# 输出：(900, 225)


# 创建加载器——train_dl、test_dl
BATCHSIZE = 16										# 批量大小被设置为 16，意味着每次加载 16 个样本进行训练或测试
train_dl = torch.utils.data.DataLoader(				# 创建了一个训练数据加载器 train_dl。
    train_ds,
    batch_size=BATCHSIZE,
    shuffle=True									# 这个参数指示数据加载器在每个训练周期（epoch）开始时是否对训练数据进行洗牌（随机打乱顺序）
)													# 随机化数据的顺序可以帮助模型更好地学习。							
test_dl = torch.utils.data.DataLoader(
    test_ds,
    batch_size=BATCHSIZE
)

# 取出一个批次的图像和标签看看数据集
imgs, labels = next(iter(train_dl))					# 提取一个批次的训练数据
imgs.shape											# 输出：torch.Size([16, 3, 96, 96])
imgs[0].shape										# 输出：torch.Size([3, 96, 96])
im = imgs[0].permute(1, 2, 0)            			# 交换channel和宽、高的位置
im.shape											# 输出：torch.Size([96, 96, 3])

# 类和标签的转换
train_ds.class_to_idx								# {'cloudy': 0, 'rain': 1, 'shine': 2, 'sunrise': 3}
id_to_class = dict((v, k) for k, v in train_ds.class_to_idx.items())	# 利用字典方式交换标签和类名
id_to_class											# 输出：{0: 'cloudy', 1: 'rain', 2: 'shine', 3: 'sunrise'}

# 自己加载和查看数据集的方法
plt.figure(figsize=(12, 8))							
for i, (img, label) in enumerate(zip(imgs[:6], labels[:6])):
    img = (img.permute(1, 2, 0).numpy() + 1)/2
    plt.subplot(2, 3, i+1)
    plt.title(id_to_class.get(label.item()))
    plt.imshow(img)
    
# 创建模型
class Net(nn.Module):                       		# 创建模型
    def __init__(self):                      		# 对所有层进行初始化定义
        super(Net, self).__init__()          		# 继承所有父类属性
        self.conv1 = nn.Conv2d(3, 16, 3)     		# 定义第一个卷积层
        self.conv2 = nn.Conv2d(16, 32, 3)    		# 定义第二个卷积层
        self.conv3 = nn.Conv2d(32, 64, 3)    		# 定义第三个卷积层
        self.pool = nn.MaxPool2d(2, 2)       		# 定义池化层
        self.fc1 = nn.Linear(64*10*10, 1024) 		# 定义全连接层（卷积层之后的图像大小需要计算）
        self.fc2 = nn.Linear(1024, 4)        		# 定义全连接层（最后输出为：4（4个分类））
    def forward(self, x):                   		# 定义调用前向传播过程
        x = F.relu(self.conv1(x))            		# 第一个激活卷积层
        x = self.pool(x)                     		# 池化
        x = F.relu(self.conv2(x))           		# 第二个激活卷积层
        x = self.pool(x)                     		# 池化
        x = F.relu(self.conv3(x))            		# 第三个激活卷积层
        x = self.pool(x)                     		# 池化
#        print(x.size())                      		# 打印层数
#        x = x.view(-1, x.size(1)*x.size(2)*x.size(3)) 
        x = x.view(-1, 64*10*10)
        x = F.relu(self.fc1(x))              		# 第四个激活全链接层
        x = self.fc2(x)                      		# 最后一个全链接层
        return x

model = Net()
model(imgs)               							# 第一全链接层Linear的输入参数为64 * 10 * 10
preds = model(imgs)
imgs.shape											# 输出：torch.Size([16, 3, 96, 96])
preds.shape      								# 查看预测结果输出，其结果应该有四个（代表四类），最大的为预测结果 输出：torch.Size([16, 4])
torch.argmax(preds, 1)      						# 算出预测结果

//////////////////////////////////////////
# 放在GPU训练方法
if torch.cuda.is_available():
    model.to('cuda')
//////////////////////////////////////////

# 定义损失函数和优化函数
loss_fn = nn.CrossEntropyLoss()
optim = torch.optim.Adam(model.parameters(), lr=0.001)
def fit(epoch, model, trainloader, testloader):
    # 训练模型
    correct = 0													# 新建正确个数变量
    total = 0													# 新建总的个数变量
    running_loss = 0											# 新建loss个数变量
    for x, y in trainloader:
        if torch.cuda.is_available():
            x, y = x.to('cuda'), y.to('cuda')
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        optim.zero_grad()
        loss.backward()
        optim.step()
        with torch.no_grad():
            y_pred = torch.argmax(y_pred, dim=1)				# 记录预测值
            correct += (y_pred == y).sum().item()				# 计算正确个数
            total += y.size(0)									# 计算总的个数（y.size(0)：y真实值，size(0):运行样本个数）
            running_loss += loss.item()							# 计算loss个数
    # 求解每个样本的loss和acc        
    epoch_loss = running_loss / len(trainloader.dataset)		# 求解每个样本的loss
    epoch_acc = correct / total									# 求解每个样本的正确率
    
    
    # 验证测试
    test_correct = 0
    test_total = 0
    test_running_loss = 0
    with torch.no_grad():
        for  x, y in testloader:
            if torch.cuda.is_available():
                x, y = x.to('cuda'), y.to('cuda')
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            y_pred = torch.argmax(y_pred, dim=1)
            test_correct += (y_pred == y).sum().item()
            test_total += y.size(0)
            test_running_loss += loss.item()
    # 求解test的loss和acc         
    epoch_test_loss = test_running_loss / len(testloader.dataset)
    epoch_test_acc = test_correct / test_total
        
    print('epoch:', epoch, 
          'loss:', round(epoch_loss, 3),
          'accuracy:', round(epoch_acc, 3),
          'test_loss:', round(epoch_test_loss, 3),
          'test_accuracy:', round(epoch_test_acc, 3)
         )
        
    return epoch_loss, epoch_acc, epoch_test_loss, epoch_test_acc

# 训练次数
epochs = 30
# 记录训练集和测试集的loss和acc
train_loss = []
train_acc = []
test_loss = []
test_acc = []
# 开始训练
for epoch in range(epochs):
    epoch_loss, epoch_acc, epoch_test_loss, epoch_test_acc = fit(epoch,
                                                                 model,
                                                                 train_dl,
                                                                 test_dl)
    train_loss.append(epoch_loss)
    train_acc.append(epoch_acc)
    test_loss.append(epoch_test_loss)
    test_acc.append(epoch_test_acc)
    
plt.plot(range(1, epochs+1), train_acc, label='train_acc')
plt.plot(range(1,epochs+1), test_acc, label='test_acc')
plt.legend()

plt.plot(range(1, epochs+1), train_loss, label='train_loss')
plt.plot(range(1,epochs+1), test_loss, label='test_loss')
plt.legend()
    
    
# 过拟合
# 在train数据得到的数据非常好，但是在test数据中得不到很好的数据，test和train差别很大，这是典型过拟合现象
```

### 8.4 Dropout抑制过拟合

为什么Dropout可以解决过拟合？

1. 取平均的作用（可以将所有神经元权值取平均）
2. 减少神经元之间复杂的共适应关系：因为dropout程序导致两个神经元不一定每次都在一个dropout网络中出现。权值的更新不再依赖于有固定关系的隐含节点的共同作用，阻止了某些特征仅仅在某些特征仅仅在其他特定特征下才有效果的情况。
3. 类似于性别在生物进化中的角色：性别的出现可以繁衍出适应新环境的变种，有效的阻止过拟合，即避免环境改变时物种可能面临的灭绝

### 8.5 四种天气图片分类 Dropout抑制过拟合

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import torchvision
import os
import shutil

from torchvision import datasets, transforms
torchvision.datasets.ImageFolder        			# 从分类的文件夹中创建数据集

# 创建目录
base_dir = r'./dataset/4weather'					# 创建数据集基本目录
train_dir = os.path.join(base_dir, 'train')			# 在数据集基本目录创建训练集目录
test_dir = os.path.join(base_dir, 'test')			# 在数据集基本目录创建测试集目录

# 创建数据集对象——tranform、train_ds、test_ds
transform = transforms.Compose([
    transforms.Resize((96, 96)),					# 将输入的图像调整为大小为 96x96	
    transforms.ToTensor(),							# 将调整尺寸后的图像转换为 PyTorch 的张量（tensor）
    transforms.Normalize(mean=[0.5, 0.5, 0.5],		# 对图像进行标准化处理，以便将像素值范围缩放到 -1 到 1 之间。
                         std=[0.5, 0.5, 0.5])		# 它从每个通道（红色、绿色和蓝色）中减去均值（0.5）并将结果除以标准差（0.5）
])													# 助于提高模型的训练稳定性和性能，因为它有助于确保输入数据的分布类似于标准正态分布。
train_ds = torchvision.datasets.ImageFolder(		# 使用 PyTorch 的 ImageFolder 数据集类来创建一个数据集对象
    train_dir, 										# 包含训练图像数据的文件夹路径
    transform=transform								# 使用了之前定义的 transform 对象
)
test_ds = torchvision.datasets.ImageFolder(
    test_dir, 										# 包含测试图像数据的文件夹路径
    transform=transform
)

# 创建加载器——train_dl、test_dl
BATCHSIZE = 16										# 批量大小被设置为 16，意味着每次加载 16 个样本进行训练或测试
train_dl = torch.utils.data.DataLoader(				# 创建了一个训练数据加载器 train_dl。
    train_ds,
    batch_size=BATCHSIZE,
    shuffle=True									# 这个参数指示数据加载器在每个训练周期（epoch）开始时是否对训练数据进行洗牌（随机打乱顺序）
)													# 随机化数据的顺序可以帮助模型更好地学习。							
test_dl = torch.utils.data.DataLoader(
    test_ds,
    batch_size=BATCHSIZE
)

################################################################################
# 创建一个Dropout层——有一定抑制过拟合效果
class Net(nn.Module):                       		# 创建模型
    def __init__(self):                      		# 对所有层进行初始化定义
        super(Net, self).__init__()          		# 继承所有父类属性
        self.conv1 = nn.Conv2d(3, 16, 3)     		# 定义第一个卷积层
        self.conv2 = nn.Conv2d(16, 32, 3)    		# 定义第二个卷积层
        self.conv3 = nn.Conv2d(32, 64, 3)    		# 定义第三个卷积层
        self.drop = nn.Dropout(0.5)                 # 随机丢弃0.5的数据
        self.pool = nn.MaxPool2d(2, 2)       		# 定义池化层
        self.fc1 = nn.Linear(64*10*10, 1024) 		# 定义全连接层（卷积层之后的图像大小需要计算）
        self.fc2 = nn.Linear(1024, 256)        		# 定义全连接层
        self.fc3 = nn.Linear(256, 4)                # 定义全连接层（最后输出为：4（4个分类））
    def forward(self, x):                   		# 定义调用前向传播过程
        x = self.pool(F.relu(self.conv1(x)))        # 第一个池化、激活、卷积层
        x = self.pool(F.relu(self.conv2(x)))  		# 第二个激活卷积层
        x = self.pool(F.relu(self.conv3(x)))        # 第三个激活卷积层
        x = x.view(-1, 64*10*10)
        x = F.relu(self.fc1(x))              		# 第四个激活全链接层
        x = self.drop(x)                            # 在全链接层增加一个Dropout层
        x = F.relu(self.fc2(x))              		# 第五个激活全链接层
        x = self.drop(x)                            # 在全链接层增加一个Dropout层
        x = self.fc3(x)                      		# 最后一个全链接层
        return x
################################################################################
# 再添加Dropout2层
class Net(nn.Module):                       		# 创建模型
    def __init__(self):                      		# 对所有层进行初始化定义
        super(Net, self).__init__()          		# 继承所有父类属性
        self.conv1 = nn.Conv2d(3, 16, 3)     		# 定义第一个卷积层
        self.conv2 = nn.Conv2d(16, 32, 3)    		# 定义第二个卷积层
        self.conv3 = nn.Conv2d(32, 64, 3)    		# 定义第三个卷积层
        self.drop = nn.Dropout(0.5)                 # 随机丢弃0.5的数据
        self.drop2 = nn.Dropout2d(0.5)              # 增加 Dropout2 层
        self.pool = nn.MaxPool2d(2, 2)       		# 定义池化层
        self.fc1 = nn.Linear(64*10*10, 1024) 		# 定义全连接层（卷积层之后的图像大小需要计算）
        self.fc2 = nn.Linear(1024, 256)        		# 定义全连接层
        self.fc3 = nn.Linear(256, 4)                # 定义全连接层（最后输出为：4（4个分类））
    def forward(self, x):                   		# 定义调用前向传播过程
        x = self.pool(F.relu(self.conv1(x)))        # 第一个池化、激活、卷积层
        x = self.pool(F.relu(self.conv2(x)))  		# 第二个激活卷积层
        x = self.pool(F.relu(self.conv3(x)))        # 第三个激活卷积层
        x = self.drop2(x)                           # 在卷积层后增加 dropout2 层
        x = x.view(-1, 64*10*10)
        x = F.relu(self.fc1(x))              		# 第四个激活全链接层
        x = self.drop(x)                            # 在全链接层增加一个Dropout层
        x = F.relu(self.fc2(x))              		# 第五个激活全链接层
        x = self.drop(x)                            # 在全链接层增加一个Dropout层
        x = self.fc3(x)                      		# 最后一个全链接层
        return x
################################################################################

model = Net()
if torch.cuda.is_available():
    model.to('cuda')
    
////////////////////////////////////////////////////////////////////////////////
# 添加两个训练模式
# model.train() 训练模式
# model.eval() 预测模式 # 主要影响 Dropout层 和 BN层

def fit(epoch, model, trainloader, testloader):
    # 训练模型
    correct = 0													# 新建正确个数变量
    total = 0													# 新建总的个数变量
    running_loss = 0											# 新建loss个数变量
    model.train()                                               # 将模型设为训练模式，Dropout发挥作用
    for x, y in trainloader:
        if torch.cuda.is_available():
            x, y = x.to('cuda'), y.to('cuda')
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        optim.zero_grad()
        loss.backward()
        optim.step()
        with torch.no_grad():
            y_pred = torch.argmax(y_pred, dim=1)				# 记录预测值
            correct += (y_pred == y).sum().item()				# 计算正确个数
            total += y.size(0)									# 计算总的个数（y.size(0)：y真实值，size(0):运行样本个数）
            running_loss += loss.item()							# 计算loss个数
    # 求解每个样本的loss和acc        
    epoch_loss = running_loss / len(trainloader.dataset)		# 求解每个样本的loss
    epoch_acc = correct / total									# 求解每个样本的正确率
    
    
    # 验证测试
    test_correct = 0
    test_total = 0
    test_running_loss = 0
    model.eval()                                                 # 将模型设为预测模式，Dropout不发挥作用
    with torch.no_grad():
        for  x, y in testloader:
            if torch.cuda.is_available():
                x, y = x.to('cuda'), y.to('cuda')
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            y_pred = torch.argmax(y_pred, dim=1)
            test_correct += (y_pred == y).sum().item()
            test_total += y.size(0)
            test_running_loss += loss.item()
    # 求解test的loss和acc         
    epoch_test_loss = test_running_loss / len(testloader.dataset)
    epoch_test_acc = test_correct / test_total
        
    print('epoch:', epoch, 
          'loss:', round(epoch_loss, 3),
          'accuracy:', round(epoch_acc, 3),
          'test_loss:', round(epoch_test_loss, 3),
          'test_accuracy:', round(epoch_test_acc, 3)
         )
        
    return epoch_loss, epoch_acc, epoch_test_loss, epoch_test_acc
////////////////////////////////////////////////////////////////////////////////

# 记录训练集和测试集的loss和acc
train_loss = []
train_acc = []
test_loss = []
test_acc = []
# 开始训练
for epoch in range(epochs):
    epoch_loss, epoch_acc, epoch_test_loss, epoch_test_acc = fit(epoch,
                                                                 model,
                                                                 train_dl,
                                                                 test_dl)
    train_loss.append(epoch_loss)
    train_acc.append(epoch_acc)
    test_loss.append(epoch_test_loss)
    test_acc.append(epoch_test_acc)

plt.plot(range(1, epochs+1), train_acc, label='train_acc')
plt.plot(range(1,epochs+1), test_acc, label='test_acc')
plt.legend()

plt.plot(range(1, epochs+1), train_loss, label='train_loss')
plt.plot(range(1,epochs+1), test_loss, label='test_loss')
plt.legend()

```

只有Dropout层时的抑制效果为：

![image-20230918151321633](https://raw.githubusercontent.com/Noregret327/picture/master/202309181513680.png)

增加Dropout2层时的抑制效果为：

![image-20230918145845677](https://raw.githubusercontent.com/Noregret327/picture/master/202309181458746.png)

### 8.6 标准化

#### 1）什么是标准化：

- 传统机器学习中标准化也叫做归一化，一般是<font color=blue size=4>**将数据映射到指定的范围，用于去除不同维度数据的量纲以及量纲单位。**</font>
- 数据标准化让机器学习模型看到的<font color=red size=4>**不同样本彼此之间更加相似**</font>，这有助于模型的学习与对新数据的泛化。

#### 2）常见的数据标准化形式：

- 标准化
- 归一化

#### 3）归一化：

- 归一化：Normalization
- 将数据减去其平均值使其中心为0，然后将数据除以其他标准差使其标准差为1——<font color=red size=4>**（将数据分布在0~1之间——减均值、除方差）**</font>

#### 4）批标准化：

- 批标准化：Batch Normalization
- 批标准化和普通的数据标准化类似，是将分散的数据统一的一种做法，也是优化神经网络的一种方法
- <font color=blue size=4>**不仅在将数据输入模型之前对数据做标准化，在网络的每一次变换之后都应该考虑数据标准化**</font>
- 即使在训练过程中均值和方差随时间发生变化，它也可以适应性地将数据标准化
- <font color=red size=4>**批标准化解决的问题是梯度消失与梯度爆炸。**</font>

#### 5）批标准化的好处：

- 具有正则化的效果——（正则化是一种用于控制机器学习模型复杂度的技术，目的是防止模型在训练数据上过拟合。）
- 提高模型的泛化能力——（模型的泛化能力指的是模型在未见过的新数据上的性能表现。一个好的机器学习模型应该能够在训练数据之外的数据上做出准确的预测或泛化。）
- 允许更高的学习速率从而加速收敛——（学习速率是深度学习和梯度下降优化算法中的一个重要超参数。它决定了在每次参数更新时，模型权重（参数）的调整幅度。）
- 批标准化有助于梯度传播，因此允许更深的网络，对于<font color=blue size=4>**有些特别深的网络，只有包含多个Batch Normalization层时才能进行训练。**</font>

#### 6）批标准化的实现：

- BatchNormalization层通常在卷积层或密集连接层之后使用

1. nn.BatchNorm1d()
2. nn.BatchNorm2d()

#### 7）批标准化的实现过程

1. 求每一个训练批次数据的<font color=red size=4>**均值**</font>
2. 求每一个训练批次数据的<font color=red size=4>**方差**</font>
3. 数据进行标准化
4. 训练参数<font color=red size=4>**Y，B**</font>
5. 输出y通过Y与B的线性变换得到原来的数值

- 在训练的正向传播中，不会改变当前输出，只记录下Y，B。
- 在反向传播的时候，根据求得的Y，B通过链式求导方式，求出学习速率以至改变权值。

#### 8）批标准化的使用位置

- model.train()
- model.eval()
- 训练模式：将<font color=blue size=4>**使用当前批输入的均值和方差**</font>>对其输入进行标准化。
- 推理模式：将<font color=blue size=4>**使用在训练期间学习的移动统计数据的均值和方差**</font>>来标准化其输入。
- 原始论文在CNN中一般应作用与非线性激活函数之前，但是，实际上放在激活函数之后效果可能更好。

### 8.7 四种天气图片分类 批标准化 BN层

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import torchvision
import os
import shutil

# 创建目录
base_dir = r'./dataset/4weather'					# 创建数据集基本目录
train_dir = os.path.join(base_dir, 'train')			# 在数据集基本目录创建训练集目录
test_dir = os.path.join(base_dir, 'test')			# 在数据集基本目录创建测试集目录

from torchvision import datasets, transforms

# 创建数据集对象——tranform、train_ds、test_ds
transform = transforms.Compose([
    transforms.Resize((96, 96)),					# 将输入的图像调整为大小为 96x96	
    transforms.ToTensor(),							# 将调整尺寸后的图像转换为 PyTorch 的张量（tensor）
    transforms.Normalize(mean=[0.5, 0.5, 0.5],		# 对图像进行标准化处理，以便将像素值范围缩放到 -1 到 1 之间。
                         std=[0.5, 0.5, 0.5])		# 它从每个通道（红色、绿色和蓝色）中减去均值（0.5）并将结果除以标准差（0.5）
])													# 助于提高模型的训练稳定性和性能，因为它有助于确保输入数据的分布类似于标准正态分布。
train_ds = torchvision.datasets.ImageFolder(		# 使用 PyTorch 的 ImageFolder 数据集类来创建一个数据集对象
    train_dir, 										# 包含训练图像数据的文件夹路径
    transform=transform								# 使用了之前定义的 transform 对象
)
test_ds = torchvision.datasets.ImageFolder(
    test_dir, 										# 包含测试图像数据的文件夹路径
    transform=transform
)

# 创建加载器——train_dl、test_dl
BATCHSIZE = 16										# 批量大小被设置为 16，意味着每次加载 16 个样本进行训练或测试
train_dl = torch.utils.data.DataLoader(				# 创建了一个训练数据加载器 train_dl。
    train_ds,
    batch_size=BATCHSIZE,
    shuffle=True									# 这个参数指示数据加载器在每个训练周期（epoch）开始时是否对训练数据进行洗牌（随机打乱顺序）
)													# 随机化数据的顺序可以帮助模型更好地学习。							
test_dl = torch.utils.data.DataLoader(
    test_ds,
    batch_size=BATCHSIZE
)

####################################################
# 创建模型（添加BN层——在卷积层和全链接层都添加一个BN）
###################################################

class Net(nn.Module):                       		# 创建模型
    def __init__(self):                      		# 对所有层进行初始化定义
        super(Net, self).__init__()          		# 继承所有父类属性
        self.conv1 = nn.Conv2d(3, 16, 3)     		# 定义第一个卷积层
        self.bn1 = nn.BatchNorm2d(16)               # 定义第一个BN层——参数看卷积层的输出
        self.conv2 = nn.Conv2d(16, 32, 3)    		# 定义第二个卷积层
        self.bn2 = nn.BatchNorm2d(32)               # 定义第二个BN层——参数看卷积层的输出
        self.conv3 = nn.Conv2d(32, 64, 3)    		# 定义第三个卷积层
        self.bn3 = nn.BatchNorm2d(64)               # 定义第三个BN层——参数看卷积层的输出
        self.drop = nn.Dropout(0.5)                 # 随机丢弃0.5的数据
        self.pool = nn.MaxPool2d(2, 2)       		# 定义池化层
        self.fc1 = nn.Linear(64*10*10, 1024) 		# 定义全连接层（卷积层之后的图像大小需要计算）
        self.bn_f1 = nn.BatchNorm1d(1024)           # 定义第一个全链接的BN层——参数看卷积层的输出
        self.fc2 = nn.Linear(1024, 256)        		# 定义全连接层
        self.bn_f2 = nn.BatchNorm1d(256)            # 定义第二个全链接的BN层——参数看卷积层的输出
        self.fc3 = nn.Linear(256, 4)                # 定义全连接层（最后输出为：4（4个分类））

    def forward(self, x):                   		# 定义调用前向传播过程
        x = self.pool(F.relu(self.conv1(x)))        # 第一个池化、激活、卷积层
        x = self.bn1(x)
        x = self.pool(F.relu(self.conv2(x)))  		# 第二个激活卷积层
        x = self.bn2(x)
        x = self.pool(F.relu(self.conv3(x)))        # 第三个激活卷积层
        x = self.bn3(x)
        x = x.view(-1, 64*10*10)
        x = F.relu(self.fc1(x))              		# 第四个激活全链接层
        x = self.bn_f1(x)
        x = self.drop(x)                            # 在全链接层增加一个Dropout层
        x = F.relu(self.fc2(x))              		# 第五个激活全链接层
        x = self.bn_f2(x)
        x = self.drop(x)                            # 在全链接层增加一个Dropout层
        x = self.fc3(x)                      		# 最后一个全链接层
        return x

model = Net()
if torch.cuda.is_available():
    model.to('cuda')

def fit(epoch, model, trainloader, testloader):
    # 训练模型
    correct = 0													# 新建正确个数变量
    total = 0													# 新建总的个数变量
    running_loss = 0											# 新建loss个数变量
    model.train()                                               # 将模型设为训练模式，Dropout发挥作用
    for x, y in trainloader:
        if torch.cuda.is_available():
            x, y = x.to('cuda'), y.to('cuda')
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        optim.zero_grad()
        loss.backward()
        optim.step()
        with torch.no_grad():
            y_pred = torch.argmax(y_pred, dim=1)				# 记录预测值
            correct += (y_pred == y).sum().item()				# 计算正确个数
            total += y.size(0)									# 计算总的个数（y.size(0)：y真实值，size(0):运行样本个数）
            running_loss += loss.item()							# 计算loss个数
    # 求解每个样本的loss和acc        
    epoch_loss = running_loss / len(trainloader.dataset)		# 求解每个样本的loss
    epoch_acc = correct / total									# 求解每个样本的正确率
    
    
    # 验证测试
    test_correct = 0
    test_total = 0
    test_running_loss = 0
    model.eval()                                                 # 将模型设为预测模式，Dropout不发挥作用
    with torch.no_grad():
        for  x, y in testloader:
            if torch.cuda.is_available():
                x, y = x.to('cuda'), y.to('cuda')
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            y_pred = torch.argmax(y_pred, dim=1)
            test_correct += (y_pred == y).sum().item()
            test_total += y.size(0)
            test_running_loss += loss.item()
    # 求解test的loss和acc         
    epoch_test_loss = test_running_loss / len(testloader.dataset)
    epoch_test_acc = test_correct / test_total
        
    print('epoch:', epoch, 
          'loss:', round(epoch_loss, 3),
          'accuracy:', round(epoch_acc, 3),
          'test_loss:', round(epoch_test_loss, 3),
          'test_accuracy:', round(epoch_test_acc, 3)
         )
        
    return epoch_loss, epoch_acc, epoch_test_loss, epoch_test_acc

# 参数设定
epochs = 30
loss_fn = nn.CrossEntropyLoss()
optim = torch.optim.Adam(model.parameters(), lr=0.001)

# 记录训练集和测试集的loss和acc
train_loss = []
train_acc = []
test_loss = []
test_acc = []
# 开始训练
for epoch in range(epochs):
    epoch_loss, epoch_acc, epoch_test_loss, epoch_test_acc = fit(epoch,
                                                                 model,
                                                                 train_dl,
                                                                 test_dl)
    train_loss.append(epoch_loss)
    train_acc.append(epoch_acc)
    test_loss.append(epoch_test_loss)
    test_acc.append(epoch_test_acc)
    
# 绘图
plt.plot(range(1, epochs+1), train_acc, label='train_acc')
plt.plot(range(1,epochs+1), test_acc, label='test_acc')
plt.legend()

plt.plot(range(1, epochs+1), train_loss, label='train_loss')
plt.plot(range(1,epochs+1), test_loss, label='test_loss')
plt.legend()
```

修改后的模型训练结果如下图：

![image-20230918163209594](https://raw.githubusercontent.com/Noregret327/picture/master/202309181632663.png)

### 8.8 超参数选择

#### 1）网络容量：

可以认为与<font color=blue size=4>网络中的可训练参数</font>成正比

#### 2）什么是超参数

所谓超参数，就是搭建神经网络中，需要我们自己选择的参数（不是通过梯度下降算法去优化的参数），比如：中间层的神经元个数、学习速率

#### 3）提高网络的拟合能力

最简单直接的方法是：

1. 增加层——这个更好
2. 增加隐藏神经元个数——这个效果没那么明显

这两个方法哪种更好：单纯增加神经元个数对于网络性能的提升并不明显，增加层会大大提高网络的拟合能力，这也是为什么现在深度学习的层越来越深的原因。<font color=red size=4>单层的神经元个数，不能太小，太小的话会造成信息瓶颈，使得模型欠拟合。</font>

#### 4）参数选择原则

<font color=red size=4>首先开发一个过拟合的模型：</font>

1. 添加更多的层
2. 让每一层变得更大
3. 训练更多的轮次

<font color=red size=4>然后抑制过拟合：</font>

1. dropout
2. 正则化
3. 图像增强

<font color=red size=4>再次调节超参数：</font>

- 学习速率
- 隐藏层单元数
- 训练轮次

<font color=red size=4>**总的原则：保证神经网络容量足够拟合数据！**</font>

<font color=blue size=4>！！！构建网络的总原则！！！：</font>

1. 增大网络容量，直到过拟合
2. 采取措施抑制过拟合
3. 继续增大网络容量，直到过拟合



## 9.预训练模型（迁移学习）

预训练网络是一个保存好的、之前已经在大型数据集（大规模图像分类任务）上训练好的卷积神经网络。

如果模型足够大，那么这个模型能够有效的提取实际特征的模型。即使新问题和新任务与原始任务完全不同，学习到的特征在不同问题之间是可移植的。——针对小数据问题也非常有效

### 9.1 pytorch内置预训练网络

Pytorch库包含模型框架有：

- VGG16
- VGG19
- densenet
- ResNet
- mobilenet
- Inception V3
- Xception

**关于 ImageNet：**

ImageNet是一个手动标注好类别的图片数据库，目前已有22 000个类别。——为了机器视觉研究

ILSVRC比赛：是一个视觉识别比赛，这个图片分类比赛是训练一个模型，能够将输入图片正确分类到1000个类别中的某个类别。训练集120万、验证集5万、测试集10万。1000个图片类别是我们日常中遇到的，如狗、猫、各种家具物品、车辆类型等。

### 9.2 预训练模型——VGG16

1.修改输入图片的大小：（192，192）——防止后面的网络比核还小

```python
transform = transforms.Compose([
    transforms.Resize((192, 192)),					# 将输入的图像调整为大小为 192 x 192	
    transforms.ToTensor(),							# 将调整尺寸后的图像转换为 PyTorch 的张量（tensor）
    transforms.Normalize(mean=[0.5, 0.5, 0.5],		# 对图像进行标准化处理，以便将像素值范围缩放到 -1 到 1 之间。
                         std=[0.5, 0.5, 0.5])		# 它从每个通道（红色、绿色和蓝色）中减去均值（0.5）并将结果除以标准差（0.5）
])
```

2.把自己定义手写的 model 从 torchvision 导入

```python
model = torchvision.models.vgg16(pretrained=True)
# 运行下载vgg16权重模型，可以提前下载模型（vgg16.pth）放在下载的路径里
```

3.冻结参数

- **迁移学习（Transfer Learning）**：在某些情况下，你可能希望利用预训练的模型来解决新的任务，而不是从头开始训练一个全新的模型。预训练的模型通常在大规模数据上进行了训练，学到了丰富的特征表示。然后，你可以将这个模型的部分或全部层次用于新任务，但不希望改变这些层次的参数。
- **避免权重更新**：通过将模型的参数的 `requires_grad` 设置为 `False`，你告诉 PyTorch 不要对这些参数进行梯度更新。这可以防止这些参数在训练中被误更新，从而保持了预训练模型的特征提取部分不变。
- **节省计算资源**：冻结不需要更新的参数可以减少计算和内存开销。在迁移学习中，通常只需微调模型的一部分，所以只有一部分参数需要梯度更新。

```python
for p in model.features.parameters():
    p.requires_grad = False

# 将一个 PyTorch 模型中的 requires_grad 设置为 False，从而冻结这些参数，防止它们在后续的训练中被更新。这种操作通常用于迁移学习或特定的模型微调场景。
```

4.修改模型的输出参数

因为天气输出只有四种结果，所以特征输出改为4

```python
model.classifier[-1].out_features = 4
# 优化函数也跟着修改
optim = torch.optim.Adam(model.classifier.parameters(), lr=0.0001)
```

#### 完整代码

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import torchvision
from torchvision import datasets, transforms
import os

# 创建目录
base_dir = r'./dataset/4weather'					# 创建数据集基本目录
train_dir = os.path.join(base_dir, 'train')			# 在数据集基本目录创建训练集目录
test_dir = os.path.join(base_dir, 'test')			# 在数据集基本目录创建测试集目录

# 创建数据集对象——tranform、train_ds、test_ds
transform = transforms.Compose([
    transforms.Resize((192, 192)),					# 将输入的图像调整为大小为 192x192	
    transforms.ToTensor(),							# 将调整尺寸后的图像转换为 PyTorch 的张量（tensor）
    transforms.Normalize(mean=[0.5, 0.5, 0.5],		# 对图像进行标准化处理，以便将像素值范围缩放到 -1 到 1 之间。
                         std=[0.5, 0.5, 0.5])		# 它从每个通道（红色、绿色和蓝色）中减去均值（0.5）并将结果除以标准差（0.5）
])													# 助于提高模型的训练稳定性和性能，因为它有助于确保输入数据的分布类似于标准正态分布。
train_ds = torchvision.datasets.ImageFolder(		# 使用 PyTorch 的 ImageFolder 数据集类来创建一个数据集对象
    train_dir, 										# 包含训练图像数据的文件夹路径
    transform=transform								# 使用了之前定义的 transform 对象
)
test_ds = torchvision.datasets.ImageFolder(
    test_dir, 										# 包含测试图像数据的文件夹路径
    transform=transform
)

# 创建加载器——train_dl、test_dl
BATCHSIZE = 16										# 批量大小被设置为 16，意味着每次加载 16 个样本进行训练或测试
train_dl = torch.utils.data.DataLoader(				# 创建了一个训练数据加载器 train_dl。
    train_ds,
    batch_size=BATCHSIZE,
    shuffle=True									# 这个参数指示数据加载器在每个训练周期（epoch）开始时是否对训练数据进行洗牌（随机打乱顺序）
)													# 随机化数据的顺序可以帮助模型更好地学习。							
test_dl = torch.utils.data.DataLoader(
    test_ds,
    batch_size=BATCHSIZE
)

# 模型加载
model = torchvision.models.vgg16(pretrained=True)

# 修改模型参数
# 冻结参数
for p in model.features.parameters():
    p.requires_grad = False
    
model.classifier[-1].out_features = 4 				# 修改模型输出
##################### 优化函数也要跟着修改 #####################
optim = torch.optim.Adam(model.classifier.parameters(), lr=0.0001)
loss_fn = nn.CrossEntropyLoss()


def fit(epoch, model, trainloader, testloader):
    # 训练模型
    correct = 0													# 新建正确个数变量
    total = 0													# 新建总的个数变量
    running_loss = 0											# 新建loss个数变量
    model.train()                                               # 将模型设为训练模式，Dropout发挥作用
    for x, y in trainloader:
        if torch.cuda.is_available():
            x, y = x.to('cuda'), y.to('cuda')
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        optim.zero_grad()
        loss.backward()
        optim.step()
        with torch.no_grad():
            y_pred = torch.argmax(y_pred, dim=1)				# 记录预测值
            correct += (y_pred == y).sum().item()				# 计算正确个数
            total += y.size(0)									# 计算总的个数（y.size(0)：y真实值，size(0):运行样本个数）
            running_loss += loss.item()							# 计算loss个数
    # 求解每个样本的loss和acc        
    epoch_loss = running_loss / len(trainloader.dataset)		# 求解每个样本的loss
    epoch_acc = correct / total									# 求解每个样本的正确率
    
    
    # 验证测试
    test_correct = 0
    test_total = 0
    test_running_loss = 0
    model.eval()                                                 # 将模型设为预测模式，Dropout不发挥作用
    with torch.no_grad():
        for  x, y in testloader:
            if torch.cuda.is_available():
                x, y = x.to('cuda'), y.to('cuda')
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            y_pred = torch.argmax(y_pred, dim=1)
            test_correct += (y_pred == y).sum().item()
            test_total += y.size(0)
            test_running_loss += loss.item()
    # 求解test的loss和acc         
    epoch_test_loss = test_running_loss / len(testloader.dataset)
    epoch_test_acc = test_correct / test_total
        
    print('epoch:', epoch, 
          'loss:', round(epoch_loss, 3),
          'accuracy:', round(epoch_acc, 3),
          'test_loss:', round(epoch_test_loss, 3),
          'test_accuracy:', round(epoch_test_acc, 3)
         )
        
    return epoch_loss, epoch_acc, epoch_test_loss, epoch_test_acc

# 设置训练次数
epochs = 10
# 记录训练集和测试集的loss和acc
train_loss = []
train_acc = []
test_loss = []
test_acc = []
# 开始训练
for epoch in range(epochs):
    epoch_loss, epoch_acc, epoch_test_loss, epoch_test_acc = fit(epoch,
                                                                 model,
                                                                 train_dl,
                                                                 test_dl)
    train_loss.append(epoch_loss)
    train_acc.append(epoch_acc)
    test_loss.append(epoch_test_loss)
    test_acc.append(epoch_test_acc)
   
# 绘图
plt.plot(range(1, epochs+1), train_acc, label='train_acc')
plt.plot(range(1,epochs+1), test_acc, label='test_acc')
plt.legend()

plt.plot(range(1, epochs+1), train_loss, label='train_loss')
plt.plot(range(1,epochs+1), test_loss, label='test_loss')
plt.legend()
```

输出结果：

![image-20230918210348957](https://raw.githubusercontent.com/Noregret327/picture/master/202309182103027.png)

### 9.3 数据增强

数据增强方法主要是：人为的对图片进行变换、缩放、裁剪、翻转等

```python
# 常用的数据增强模型：
# 对位置方面处理
transforms.RandomCrop                                 # 随机位置裁剪
transforms.CenterCrop                                 # 中间位置裁剪
transforms.RandomHorizontalFlip                       # 随机水平翻转
transforms.RandomVerticalFlip                         # 随机上下翻转
transforms.RandomRotation                             # 随机旋转一个角度

# 对颜色方面处理
transforms.ColorJitter(brightness=1)                  # 明亮度调整 
transforms.ColorJitter(contrast=1)                    # 对比度
transforms.ColorJitter(saturation=0.5)                # 饱和度
transforms.ColorJitter(hue=0.5)                       # 颜色
ransforms.RandomGrayscale(p=0.5)                      # 随机灰度化
```

1.对训练集的数据进行处理，测试集的保持不变

```python
# 创建数据集对象——tranform、train_ds、test_ds
train_transform = transforms.Compose([
    transforms.Resize(224),                                # 裁剪最大值为224图片
    transforms.RandomCrop(192),                            # 随机裁剪
    transforms.RandomHorizontalFlip(),                     # 随机翻转
    transforms.RandomRotation(0.2),                        # 随机旋转
    transforms.ColorJitter(brightness=0.5),                # 添加明亮度
    transforms.ColorJitter(contrast=1),                    # 添加对比度
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])
])
test_transform = transforms.Compose([
    transforms.Resize((192, 192)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])
])
# 加载数据集
train_ds = torchvision.datasets.ImageFolder(
    train_dir,
    transform=train_transform
)
test_ds = torchvision.datasets.ImageFolder(
    test_dir, 
    transform=test_transform
)
```

#### 完整代码

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import torchvision
import os
from torchvision import datasets, transforms

# 创建目录
base_dir = r'./dataset/4weather'					# 创建数据集基本目录
train_dir = os.path.join(base_dir, 'train')			# 在数据集基本目录创建训练集目录
test_dir = os.path.join(base_dir, 'test')			# 在数据集基本目录创建测试集目录

# 创建数据集对象——tranform、train_ds、test_ds
train_transform = transforms.Compose([
    transforms.Resize(224),                                # 裁剪最大值为224图片
    transforms.RandomCrop(192),                            # 随机裁剪
    transforms.RandomHorizontalFlip(),                     # 随机翻转
    transforms.RandomRotation(0.2),                        # 随机旋转
    transforms.ColorJitter(brightness=0.5),                # 添加明亮度
    transforms.ColorJitter(contrast=1),                    # 添加对比度
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])
])
test_transform = transforms.Compose([
    transforms.Resize((192, 192)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])
])

train_ds = torchvision.datasets.ImageFolder(
    train_dir,
    transform=train_transform
)
test_ds = torchvision.datasets.ImageFolder(
    test_dir, 
    transform=test_transform
)

# 创建加载器——train_dl、test_dl
BATCHSIZE = 16										# 批量大小被设置为 16，意味着每次加载 16 个样本进行训练或测试
train_dl = torch.utils.data.DataLoader(				# 创建了一个训练数据加载器 train_dl。
    train_ds,
    batch_size=BATCHSIZE,
    shuffle=True									# 这个参数指示数据加载器在每个训练周期（epoch）开始时是否对训练数据进行洗牌（随机打乱顺序）
)													# 随机化数据的顺序可以帮助模型更好地学习。							
test_dl = torch.utils.data.DataLoader(
    test_ds,
    batch_size=BATCHSIZE
)

# 加载模型
model = torchvision.models.vgg16(pretrained=True)

# 修改模型
for p in model.features.parameters():
    p.requires_grad = False
model.classifier[-1].out_features = 4
optim = torch.optim.Adam(model.classifier.parameters(), lr=0.0001)
loss_fn = nn.CrossEntropyLoss()

def fit(epoch, model, trainloader, testloader):
    # 训练模型
    correct = 0													# 新建正确个数变量
    total = 0													# 新建总的个数变量
    running_loss = 0											# 新建loss个数变量
    model.train()                                               # 将模型设为训练模式，Dropout发挥作用
    for x, y in trainloader:
        if torch.cuda.is_available():
            x, y = x.to('cuda'), y.to('cuda')
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        optim.zero_grad()
        loss.backward()
        optim.step()
        with torch.no_grad():
            y_pred = torch.argmax(y_pred, dim=1)				# 记录预测值
            correct += (y_pred == y).sum().item()				# 计算正确个数
            total += y.size(0)									# 计算总的个数（y.size(0)：y真实值，size(0):运行样本个数）
            running_loss += loss.item()							# 计算loss个数
    # 求解每个样本的loss和acc        
    epoch_loss = running_loss / len(trainloader.dataset)		# 求解每个样本的loss
    epoch_acc = correct / total									# 求解每个样本的正确率
    
    
    # 验证测试
    test_correct = 0
    test_total = 0
    test_running_loss = 0
    model.eval()                                                 # 将模型设为预测模式，Dropout不发挥作用
    with torch.no_grad():
        for  x, y in testloader:
            if torch.cuda.is_available():
                x, y = x.to('cuda'), y.to('cuda')
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            y_pred = torch.argmax(y_pred, dim=1)
            test_correct += (y_pred == y).sum().item()
            test_total += y.size(0)
            test_running_loss += loss.item()
    # 求解test的loss和acc         
    epoch_test_loss = test_running_loss / len(testloader.dataset)
    epoch_test_acc = test_correct / test_total
        
    print('epoch:', epoch, 
          'loss:', round(epoch_loss, 3),
          'accuracy:', round(epoch_acc, 3),
          'test_loss:', round(epoch_test_loss, 3),
          'test_accuracy:', round(epoch_test_acc, 3)
         )
        
    return epoch_loss, epoch_acc, epoch_test_loss, epoch_test_acc
    
epochs = 10
# 记录训练集和测试集的loss和acc
train_loss = []
train_acc = []
test_loss = []
test_acc = []
# 开始训练
for epoch in range(epochs):
    epoch_loss, epoch_acc, epoch_test_loss, epoch_test_acc = fit(epoch,
                                                                 model,
                                                                 train_dl,
                                                                 test_dl)
    train_loss.append(epoch_loss)
    train_acc.append(epoch_acc)
    test_loss.append(epoch_test_loss)
    test_acc.append(epoch_test_acc)
   
# 绘图   
plt.plot(range(1, epochs+1), train_acc, label='train_acc')
plt.plot(range(1,epochs+1), test_acc, label='test_acc')
plt.legend()

plt.plot(range(1, epochs+1), train_loss, label='train_loss')
plt.plot(range(1,epochs+1), test_loss, label='test_loss')
plt.legend()    
```

输出：

![image-20230918211438158](C:/Users/14224/AppData/Roaming/Typora/typora-user-images/image-20230918211438158.png)



### 9.4 学习速率衰减

1.手写数据学习速率衰减方法

```python
# 手写数据衰减方法
for p in optim.param_groups:
    p['lr'] *= 0.9
```

2.学习速率衰减导入

```python
from torch.optim import lr_scheduler

exp_lr_scheduler = lr_scheduler.StepLR(optim, step_size=5, gamma=0.1)

# # 一些常用的学习速率衰减
# lr_scheduler.StepLR(optim, step_size=5, gamma=0.1)
# lr_scheduler.MultiStepLR(optim, [20, 50, 80], gamma=0.1)
# lr_scheduler.ExponentialLR(optim, gamma=0.1)
```

3.训练方式的修改

```python
def fit(epoch, model, trainloader, testloader):
    # 训练模型
    correct = 0													# 新建正确个数变量
    total = 0													# 新建总的个数变量
    running_loss = 0											# 新建loss个数变量
    model.train()                                               # 将模型设为训练模式，Dropout发挥作用
    for x, y in trainloader:
        if torch.cuda.is_available():
            x, y = x.to('cuda'), y.to('cuda')
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        optim.zero_grad()
        loss.backward()
        optim.step()
        with torch.no_grad():
            y_pred = torch.argmax(y_pred, dim=1)				# 记录预测值
            correct += (y_pred == y).sum().item()				# 计算正确个数
            total += y.size(0)									# 计算总的个数（y.size(0)：y真实值，size(0):运行样本个数）
            running_loss += loss.item()							# 计算loss个数
    
    # /////////////////////////////增加一步对学习速率衰减//////////////////////
    exp_lr_scheduler.step()
    //////////////////////////////////////////////////////////////////////
    # 求解每个样本的loss和acc        
    epoch_loss = running_loss / len(trainloader.dataset)		# 求解每个样本的loss
    epoch_acc = correct / total									# 求解每个样本的正确率
    
    
    # 验证测试
    test_correct = 0
    test_total = 0
    test_running_loss = 0
    model.eval()                                                 # 将模型设为预测模式，Dropout不发挥作用
    with torch.no_grad():
        for  x, y in testloader:
            if torch.cuda.is_available():
                x, y = x.to('cuda'), y.to('cuda')
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            y_pred = torch.argmax(y_pred, dim=1)
            test_correct += (y_pred == y).sum().item()
            test_total += y.size(0)
            test_running_loss += loss.item()
    # 求解test的loss和acc         
    epoch_test_loss = test_running_loss / len(testloader.dataset)
    epoch_test_acc = test_correct / test_total
        
    print('epoch:', epoch, 
          'loss:', round(epoch_loss, 3),
          'accuracy:', round(epoch_acc, 3),
          'test_loss:', round(epoch_test_loss, 3),
          'test_accuracy:', round(epoch_test_acc, 3)
         )
        
    return epoch_loss, epoch_acc, epoch_test_loss, epoch_test_acc
```

#### 完整代码

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import torchvision
import os
from torchvision import datasets, transforms

# 创建目录
base_dir = r'./dataset/4weather'					# 创建数据集基本目录
train_dir = os.path.join(base_dir, 'train')			# 在数据集基本目录创建训练集目录
test_dir = os.path.join(base_dir, 'test')			# 在数据集基本目录创建测试集目录

# 创建数据集对象——tranform、train_ds、test_ds
train_transform = transforms.Compose([
    transforms.Resize(224),                                # 裁剪最大值为224图片
    transforms.RandomCrop(192),                            # 随机裁剪
    transforms.RandomHorizontalFlip(),                     # 随机翻转
    transforms.RandomRotation(0.2),                        # 随机旋转
    transforms.ColorJitter(brightness=0.5),                # 添加明亮度
    transforms.ColorJitter(contrast=1),                    # 添加对比度
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])
])
test_transform = transforms.Compose([
    transforms.Resize((192, 192)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])
])

train_ds = torchvision.datasets.ImageFolder(
    train_dir,
    transform=train_transform
)
test_ds = torchvision.datasets.ImageFolder(
    test_dir, 
    transform=test_transform
)

# 创建加载器——train_dl、test_dl
BATCHSIZE = 16										# 批量大小被设置为 16，意味着每次加载 16 个样本进行训练或测试
train_dl = torch.utils.data.DataLoader(				# 创建了一个训练数据加载器 train_dl。
    train_ds,
    batch_size=BATCHSIZE,
    shuffle=True									# 这个参数指示数据加载器在每个训练周期（epoch）开始时是否对训练数据进行洗牌（随机打乱顺序）
)													# 随机化数据的顺序可以帮助模型更好地学习。							
test_dl = torch.utils.data.DataLoader(
    test_ds,
    batch_size=BATCHSIZE
)

# 加载模型
model = torchvision.models.vgg16(pretrained=True)
# 冻结参数
for p in model.features.parameters():
    p.requires_grad = False
# 模型参数修改
model.classifier[-1].out_features = 4
optim = torch.optim.Adam(model.classifier.parameters(), lr=0.0001)
loss_fn = nn.CrossEntropyLoss()

########### 导入学习速率衰减 #############
from torch.optim import lr_scheduler
exp_lr_scheduler = lr_scheduler.StepLR(optim, step_size=5, gamma=0.1)

# 定义训练
def fit(epoch, model, trainloader, testloader):
    # 训练模型
    correct = 0													# 新建正确个数变量
    total = 0													# 新建总的个数变量
    running_loss = 0											# 新建loss个数变量
    model.train()                                               # 将模型设为训练模式，Dropout发挥作用
    for x, y in trainloader:
        if torch.cuda.is_available():
            x, y = x.to('cuda'), y.to('cuda')
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        optim.zero_grad()
        loss.backward()
        optim.step()
        with torch.no_grad():
            y_pred = torch.argmax(y_pred, dim=1)				# 记录预测值
            correct += (y_pred == y).sum().item()				# 计算正确个数
            total += y.size(0)									# 计算总的个数（y.size(0)：y真实值，size(0):运行样本个数）
            running_loss += loss.item()							# 计算loss个数
    
    # /////////////////////////////增加一步对学习速率衰减//////////////////////
    exp_lr_scheduler.step()
    //////////////////////////////////////////////////////////////////////
    # 求解每个样本的loss和acc        
    epoch_loss = running_loss / len(trainloader.dataset)		# 求解每个样本的loss
    epoch_acc = correct / total									# 求解每个样本的正确率
    
    
    # 验证测试
    test_correct = 0
    test_total = 0
    test_running_loss = 0
    model.eval()                                                 # 将模型设为预测模式，Dropout不发挥作用
    with torch.no_grad():
        for  x, y in testloader:
            if torch.cuda.is_available():
                x, y = x.to('cuda'), y.to('cuda')
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            y_pred = torch.argmax(y_pred, dim=1)
            test_correct += (y_pred == y).sum().item()
            test_total += y.size(0)
            test_running_loss += loss.item()
    # 求解test的loss和acc         
    epoch_test_loss = test_running_loss / len(testloader.dataset)
    epoch_test_acc = test_correct / test_total
        
    print('epoch:', epoch, 
          'loss:', round(epoch_loss, 3),
          'accuracy:', round(epoch_acc, 3),
          'test_loss:', round(epoch_test_loss, 3),
          'test_accuracy:', round(epoch_test_acc, 3)
         )
        
    return epoch_loss, epoch_acc, epoch_test_loss, epoch_test_acc

# 训练次数
epochs = 10
# 记录训练集和测试集的loss和acc
train_loss = []
train_acc = []
test_loss = []
test_acc = []
# 开始训练
for epoch in range(epochs):
    epoch_loss, epoch_acc, epoch_test_loss, epoch_test_acc = fit(epoch,
                                                                 model,
                                                                 train_dl,
                                                                 test_dl)
    train_loss.append(epoch_loss)
    train_acc.append(epoch_acc)
    test_loss.append(epoch_test_loss)
    test_acc.append(epoch_test_acc)

# 绘图
plt.plot(range(1, epochs+1), train_acc, label='train_acc')
plt.plot(range(1,epochs+1), test_acc, label='test_acc')
plt.legend()

plt.plot(range(1, epochs+1), train_loss, label='train_loss')
plt.plot(range(1,epochs+1), test_loss, label='test_loss')
plt.legend()
```

输出：

![image-20230918211517106](C:/Users/14224/AppData/Roaming/Typora/typora-user-images/image-20230918211517106.png)



### 9.5 预训练网络——ResNet

1.导入模型

```python
model = torchvision.models.resnet18(pretrained=True)
```

2.修改模型

```python
# 冻结参数
for param in model.parameters():
    param.requires_grad = False

# 提取最后一层
in_f = model.fc.in_features
model.fc = nn.Linear(in_f, 4)          # 将ResNet网络最后一层替换掉

# 修改优化函数
optim = torch.optim.Adam(model.fc.parameters(), lr=0.0001)
```

#### 完整代码

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import torchvision
import os
from torchvision import datasets, transforms

# 创建目录
base_dir = r'./dataset/4weather'					# 创建数据集基本目录
train_dir = os.path.join(base_dir, 'train')			# 在数据集基本目录创建训练集目录
test_dir = os.path.join(base_dir, 'test')			# 在数据集基本目录创建测试集目录

# 创建数据集对象——tranform、train_ds、test_ds
train_transform = transforms.Compose([
    transforms.Resize(224),                                # 裁剪最大值为224图片
    transforms.RandomCrop(192),                            # 随机裁剪
    transforms.RandomHorizontalFlip(),                     # 随机翻转
    transforms.RandomRotation(0.2),                        # 随机旋转
    transforms.ColorJitter(brightness=0.5),                # 添加明亮度
    transforms.ColorJitter(contrast=1),                    # 添加对比度
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])
])
test_transform = transforms.Compose([
    transforms.Resize((192, 192)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])
])

train_ds = torchvision.datasets.ImageFolder(
    train_dir,
    transform=train_transform
)
test_ds = torchvision.datasets.ImageFolder(
    test_dir, 
    transform=test_transform
)

# 创建加载器——train_dl、test_dl
BATCHSIZE = 32										# 批量大小被设置为 32，意味着每次加载 32 个样本进行训练或测试
train_dl = torch.utils.data.DataLoader(				# 创建了一个训练数据加载器 train_dl。
    train_ds,
    batch_size=BATCHSIZE,
    shuffle=True									# 这个参数指示数据加载器在每个训练周期（epoch）开始时是否对训练数据进行洗牌（随机打乱顺序）
)													# 随机化数据的顺序可以帮助模型更好地学习。							
test_dl = torch.utils.data.DataLoader(
    test_ds,
    batch_size=BATCHSIZE
)

# 加载模型
model = torchvision.models.resnet18(pretrained=True)
# 修改模型参数
for param in model.parameters():
    param.requires_grad = False
# 提取模型最后一层
in_f = model.fc.in_features    
model.fc = nn.Linear(in_f, 4)          # 将ResNet网络最后一层替换掉

# 放在GPU训练
if torch.cuda.is_available():
    model.to('cuda')
    
# 优化函数——要跟着模型修改
optim = torch.optim.Adam(model.fc.parameters(), lr=0.0001)

# 损失函数
loss_fn = nn.CrossEntropyLoss()

def fit(epoch, model, trainloader, testloader):
    # 训练模型
    correct = 0
    total = 0
    running_loss = 0
    model.train()
    for x, y in trainloader:
        if torch.cuda.is_available():
            x, y = x.to('cuda'), y.to('cuda')
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        optim.zero_grad()
        loss.backward()
        optim.step()
        with torch.no_grad():
            y_pred = torch.argmax(y_pred, dim=1)
            correct += (y_pred == y).sum().item()
            total += y.size(0)
            running_loss += loss.item()
    
    # 求解每个样本的loss和acc        
    epoch_loss = running_loss / len(trainloader.dataset)
    epoch_acc = correct / total
    
    
    # 验证测试
    test_correct = 0
    test_total = 0
    test_running_loss = 0
    model.eval()                                           
    with torch.no_grad():
        for  x, y in testloader:
            if torch.cuda.is_available():
                x, y = x.to('cuda'), y.to('cuda')
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            y_pred = torch.argmax(y_pred, dim=1)
            test_correct += (y_pred == y).sum().item()
            test_total += y.size(0)
            test_running_loss += loss.item()
    # 求解test的loss和acc         
    epoch_test_loss = test_running_loss / len(testloader.dataset)
    epoch_test_acc = test_correct / test_total
        
    print('epoch:', epoch, 
          'loss:', round(epoch_loss, 3),
          'accuracy:', round(epoch_acc, 3),
          'test_loss:', round(epoch_test_loss, 3),
          'test_accuracy:', round(epoch_test_acc, 3)
         )
        
    return epoch_loss, epoch_acc, epoch_test_loss, epoch_test_acc

# 训练次数
epochs = 50
# 记录训练集和测试集的loss和acc
train_loss = []
train_acc = []
test_loss = []
test_acc = []
# 开始训练
for epoch in range(epochs):
    epoch_loss, epoch_acc, epoch_test_loss, epoch_test_acc = fit(epoch,
                                                                 model,
                                                                 train_dl,
                                                                 test_dl)
    train_loss.append(epoch_loss)
    train_acc.append(epoch_acc)
    test_loss.append(epoch_test_loss)
    test_acc.append(epoch_test_acc)

# 绘图
plt.plot(range(1, epochs+1), train_acc, label='train_acc')
plt.plot(range(1,epochs+1), test_acc, label='test_acc')
plt.legend()

plt.plot(range(1, epochs+1), train_loss, label='train_loss')
plt.plot(range(1,epochs+1), test_loss, label='test_loss')
plt.legend()    
```

输出：

![image-20230918212508809](https://raw.githubusercontent.com/Noregret327/picture/master/202309182125873.png)
