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



### 6.1 完整代码

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

# 绘制acc变化
plt.plot(range(1, epochs+1), train_acc, label='train_acc')
plt.plot(range(1, epochs+1), test_acc, label='test_acc')
plt.legend()
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

- <font color=0x0ff size=4>对梯度进行缩放的参数</font>被称为<font color=red size=5>学习速率（learning rate）</font>
- 如果学习速率太小，则找到损失函数极小值点时可能需要许多轮迭代；如果太大，则算法可能会“跳过”极小值点并且因周期性的“跳跃”而永远无法找到极小值点。

