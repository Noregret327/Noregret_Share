# Anaconda指令

## 一、基本指令

```python
conda create -n 环境名 python=3.8				//创建环境
conda info --envs							  //查看环境
conda activate 环境名							//激活环境
deactivate 环境名								//退出环境
conda remove -n 环境名 --all					//删除环境

//安装环境包
conda install 包名称
pip install 包名称 -i https://pypi.tuna.tsinghua.edu.cn/simple		//（清华镜像）
pip install 包名称 -i https://pypi.doubanio.com/simple/ 			//（豆瓣镜像）

//查看环境包
conda list
pip list

//查看配置
conda config --show
```



## 二、环境包

### 1、sklearn

```python
pip install -U scikit-learn

conda install scikit-learn

//升级scikit-learn：
conda update scikit-learn

//卸载scikit-learn：
conda remove scikit-learn
```

### 2、CV2

```python
pip install opencv-python
```

### 3、kera

```python
pip install keras -i https://pypi.doubanio.com/simple
```

### 4、keras.backend.tensorflow_backend

```
pip install tensorflow==2.2.0
pip install Keras==2.2.0
```

### 5、tensorflow

```python
pip install tensorflow
```

### 6、sklearn

```python
conda install scikit-learn
pip install -U scikit-learn
```

### 7、charset_normalizer

```
python -m pip install requests chardet
```

### 8、pandas

```
pip install pandas
```

