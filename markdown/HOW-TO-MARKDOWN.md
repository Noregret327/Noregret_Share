# 本文记录了一些常用的markdown语法
## 方便书写时查看
> *We* *believe* *that* *writing* *is* *about* *content*, *about* *what* *you* *want* *to* *say* – *not* *about* *fancy* *formatting*.
***
```
标题
# 一级标题
## 二级标题
### 三级标题
#### 四级标题
##### 五级标题
###### 六级标题
```
# 一级标题
## 二级标题
### 三级标题
#### 四级标题
##### 五级标题
###### 六级标题
***
```
无序列表
- 1
+ 2
* 3
```
* 1
- 2
+ 3
***
```
有序列表
1. 1
2. 2
3. 3
```
1. 1
2. 2
3. 3
***
```
列表嵌套
上一级与下一级之间加三个空格
```
1. 1
   1. 1 
      1. 1
***
```
引用
> 一级引用
>> 二级引用
>> 三级引用
```
>一级引用
>>二级引用
>>>三级引用
***
`字体加粗`
`**A**`
**A**
`字体倾斜`
`*A*`
*A*
`字体倾斜加粗`
`***A***`
***A***
`字体删除线`
`~~ABCDEF~~`
~~ABCDEF~~
***
`分割线`
`***`
***
***
`流程图`
```
flow
st=>start: 开始
op=>operation: My Operation
cond=>condition: Yes or No?
e=>end
st->op->cond
cond(yes)->e
cond(no)->op
```

```flow
st=>start: 开始
op=>operation: My Operation
cond=>condition: Yes or No?
e=>end
st->op->cond
cond(yes)->e
cond(no)->op
```
***
`表格`
```
|1|2|3|
|:---|:---:|---:|
|文字居左|文字居中|文字居右|
```
|1|2|3|
|:---|:---:|---:|
|文字居左|文字居中|文字居右|
***
```
`单行代码`

(```）
代码块
代码块
（```)
此处括号为了防止转义
```
`单行代码`
```
代码块
代码块
```
***
```
超链接
[链接名称](网址)
[百度](http://baidu.com)
```
[百度](http://baidu.com)
***
```
图片
![图片alt](图片地址 ''图片title'')
语法中图片Alt的意思是如果图片因为某些原因不能显示，就用定义的图片Alt文字来代替图片。 
图片Title则和链接中的Title一样，表示鼠标悬停与图片上时出现的文字。 
Alt 和 Title 都不是必须的，可以省略，但建议写上。
```
![blockchain](https://ss0.bdstatic.com/70cFvHSh_Q1YnxGkpoWK1HF6hhy/it/u=702257389,1274025419&fm=27&gp=0.jpg "区块链")
***
[在MARKDOWN中使用特殊字符](https://blog.csdn.net/vola9527/article/details/69948411)
***
[参考1](https://www.jianshu.com/p/ebe52d2d468f)
[参考2](https://sspai.com/post/25137)
[参考3](https://blog.csdn.net/qq_40942329/article/details/78724322)
[参考4流程图](https://segmentfault.com/a/1190000006247465)
***

