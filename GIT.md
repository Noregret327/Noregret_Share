# GIT

## GIT基本操作

每次在要上传资料前必须！！！！**先pull下来**

* 将远程仓库资料同步到本地

```bash
git pull origin
```

* 本地仓库更新好后，准备上传，先堆入暂存区

```bash
git add .
```

`这里这个指令是将整个仓库更新部分都堆入暂存区，如果说还有没有完成的笔记或资料，就不要用全部堆入`

```bash
git add 具体文件或文件夹
```

* 命名本次更新的内容

```bash
git commit -m "第一次更新"
```

* 将本地仓库更新到远程仓库

```bash
git push
```



## 创建git仓库/并创造分支

1、首先在git网站进行创建

2、克隆到本地

```bash
git clone http
```

3、可以查看master分支

```bash    
git branch
```

4、创建分支dev

> 主要分支：Master和Develop。前者用于正式发布，后者用于日常开发。

```bash
git branch dev
```

5、打开分支dev

```bash
git switch dev
```

6、推送到暂存区

```bash
git add ./
```

7、检查状态

```bash
git statu
```

8、写commit

```bash
git commit -m "更新"
```

9、推送

```bash
git push origin dev
```

10、到github 网址完成 分支推送请求

11、可以通过git switch去切换主分支和分支，最好我们在分支修改完后推送到上去，然后再上去github网站修改合并，这样可以保证主分支不会混乱，记得每次pull下来



## 重新连接git

1、重新配置“名字”和“邮件”

```
git config --global user.name"name"
git config --global user.email"520@qq.com"
```

name：要改为自己想设置的名字

520@qq.com：改为自己想设置的邮箱



2、删除".ssh"文件夹的“Know_hosts”

".ssh"文件一般在“C:\Users\Administrator”里面



3、获取ssh

```
ssh-keygen -t rsa -C "520@qq.com"
```

然后一直回车知道结束，系统会在之前的“.ssh”文件夹里生成“id_rsa.pub”文件，并用记事本打开复制里面的内容



4、打开https://github.com/，在ssh设置里把“id_rsa.pub”文件里的内容粘贴进去就行



## 恢复版本

```bash
git checkout HEAD text.py
```

从最后一次提交里面恢复版本



## 增加单个文件

```bash
git add text.py
```

加入暂存区