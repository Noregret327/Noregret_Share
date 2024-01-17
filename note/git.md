# GIT

## 常用命令

###### 状态查看

```
git status
```

###### 初始化

```
git init
```

###### 下载项目

```
git clone http：...（项目的GitHub）
```

###### 提交到本地

```
git add .
git commit -m"更新了XX"
```

###### 拉取和推送

```
git pull
git push
```

###### 版本查看与切换

```
git reflog
git reset --hard xxxx
gitreset --hard HEAD~2(后退步数)
```

###### 查看global信息

```
git config --global -l
```



## 分支

###### 查看当前分支

```git
git branch
```

###### 创建分支并切换到创建分支

```git
# dev：分支名
git checkout -b dev
```

###### 切换回主分支

```
# master：主分支
git checkout master
```

###### 合并分支

```
# 将dev合并到master——（先切换到主分支，再合并）
git checkout master
git merge dev
```

###### 删除分支

```
git branch -d dev
```

###### 分支的拉取和推送

```
git pull origin dev
git push origin dev
```





## 问题

### **解决问题：**

git pull fatal: unable to access 'https://github.com/Noregret327/andorid_socket_scale_demo.git/': 

Failed to connect to github.com port 443 after 21111 ms: Couldn't connect to server

或者

fatal: unable to access ‘https://github.com/.../.git‘: Could not resolve host: github.com

### **解决方案：**

用cmd命令进行刷新dns缓存

```cmd
ipconfig/flushdns
```

用git清理dns缓存

```
git config --global --unset http.proxy 
git config --global --unset https.proxy
```

或者

```
git config --global http.sslVerifyfalse
git config --global --unset http.proxy
git config --global --unset https.proxy
git config --global http.sslBackend "openssl"
```

再或者

```
export http_proxy=""
export https_proxy=""
export HTTP_PROXY=""
export HTTPS_PROXY=""
```

再或者

```
unset http_proxy
unset ftp_proxy
unset all_proxy
unset https_proxy
unset no_proxy
```



### 关于SSH

**测SSH指令**

```
ssh -T git@github.com
```

**重新生成SSH**

```
ssh-keygen -t rsa -C “your_email.com”
```



### 关于ping不通/超时

**1.错误检测**

```
ping www.baidu.com
```

百度能ping通说明网络没问题

```
ping github.com
```

GitHub ping不通本地DNS无法解析导致

**2.解决方法**

路径：C:\Windows\System32\drivers\etc\hosts

末尾添加

```
192.30.255.112  github.com git 
185.31.16.184 github.global.ssl.fastly.net  
```

或

```
20.205.243.166 github.com
108.160.170.44 github.global.ssl.fastly.net
```

