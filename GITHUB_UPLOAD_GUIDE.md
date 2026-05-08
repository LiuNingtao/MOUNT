# GitHub上传指南

本指南说明如何将MOUNT项目上传到GitHub仓库。

## 前置要求

1. **安装Git**: 确保已在系统上安装Git
   - 下载地址: https://git-scm.com/downloads

2. **配置GitHub认证**: 
   - 使用GitHub账号登录
   - 建议配置SSH密钥或使用Personal Access Token

## 使用方法

### 方法1: 使用批处理脚本 (推荐Windows用户)

双击运行:
```
upload_to_github.bat
```

### 方法2: 使用PowerShell脚本

在PowerShell中运行:
```powershell
.\upload_to_github.ps1
```

### 方法3: 手动执行命令

如果你想手动执行，可以按照以下步骤:

```bash
# 1. 添加远程仓库
git remote add mount https://github.com/LiuNingtao/MOUNT.git

# 2. 添加所有更改
git add .

# 3. 提交更改
git commit -m "Update MOUNT"

# 4. 推送到GitHub
git push -u mount main
```

## 首次上传注意事项

### 如果GitHub仓库是空的:

1. 先在GitHub上创建仓库 (https://github.com/LiuNingtao/MOUNT)
2. 确保仓库名称正确
3. 运行上传脚本

### 如果GitHub仓库已有内容:

脚本会自动将本地更改推送到远程仓库。

## 常见问题

### Q: 提示"remote already exists"
A: 正常，脚本会自动更新远程仓库URL。

### Q: 认证失败
A: 请确保:
- 已配置GitHub账号
- 使用Personal Access Token (推荐)
- 或使用SSH方式

### Q: 如何创建Personal Access Token?
1. 访问 https://github.com/settings/tokens
2. 点击 "Generate new token"
3. 选择 `repo` 权限
4. 生成并保存token

### Q: 想推送到不同的分支?
修改脚本中的 `BRANCH_NAME` 变量，或手动指定:
```bash
git push -u mount your-branch-name
```

## 查看远程仓库

```bash
# 查看当前远程仓库
git remote -v

# 查看所有分支
git branch -a
```

## 其他有用的Git命令

```bash
# 查看状态
git status

# 查看提交历史
git log --oneline

# 拉取远程更新
git pull mount main

# 查看差异
git diff
```
