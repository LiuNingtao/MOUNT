# MOUNT GitHub Upload Script (PowerShell)
# 使用前请确保已安装Git并配置好GitHub认证

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "MOUNT GitHub Upload Script" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

$GITHUB_REPO = "https://github.com/LiuNingtao/MOUNT.git"
$BRANCH_NAME = "main"

# Step 1: 检查git状态
Write-Host "[1/6] 检查git状态..." -ForegroundColor Yellow
git status
Write-Host ""

# Step 2: 添加/设置远程仓库
Write-Host "[2/6] 添加远程仓库..." -ForegroundColor Yellow
git remote add mount $GITHUB_REPO 2>$null
git remote set-url mount $GITHUB_REPO
Write-Host "远程仓库已设置为: $GITHUB_REPO" -ForegroundColor Green
Write-Host ""

# Step 3: 添加所有更改
Write-Host "[3/6] 添加所有更改..." -ForegroundColor Yellow
git add .
Write-Host ""

# Step 4: 提交更改
Write-Host "[4/6] 提交更改..." -ForegroundColor Yellow
$COMMIT_MSG = Read-Host "请输入提交信息 (默认: Update MOUNT)"
if ([string]::IsNullOrWhiteSpace($COMMIT_MSG)) {
    $COMMIT_MSG = "Update MOUNT"
}
git commit -m $COMMIT_MSG
Write-Host ""

# Step 5: 推送到GitHub
Write-Host "[5/6] 推送到GitHub..." -ForegroundColor Yellow
Write-Host "正在推送到 $GITHUB_REPO ($BRANCH_NAME)" -ForegroundColor Green
git push -u mount $BRANCH_NAME
Write-Host ""

# Step 6: 完成
Write-Host "[6/6] 完成！" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "上传成功！" -ForegroundColor Green
Write-Host "访问: $GITHUB_REPO" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Read-Host "按Enter键退出"
