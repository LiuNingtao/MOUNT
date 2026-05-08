@echo off
echo ========================================
echo MOUNT GitHub Upload Script
echo ========================================
echo.

:: 设置仓库信息
set GITHUB_REPO=https://github.com/LiuNingtao/MOUNT.git
set BRANCH_NAME=main

echo [1/6] 检查git状态...
git status
echo.

echo [2/6] 添加远程仓库...
git remote add mount %GITHUB_REPO% 2>nul
git remote set-url mount %GITHUB_REPO%
echo 远程仓库已设置为: %GITHUB_REPO%
echo.

echo [3/6] 添加所有更改...
git add .
echo.

echo [4/6] 提交更改...
set /p COMMIT_MSG=请输入提交信息 (默认: Update MOUNT): 
if "%COMMIT_MSG%"=="" set COMMIT_MSG=Update MOUNT
git commit -m "%COMMIT_MSG%"
echo.

echo [5/6] 推送到GitHub...
echo 正在推送到 %GITHUB_REPO% (%BRANCH_NAME%)
git push -u mount %BRANCH_NAME%
echo.

echo [6/6] 完成！
echo ========================================
echo 上传成功！
echo 访问: %GITHUB_REPO%
echo ========================================
echo.
pause
