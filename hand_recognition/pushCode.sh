#!/bin/bash
# push code

echo "请输入git地址:";
read git_ku
git init
git add ./*
git commit -m "first commit"
git remote add origin1 ${git_ku}
git push origin1 master
rm -rf .git/