#!/bin/bash

set -e

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"


# 配置项
GITHUB_REPO="https://github.com/flagos-ai/FlagGems"
CACHE_DIR="cache"
SOURCE_DIR="tests"   # 例如: subdir/to/copy
TARGET_DIR=$CACHE_DIR/accuracy
PYTHON_SCRIPT="generate_test_from_gems.py" # 例如: process.py

# 创建缓存目录
mkdir -p "$CACHE_DIR"

# Clone 仓库到缓存目录
REPO_NAME=$(basename "$GITHUB_REPO" .git)
REPO_PATH="$CACHE_DIR/$REPO_NAME"

# 如果仓库目录已存在,先删除
if [ -d "$REPO_PATH" ]; then
    echo "Removing existing repository at $REPO_PATH..."
    rm -rf "$REPO_PATH"
fi

echo "Cloning repository to $REPO_PATH..."
git clone "$GITHUB_REPO" "$REPO_PATH"

# 复制指定目录到目标路径
echo "Copying $SOURCE_DIR to $TARGET_DIR..."
mkdir -p "$TARGET_DIR"
cp -r "$REPO_PATH/$SOURCE_DIR"/* "$TARGET_DIR/"

# 运行同级目录下的 Python 脚本
echo "Running Python script..."
python "$SCRIPT_DIR/$PYTHON_SCRIPT" --path "$TARGET_DIR"

# 清理缓存
echo "Cleaning up cache..."
rm -rf "$REPO_PATH"

echo "Done!"

echo "Start testing updated accuracy tests..."

python "script/test_updated_accuracy_ut.py" --name "all" | tee "cache/test_updated_accuracy_ut.log"