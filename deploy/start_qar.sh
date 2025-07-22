#!/bin/bash

# QAR应用启动脚本

set -e

echo "启动QAR应用..."
echo "当前时间: $(date)"
echo "当前用户: $(whoami)"

# 切换到项目目录
cd /home/ubuntu/qar-analysis
echo "工作目录: $(pwd)"

# 检查虚拟环境
if [ ! -f "venv/bin/activate" ]; then
    echo "错误: 虚拟环境不存在"
    exit 1
fi

# 激活虚拟环境
echo "激活虚拟环境..."
source venv/bin/activate

# 检查Python和uvicorn
echo "Python路径: $(which python)"
echo "Python版本: $(python --version)"
echo "uvicorn路径: $(which uvicorn)"

# 设置Python路径
export PYTHONPATH=/home/ubuntu/qar-analysis/src
echo "PYTHONPATH: $PYTHONPATH"

# 测试应用导入
echo "测试应用导入..."
python -c "
import sys
sys.path.insert(0, 'src')
from api.main import app
print('应用导入成功')
"

# 加载环境变量（如果存在）
if [ -f "config/production.env" ]; then
    echo "加载环境变量..."
    set -a
    source config/production.env
    set +a
fi

# 启动应用
echo "启动uvicorn服务器..."
exec uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 2
