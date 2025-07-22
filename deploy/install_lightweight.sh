#!/bin/bash

# QAR数据分析系统 - 轻量级依赖安装脚本
# 避免下载大型CUDA包，节省磁盘空间

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

echo "📦 QAR系统轻量级依赖安装"
echo "================================"

# 检查项目目录
APP_DIR="/home/ubuntu/qar-analysis"
if [ ! -d "$APP_DIR" ]; then
    print_error "项目目录不存在: $APP_DIR"
    exit 1
fi

cd $APP_DIR

# 检查并创建虚拟环境
if [ -d "venv" ]; then
    print_warning "虚拟环境已存在，将重新创建..."
    rm -rf venv
fi

print_status "创建Python虚拟环境..."
if command -v python3.10 &> /dev/null; then
    python3.10 -m venv venv
elif command -v python3 &> /dev/null; then
    python3 -m venv venv
else
    print_error "未找到Python3，请先安装Python"
    exit 1
fi

# 检查虚拟环境是否创建成功
if [ ! -f "venv/bin/activate" ]; then
    print_error "虚拟环境创建失败"
    print_status "尝试手动创建..."

    # 尝试不同的方法创建虚拟环境
    python3 -m venv venv --clear

    if [ ! -f "venv/bin/activate" ]; then
        print_error "虚拟环境创建失败，请检查Python安装"
        exit 1
    fi
fi

print_success "虚拟环境创建成功"

# 激活虚拟环境
print_status "激活虚拟环境..."
source venv/bin/activate

print_status "当前磁盘空间:"
df -h | grep -E "(Filesystem|/dev/)"

echo
print_status "开始安装轻量级依赖..."

# 升级pip
print_status "升级pip..."
pip install --upgrade pip setuptools wheel

# 安装核心Web框架依赖
print_status "安装Web框架依赖..."
pip install \
    fastapi==0.104.1 \
    uvicorn[standard]==0.24.0 \
    jinja2==3.1.2 \
    python-multipart==0.0.6 \
    aiofiles==23.2.1

# 安装数据处理依赖
print_status "安装数据处理依赖..."
pip install \
    pandas==2.1.3 \
    numpy==1.24.3 \
    scipy==1.11.4 \
    scikit-learn==1.3.2

# 安装数据可视化依赖
print_status "安装数据可视化依赖..."
pip install \
    plotly==5.17.0 \
    matplotlib==3.8.2 \
    seaborn==0.13.0 \
    folium==0.15.0

# 安装配置管理依赖
print_status "安装配置管理依赖..."
pip install \
    python-dotenv==1.0.0 \
    pydantic==2.5.0 \
    pydantic-settings==2.1.0

# 安装HTTP客户端
print_status "安装HTTP客户端..."
pip install \
    httpx==0.25.2 \
    requests==2.31.0

# 安装其他必要工具
print_status "安装其他工具..."
pip install \
    python-dateutil==2.8.2 \
    tqdm==4.66.1 \
    psutil==5.9.6 \
    statsmodels==0.14.1 \
    pyproj==3.6.1

# 可选：安装CPU版本的PyTorch (如果需要机器学习功能)
read -p "是否安装CPU版本的PyTorch? (y/n): " INSTALL_TORCH
if [[ "$INSTALL_TORCH" =~ ^[Yy]$ ]]; then
    print_status "安装CPU版本的PyTorch..."
    pip install torch==2.1.1 torchvision==0.16.1 --index-url https://download.pytorch.org/whl/cpu
    print_success "PyTorch CPU版本安装完成"
else
    print_warning "跳过PyTorch安装"
fi

# 可选：安装数据库支持
read -p "是否安装数据库支持? (y/n): " INSTALL_DB
if [[ "$INSTALL_DB" =~ ^[Yy]$ ]]; then
    print_status "安装数据库支持..."
    pip install \
        psycopg2-binary==2.9.9 \
        sqlalchemy==2.0.23
    print_success "数据库支持安装完成"
fi

# 可选：安装Redis支持
read -p "是否安装Redis支持? (y/n): " INSTALL_REDIS
if [[ "$INSTALL_REDIS" =~ ^[Yy]$ ]]; then
    print_status "安装Redis支持..."
    pip install \
        redis==5.0.1 \
        hiredis==2.2.3
    print_success "Redis支持安装完成"
fi

print_success "轻量级依赖安装完成！"

# 显示安装的包
echo
print_status "已安装的包列表:"
pip list | grep -E "(fastapi|uvicorn|pandas|numpy|plotly|scikit-learn)"

# 显示磁盘使用情况
echo
print_status "安装后磁盘空间:"
df -h | grep -E "(Filesystem|/dev/)"

# 显示虚拟环境大小
venv_size=$(du -sh venv | cut -f1)
print_status "虚拟环境大小: $venv_size"

echo
print_success "🎉 轻量级依赖安装完成！"
print_status "相比完整版本，节省了约1-2GB的磁盘空间"

echo
print_status "注意事项:"
echo "1. 此版本不包含GPU支持，仅使用CPU进行计算"
echo "2. 如果需要GPU支持，请安装完整版本的依赖"
echo "3. 机器学习功能可能会比GPU版本慢一些"
echo "4. 所有Web功能和数据可视化功能正常工作"
