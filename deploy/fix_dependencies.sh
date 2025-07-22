#!/bin/bash

# QAR系统依赖修复脚本

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

APP_DIR="/home/ubuntu/qar-analysis"

echo "📦 修复QAR系统Python依赖"
echo "================================"

# 1. 检查项目目录
if [ ! -d "$APP_DIR" ]; then
    print_error "项目目录不存在: $APP_DIR"
    exit 1
fi

cd $APP_DIR

# 2. 检查虚拟环境
if [ ! -f "venv/bin/activate" ]; then
    print_error "虚拟环境不存在，请先运行 install_lightweight.sh"
    exit 1
fi

# 3. 激活虚拟环境
print_status "激活虚拟环境..."
source venv/bin/activate

# 4. 升级pip
print_status "升级pip..."
pip install --upgrade pip

# 5. 检查并安装缺失的核心依赖
print_status "检查并安装核心依赖..."

core_packages=(
    "fastapi==0.104.1"
    "uvicorn[standard]==0.24.0"
    "jinja2==3.1.2"
    "python-multipart==0.0.6"
    "aiofiles==23.2.1"
)

for package in "${core_packages[@]}"; do
    package_name=$(echo $package | cut -d'=' -f1)
    if pip show $package_name > /dev/null 2>&1; then
        print_success "✓ $package_name 已安装"
    else
        print_warning "安装 $package..."
        pip install $package
    fi
done

# 6. 检查并安装数据处理依赖
print_status "检查并安装数据处理依赖..."

data_packages=(
    "pandas==2.1.3"
    "numpy==1.24.3"
    "scipy==1.11.4"
    "scikit-learn==1.3.2"
)

for package in "${data_packages[@]}"; do
    package_name=$(echo $package | cut -d'=' -f1)
    if pip show $package_name > /dev/null 2>&1; then
        print_success "✓ $package_name 已安装"
    else
        print_warning "安装 $package..."
        pip install $package
    fi
done

# 7. 检查并安装可视化依赖
print_status "检查并安装可视化依赖..."

viz_packages=(
    "plotly==5.17.0"
    "matplotlib==3.8.2"
    "seaborn==0.13.0"
    "folium==0.15.0"
)

for package in "${viz_packages[@]}"; do
    package_name=$(echo $package | cut -d'=' -f1)
    if pip show $package_name > /dev/null 2>&1; then
        print_success "✓ $package_name 已安装"
    else
        print_warning "安装 $package..."
        pip install $package
    fi
done

# 8. 检查并安装其他必要依赖
print_status "检查并安装其他依赖..."

other_packages=(
    "python-dotenv==1.0.0"
    "pydantic==2.5.0"
    "pydantic-settings==2.1.0"
    "httpx==0.25.2"
    "requests==2.31.0"
    "python-dateutil==2.8.2"
    "tqdm==4.66.1"
    "psutil==5.9.6"
    "statsmodels==0.14.1"
    "pyproj==3.6.1"
)

for package in "${other_packages[@]}"; do
    package_name=$(echo $package | cut -d'=' -f1)
    if pip show $package_name > /dev/null 2>&1; then
        print_success "✓ $package_name 已安装"
    else
        print_warning "安装 $package..."
        pip install $package
    fi
done

# 9. 测试应用导入
print_status "测试应用模块导入..."
python -c "
import sys
sys.path.insert(0, 'src')
try:
    from api.main import app
    print('✅ 应用模块导入成功')
except ImportError as e:
    print(f'❌ 应用模块导入失败: {e}')
    exit(1)
except Exception as e:
    print(f'⚠️ 应用模块导入警告: {e}')
"

# 10. 测试关键模块导入
print_status "测试关键模块导入..."
modules_to_test=(
    "pandas"
    "numpy"
    "plotly"
    "folium"
    "fastapi"
    "uvicorn"
    "sklearn"
    "statsmodels"
)

for module in "${modules_to_test[@]}"; do
    if python -c "import $module" 2>/dev/null; then
        print_success "✓ $module"
    else
        print_error "✗ $module"
    fi
done

# 11. 显示已安装的包
print_status "已安装的关键包:"
pip list | grep -E "(fastapi|uvicorn|pandas|numpy|plotly|folium|scikit-learn)"

# 12. 测试uvicorn启动
print_status "测试uvicorn启动..."
timeout 5s uvicorn src.api.main:app --host 127.0.0.1 --port 8003 &
sleep 2

if netstat -tuln | grep -q ":8003"; then
    print_success "✅ uvicorn测试启动成功"
    pkill -f "uvicorn.*8003" || true
else
    print_warning "⚠️ uvicorn测试启动可能有问题"
fi

deactivate

echo
echo "================================"
print_success "依赖修复完成！"

print_status "现在可以尝试启动服务:"
echo "sudo systemctl restart qar-analysis"
echo "sudo systemctl status qar-analysis"
