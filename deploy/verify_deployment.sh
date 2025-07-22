#!/bin/bash

# QAR数据分析系统 - 部署验证脚本

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

echo "🔍 QAR数据分析系统部署验证"
echo "================================"

# 1. 检查项目文件
print_status "检查项目文件结构..."
APP_DIR="/home/ubuntu/qar-analysis"

if [ ! -d "$APP_DIR" ]; then
    print_error "项目目录不存在: $APP_DIR"
    exit 1
fi

required_files=(
    "src/api/main.py"
    "templates/base.html"
    "requirements.txt"
    "venv/bin/activate"
    "config/production.env"
)

missing_files=()
for file in "${required_files[@]}"; do
    if [ -f "$APP_DIR/$file" ]; then
        print_success "✓ $file"
    else
        print_error "✗ $file"
        missing_files+=("$file")
    fi
done

if [ ${#missing_files[@]} -gt 0 ]; then
    print_error "缺少必要文件，请重新部署"
    exit 1
fi

# 2. 检查Python环境
print_status "检查Python虚拟环境..."
cd $APP_DIR
source venv/bin/activate

python_version=$(python --version 2>&1)
print_success "Python版本: $python_version"

# 检查关键包
packages=("fastapi" "uvicorn" "pandas" "numpy" "plotly")
for package in "${packages[@]}"; do
    if pip show $package > /dev/null 2>&1; then
        version=$(pip show $package | grep Version | cut -d' ' -f2)
        print_success "✓ $package ($version)"
    else
        print_error "✗ $package 未安装"
    fi
done

# 3. 检查系统服务
print_status "检查系统服务状态..."
services=("qar-analysis" "nginx")

for service in "${services[@]}"; do
    if systemctl is-active --quiet $service; then
        print_success "✓ $service 服务运行中"
    else
        print_error "✗ $service 服务未运行"
        sudo systemctl status $service --no-pager
    fi
done

# 4. 检查端口监听
print_status "检查端口监听状态..."
ports=("8000:应用端口" "80:HTTP端口")

for port_info in "${ports[@]}"; do
    port=$(echo $port_info | cut -d':' -f1)
    desc=$(echo $port_info | cut -d':' -f2)
    
    if netstat -tuln | grep -q ":$port "; then
        print_success "✓ $desc ($port) 正常监听"
    else
        print_error "✗ $desc ($port) 未监听"
    fi
done

# 5. 检查文件权限
print_status "检查文件权限..."
if [ "$(stat -c %U $APP_DIR)" = "ubuntu" ]; then
    print_success "✓ 项目目录权限正确 (ubuntu)"
else
    print_warning "⚠ 项目目录权限可能不正确"
fi

if [ -d "/var/log/qar-analysis" ]; then
    if [ "$(stat -c %U /var/log/qar-analysis)" = "ubuntu" ]; then
        print_success "✓ 日志目录权限正确 (ubuntu)"
    else
        print_warning "⚠ 日志目录权限可能不正确"
    fi
else
    print_error "✗ 日志目录不存在"
fi

# 6. 测试HTTP访问
print_status "测试HTTP访问..."

# 测试健康检查端点
if curl -s http://localhost:8000/health > /dev/null; then
    print_success "✓ 应用健康检查通过"
else
    print_error "✗ 应用健康检查失败"
fi

# 测试Nginx代理
if curl -s http://localhost/ > /dev/null; then
    print_success "✓ Nginx代理访问正常"
else
    print_error "✗ Nginx代理访问失败"
fi

# 测试API端点
if curl -s http://localhost/docs > /dev/null; then
    print_success "✓ API文档页面可访问"
else
    print_warning "⚠ API文档页面访问失败"
fi

# 7. 检查配置文件
print_status "检查配置文件..."

# Nginx配置
if nginx -t > /dev/null 2>&1; then
    print_success "✓ Nginx配置语法正确"
else
    print_error "✗ Nginx配置语法错误"
fi

# Systemd配置
if systemctl is-enabled qar-analysis > /dev/null 2>&1; then
    print_success "✓ QAR服务已启用自启动"
else
    print_warning "⚠ QAR服务未启用自启动"
fi

# 8. 检查日志
print_status "检查日志文件..."
log_files=(
    "/var/log/qar-analysis/monitor.log"
    "/var/log/nginx/qar-analysis.access.log"
    "/var/log/nginx/qar-analysis.error.log"
)

for log_file in "${log_files[@]}"; do
    if [ -f "$log_file" ]; then
        size=$(du -h "$log_file" | cut -f1)
        print_success "✓ $log_file ($size)"
    else
        print_warning "⚠ $log_file 不存在"
    fi
done

# 9. 系统资源检查
print_status "检查系统资源..."

# 磁盘空间
disk_usage=$(df -h /var | tail -1 | awk '{print $5}' | sed 's/%//')
if [ "$disk_usage" -lt 80 ]; then
    print_success "✓ 磁盘空间充足 ($disk_usage%)"
else
    print_warning "⚠ 磁盘空间不足 ($disk_usage%)"
fi

# 内存使用
mem_usage=$(free | grep Mem | awk '{printf "%.0f", $3/$2 * 100.0}')
if [ "$mem_usage" -lt 80 ]; then
    print_success "✓ 内存使用正常 ($mem_usage%)"
else
    print_warning "⚠ 内存使用较高 ($mem_usage%)"
fi

# 10. 功能测试
print_status "进行功能测试..."

# 测试数据生成API
test_data='{"flight_duration_minutes": 5, "sampling_rate_hz": 1.0, "anomaly_rate": 0.05}'
if curl -s -X POST -H "Content-Type: application/json" -d "$test_data" http://localhost/api/simulate > /dev/null; then
    print_success "✓ 数据生成API测试通过"
else
    print_warning "⚠ 数据生成API测试失败"
fi

# 测试数据摘要API
if curl -s http://localhost/api/data/summary > /dev/null; then
    print_success "✓ 数据摘要API测试通过"
else
    print_warning "⚠ 数据摘要API测试失败"
fi

echo
echo "================================"
print_status "验证完成！"

# 总结
echo
echo "部署验证总结:"
echo "- 项目路径: $APP_DIR"
echo "- 运行用户: www-data"
echo "- 访问地址: http://localhost"
echo "- 健康检查: http://localhost/health"
echo "- API文档: http://localhost/docs"

echo
echo "常用命令:"
echo "- 查看服务状态: sudo systemctl status qar-analysis"
echo "- 查看应用日志: sudo journalctl -u qar-analysis -f"
echo "- 重启应用: sudo systemctl restart qar-analysis"
echo "- 系统监控: sudo /usr/local/bin/qar-monitor.sh"

if [ -f "/usr/local/bin/qar-monitor.sh" ]; then
    echo
    print_status "运行系统监控检查..."
    sudo /usr/local/bin/qar-monitor.sh
fi
