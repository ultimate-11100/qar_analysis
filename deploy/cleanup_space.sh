#!/bin/bash

# QAR数据分析系统 - 磁盘空间清理脚本

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

echo "🧹 QAR系统磁盘空间清理"
echo "================================"

# 1. 显示当前磁盘使用情况
print_status "当前磁盘使用情况:"
df -h | grep -E "(Filesystem|/dev/)"

echo
print_status "开始清理..."

# 2. 清理pip缓存
print_status "清理pip缓存..."
pip cache purge
print_success "pip缓存已清理"

# 3. 清理apt缓存
print_status "清理apt包缓存..."
sudo apt clean
sudo apt autoclean
sudo apt autoremove -y
print_success "apt缓存已清理"

# 4. 清理系统日志
print_status "清理系统日志..."
sudo journalctl --vacuum-time=7d
sudo find /var/log -name "*.log" -type f -mtime +7 -delete 2>/dev/null || true
print_success "系统日志已清理"

# 5. 清理临时文件
print_status "清理临时文件..."
sudo rm -rf /tmp/*
sudo rm -rf /var/tmp/*
print_success "临时文件已清理"

# 6. 清理Python缓存
print_status "清理Python缓存..."
find /var/www/qar-analysis -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
find /var/www/qar-analysis -name "*.pyc" -delete 2>/dev/null || true
print_success "Python缓存已清理"

# 7. 清理虚拟环境中的缓存
if [ -d "/var/www/qar-analysis/venv" ]; then
    print_status "清理虚拟环境缓存..."
    find /var/www/qar-analysis/venv -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
    find /var/www/qar-analysis/venv -name "*.pyc" -delete 2>/dev/null || true
    print_success "虚拟环境缓存已清理"
fi

# 8. 清理Docker (如果存在)
if command -v docker &> /dev/null; then
    print_status "清理Docker缓存..."
    sudo docker system prune -f 2>/dev/null || true
    print_success "Docker缓存已清理"
fi

# 9. 清理snap包缓存 (如果存在)
if command -v snap &> /dev/null; then
    print_status "清理snap包缓存..."
    sudo snap list --all | awk '/disabled/{print $1, $3}' | while read snapname revision; do
        sudo snap remove "$snapname" --revision="$revision" 2>/dev/null || true
    done
    print_success "snap包缓存已清理"
fi

# 10. 显示清理后的磁盘使用情况
echo
print_status "清理后磁盘使用情况:"
df -h | grep -E "(Filesystem|/dev/)"

# 11. 显示最大的文件和目录
echo
print_status "当前最大的文件和目录:"
echo "最大的10个目录:"
sudo du -h /var/www /home /opt /usr/local 2>/dev/null | sort -hr | head -10

echo
echo "最大的10个文件:"
sudo find /var/www /home /opt /usr/local -type f -size +100M 2>/dev/null | xargs ls -lh | sort -k5 -hr | head -10

echo
print_success "磁盘空间清理完成！"

# 12. 给出建议
echo
print_status "空间优化建议:"
echo "1. 如果不需要GPU支持，使用CPU版本的依赖包"
echo "2. 定期清理日志文件和缓存"
echo "3. 考虑将大文件移动到外部存储"
echo "4. 使用 'ncdu /' 命令查看详细的磁盘使用情况"
