#!/bin/bash

# Nginx配置修复脚本

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

# 检查是否为root用户
if [[ $EUID -ne 0 ]]; then
   print_error "此脚本需要root权限运行"
   print_status "请使用: sudo bash fix_nginx.sh"
   exit 1
fi

echo "🔧 修复Nginx配置"
echo "================================"

# 1. 检查当前Nginx配置错误
print_status "检查当前Nginx配置..."
if nginx -t 2>/dev/null; then
    print_success "Nginx配置语法正确"
else
    print_error "Nginx配置语法错误，详细信息:"
    nginx -t
fi

# 2. 备份现有配置
print_status "备份现有配置..."
if [ -f "/etc/nginx/sites-available/qar-analysis" ]; then
    cp /etc/nginx/sites-available/qar-analysis /etc/nginx/sites-available/qar-analysis.backup.$(date +%Y%m%d_%H%M%S)
    print_success "配置已备份"
fi

# 3. 创建新的Nginx配置文件
print_status "创建新的Nginx配置文件..."
cat > /etc/nginx/sites-available/qar-analysis << 'EOF'
# QAR数据分析系统 Nginx配置
upstream qar_backend {
    server 127.0.0.1:8000;
    keepalive 32;
}

server {
    listen 80;
    server_name qar.testpublicly.cn localhost _;
    
    # 日志配置
    access_log /var/log/nginx/qar-analysis.access.log;
    error_log /var/log/nginx/qar-analysis.error.log;
    
    # 客户端配置
    client_max_body_size 100M;
    client_body_timeout 60s;
    client_header_timeout 60s;
    
    # Gzip压缩
    gzip on;
    gzip_vary on;
    gzip_min_length 1024;
    gzip_types
        text/plain
        text/css
        text/xml
        text/javascript
        application/javascript
        application/xml+rss
        application/json
        image/svg+xml;
    
    # 安全头
    add_header X-Frame-Options DENY always;
    add_header X-Content-Type-Options nosniff always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header Referrer-Policy "strict-origin-when-cross-origin" always;
    
    # 静态文件服务
    location /static/ {
        alias /home/ubuntu/qar-analysis/static/;
        expires 1y;
        add_header Cache-Control "public, immutable";
        try_files $uri $uri/ =404;
    }
    
    # 可视化文件服务
    location /visualizations/ {
        alias /home/ubuntu/qar-analysis/static/visualizations/;
        expires 1h;
        add_header Cache-Control "public";
        try_files $uri $uri/ =404;
    }
    
    # 健康检查
    location /health {
        proxy_pass http://qar_backend/health;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        access_log off;
    }
    
    # API和应用代理
    location / {
        proxy_pass http://qar_backend;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_cache_bypass $http_upgrade;
        
        # 超时设置
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
        
        # 缓冲设置
        proxy_buffering on;
        proxy_buffer_size 128k;
        proxy_buffers 4 256k;
        proxy_busy_buffers_size 256k;
    }
    
    # 禁止访问敏感文件
    location ~ /\. {
        deny all;
        access_log off;
        log_not_found off;
    }
    
    location ~ \.(env|ini|conf|log)$ {
        deny all;
        access_log off;
        log_not_found off;
    }
}
EOF

# 4. 测试新配置
print_status "测试新的Nginx配置..."
if nginx -t; then
    print_success "新配置语法正确"
else
    print_error "新配置语法错误"
    exit 1
fi

# 5. 确保软链接正确
print_status "检查软链接..."
if [ ! -L "/etc/nginx/sites-enabled/qar-analysis" ]; then
    ln -s /etc/nginx/sites-available/qar-analysis /etc/nginx/sites-enabled/qar-analysis
    print_success "创建软链接"
else
    print_success "软链接已存在"
fi

# 6. 移除默认配置（如果存在）
if [ -L "/etc/nginx/sites-enabled/default" ]; then
    rm /etc/nginx/sites-enabled/default
    print_warning "已移除默认Nginx配置"
fi

# 7. 创建静态文件目录
print_status "创建静态文件目录..."
sudo -u ubuntu mkdir -p /home/ubuntu/qar-analysis/static/{css,js,images,visualizations}
chown -R ubuntu:ubuntu /home/ubuntu/qar-analysis/static

# 8. 重新加载Nginx
print_status "重新加载Nginx配置..."
if systemctl reload nginx; then
    print_success "Nginx配置重新加载成功"
else
    print_error "Nginx配置重新加载失败"
    systemctl status nginx --no-pager -l
    exit 1
fi

# 9. 检查Nginx状态
print_status "检查Nginx状态..."
if systemctl is-active --quiet nginx; then
    print_success "Nginx服务运行正常"
else
    print_error "Nginx服务未运行"
    systemctl status nginx --no-pager -l
    exit 1
fi

# 10. 测试配置
print_status "测试Nginx配置..."
if curl -s -I http://localhost/ | grep -q "HTTP"; then
    print_success "Nginx HTTP测试成功"
else
    print_warning "Nginx HTTP测试失败，可能是后端服务未启动"
fi

echo
echo "================================"
print_success "Nginx配置修复完成！"

print_status "配置文件位置:"
echo "  /etc/nginx/sites-available/qar-analysis"
echo "  /etc/nginx/sites-enabled/qar-analysis"

print_status "测试命令:"
echo "  sudo nginx -t"
echo "  curl -I http://localhost/"
echo "  curl http://localhost/health"
