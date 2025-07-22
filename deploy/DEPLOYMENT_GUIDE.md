# QAR数据分析系统 Ubuntu + Nginx 部署指南

## 系统要求

- **操作系统**: Ubuntu 20.04 LTS 或 22.04 LTS
- **CPU**: 2核心以上
- **内存**: 4GB以上
- **存储**: 20GB以上
- **Python**: 3.9+
- **域名**: 可选，用于HTTPS配置

## 快速部署 (现有项目)

### 前提条件
- 项目已上传到 `/var/www/qar-analysis`
- 使用现有的 `www-data` 用户运行

### 方法一：一键部署 (推荐)

```bash
# 进入项目目录
cd /var/www/qar-analysis

# 运行一键部署脚本
chmod +x deploy/quick_deploy.sh
bash deploy/quick_deploy.sh
```

### 方法二：分步部署

```bash
# 1. 部署应用
cd /var/www/qar-analysis
chmod +x deploy/deploy_existing.sh
bash deploy/deploy_existing.sh

# 2. 配置服务
chmod +x deploy/configure_existing.sh
sudo bash deploy/configure_existing.sh
```

### 方法三：传统部署 (如需创建新用户)

```bash
# 运行依赖安装脚本
chmod +x deploy/install_dependencies.sh
bash deploy/install_dependencies.sh

# 运行应用部署脚本
chmod +x deploy/deploy_app.sh
bash deploy/deploy_app.sh

# 运行服务配置脚本
chmod +x deploy/configure_services.sh
sudo bash deploy/configure_services.sh
```

### 5. 配置域名和SSL (可选)

```bash
# 修改Nginx配置中的域名
sudo vim /etc/nginx/sites-available/qar-analysis

# 获取SSL证书
sudo certbot --nginx -d your-domain.com -d www.your-domain.com
```

## 详细配置

### 环境变量配置

编辑 `/opt/qar-analysis/config/production.env`:

```env
# 基础配置
ENVIRONMENT=production
DEBUG=False
HOST=0.0.0.0
PORT=8000
LOG_LEVEL=info

# 数据库配置
DATABASE_URL=postgresql://qar_user:qar_password@localhost/qar_db

# Redis配置
REDIS_URL=redis://localhost:6379/0

# 安全配置
SECRET_KEY=your-secret-key-here
ALLOWED_HOSTS=your-domain.com,localhost,127.0.0.1

# 文件路径
DATA_DIR=/opt/qar-analysis/data
LOG_DIR=/var/log/qar-analysis
STATIC_DIR=/opt/qar-analysis/static
TEMPLATES_DIR=/opt/qar-analysis/templates
```

### 数据库配置

```bash
# 创建数据库用户和数据库
sudo -u postgres psql
CREATE USER qar_user WITH PASSWORD 'your-secure-password';
CREATE DATABASE qar_db OWNER qar_user;
GRANT ALL PRIVILEGES ON DATABASE qar_db TO qar_user;
\q
```

### 防火墙配置

```bash
# 配置UFW防火墙
sudo ufw allow 22/tcp      # SSH
sudo ufw allow 80/tcp      # HTTP
sudo ufw allow 443/tcp     # HTTPS
sudo ufw --force enable
```

## 服务管理

### 启动/停止服务

```bash
# QAR应用服务
sudo systemctl start qar-analysis
sudo systemctl stop qar-analysis
sudo systemctl restart qar-analysis
sudo systemctl status qar-analysis

# Nginx服务
sudo systemctl start nginx
sudo systemctl stop nginx
sudo systemctl restart nginx
sudo systemctl status nginx
```

### 查看日志

```bash
# 应用日志
sudo journalctl -u qar-analysis -f

# Nginx访问日志
sudo tail -f /var/log/nginx/qar-analysis.access.log

# Nginx错误日志
sudo tail -f /var/log/nginx/qar-analysis.error.log

# 应用错误日志
sudo tail -f /var/log/qar-analysis/error.log
```

### 监控系统

```bash
# 运行系统监控
sudo /usr/local/bin/qar-monitor.sh

# 查看监控日志
sudo tail -f /var/log/qar-analysis/monitor.log
```

## 性能优化

### Nginx优化

```nginx
# 在 /etc/nginx/nginx.conf 中添加
worker_processes auto;
worker_connections 1024;

# 启用Gzip压缩
gzip on;
gzip_vary on;
gzip_min_length 1024;
gzip_types text/plain text/css application/json application/javascript text/xml application/xml application/xml+rss text/javascript;
```

### 应用优化

```bash
# 增加worker进程数
# 编辑 /etc/systemd/system/qar-analysis.service
ExecStart=/opt/qar-analysis/venv/bin/uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### 数据库优化

```sql
-- PostgreSQL优化配置
-- 编辑 /etc/postgresql/*/main/postgresql.conf
shared_buffers = 256MB
effective_cache_size = 1GB
maintenance_work_mem = 64MB
checkpoint_completion_target = 0.9
wal_buffers = 16MB
default_statistics_target = 100
```

## 备份策略

### 数据库备份

```bash
# 创建备份脚本
cat > /usr/local/bin/backup-qar-db.sh << 'EOF'
#!/bin/bash
BACKUP_DIR="/opt/backups/qar"
DATE=$(date +%Y%m%d_%H%M%S)
mkdir -p $BACKUP_DIR
pg_dump -U qar_user -h localhost qar_db > $BACKUP_DIR/qar_db_$DATE.sql
find $BACKUP_DIR -name "*.sql" -mtime +7 -delete
EOF

chmod +x /usr/local/bin/backup-qar-db.sh

# 添加到crontab
echo "0 2 * * * /usr/local/bin/backup-qar-db.sh" | sudo crontab -
```

### 应用备份

```bash
# 备份应用文件
tar -czf /opt/backups/qar-analysis-$(date +%Y%m%d).tar.gz \
    /opt/qar-analysis \
    --exclude=/opt/qar-analysis/venv \
    --exclude=/opt/qar-analysis/data/raw \
    --exclude=/opt/qar-analysis/logs
```

## 故障排除

### 常见问题

1. **服务启动失败**
   ```bash
   sudo systemctl status qar-analysis
   sudo journalctl -u qar-analysis --no-pager
   ```

2. **Nginx配置错误**
   ```bash
   sudo nginx -t
   sudo systemctl status nginx
   ```

3. **数据库连接失败**
   ```bash
   sudo -u postgres psql -c "\l"
   sudo systemctl status postgresql
   ```

4. **端口占用**
   ```bash
   sudo netstat -tulpn | grep :8000
   sudo lsof -i :8000
   ```

### 性能监控

```bash
# 系统资源监控
htop
iotop
nethogs

# 应用性能监控
curl http://localhost:8000/health
```

## 安全建议

1. **定期更新系统**
   ```bash
   sudo apt update && sudo apt upgrade
   ```

2. **配置fail2ban**
   ```bash
   sudo apt install fail2ban
   sudo systemctl enable fail2ban
   ```

3. **定期备份**
   - 数据库备份
   - 应用文件备份
   - 配置文件备份

4. **监控日志**
   - 定期检查错误日志
   - 监控异常访问
   - 设置告警机制

## 联系支持

如遇到部署问题，请提供以下信息：
- Ubuntu版本: `lsb_release -a`
- Python版本: `python3 --version`
- 错误日志: `sudo journalctl -u qar-analysis --no-pager`
- 系统状态: `sudo /usr/local/bin/qar-monitor.sh`
