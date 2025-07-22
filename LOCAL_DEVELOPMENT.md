# QAR数据分析系统 - 本地开发指南

## 环境要求

- **Anaconda/Miniconda**: 最新版本
- **Python**: 3.10+
- **操作系统**: Windows/macOS/Linux

## 快速开始

### 1. 创建Conda环境

```bash
# 使用environment.yml创建环境
conda env create -f environment.yml

# 激活环境
conda activate qar-analysis
```

### 2. 验证环境

```bash
# 检查Python版本
python --version

# 检查关键包
python -c "import torch; print(f'PyTorch: {torch.__version__}, CPU: {not torch.cuda.is_available()}')"
python -c "import pandas as pd; print(f'Pandas: {pd.__version__}')"
python -c "import fastapi; print(f'FastAPI: {fastapi.__version__}')"
```

### 3. 启动开发服务器

```bash
# 方法1: 使用启动脚本 (推荐)
python start_local.py

# 方法2: 直接使用uvicorn
uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --reload
```

### 4. 测试系统

```bash
# 在另一个终端运行测试
python test_local.py
```

## 访问地址

- **主页**: http://127.0.0.1:8000/
- **数据分析**: http://127.0.0.1:8000/analysis
- **分析报告**: http://127.0.0.1:8000/reports
- **API文档**: http://127.0.0.1:8000/docs
- **健康检查**: http://127.0.0.1:8000/health

## 开发特性

### 自动重载
开发服务器支持代码自动重载，修改代码后无需重启服务器。

### 调试模式
- 详细的错误信息
- 调试日志输出
- 开发工具集成

### 热重载
修改以下文件会自动重载：
- `src/api/*.py` - API路由和逻辑
- `src/models/*.py` - 数据模型
- `src/visualization/*.py` - 可视化组件
- `templates/*.html` - HTML模板

## 项目结构

```
qar-analysis/
├── environment.yml          # Conda环境配置
├── start_local.py           # 本地启动脚本
├── test_local.py           # 本地测试脚本
├── config/
│   └── development.env     # 开发环境配置
├── src/
│   ├── api/               # FastAPI应用
│   ├── models/            # 数据模型
│   ├── visualization/     # 可视化组件
│   └── utils/             # 工具函数
├── templates/             # HTML模板
├── static/               # 静态文件
├── data/                 # 数据文件
└── logs/                 # 日志文件
```

## 开发工作流

### 1. 代码修改
```bash
# 修改代码文件
vim src/api/main.py

# 服务器自动重载，无需重启
```

### 2. 测试功能
```bash
# 运行测试脚本
python test_local.py

# 或手动测试
curl http://127.0.0.1:8000/health
```

### 3. 查看日志
```bash
# 查看应用日志
tail -f logs/qar_dev.log

# 或在启动脚本中查看实时日志
```

## 常见问题

### Q: Conda环境创建失败
```bash
# 清理conda缓存
conda clean --all

# 更新conda
conda update conda

# 重新创建环境
conda env create -f environment.yml --force
```

### Q: PyTorch安装问题
```bash
# 手动安装CPU版本PyTorch
conda install pytorch torchvision cpuonly -c pytorch
```

### Q: 端口占用
```bash
# 查看端口占用
netstat -ano | findstr :8000  # Windows
lsof -i :8000                 # macOS/Linux

# 修改端口
# 编辑 config/development.env 中的 PORT=8001
```

### Q: 模块导入错误
```bash
# 检查Python路径
python -c "import sys; print('\n'.join(sys.path))"

# 手动设置PYTHONPATH
export PYTHONPATH=$PWD/src:$PYTHONPATH  # Linux/macOS
set PYTHONPATH=%CD%\src;%PYTHONPATH%    # Windows
```

## 性能优化

### 开发模式优化
- 使用单个worker进程
- 启用代码重载
- 详细日志输出
- 禁用生产优化

### 内存使用
- PyTorch CPU版本内存占用较小
- 开发模式下数据集保持较小规模
- 定期清理临时文件

## 部署到生产

### 1. 导出环境
```bash
# 导出当前环境
conda env export > environment_prod.yml

# 或导出requirements.txt
pip list --format=freeze > requirements.txt
```

### 2. 生产配置
```bash
# 复制并修改配置
cp config/development.env config/production.env
# 修改生产环境设置
```

### 3. 性能测试
```bash
# 使用生产配置测试
ENVIRONMENT=production python start_local.py
```

## 贡献指南

1. Fork项目
2. 创建功能分支: `git checkout -b feature/new-feature`
3. 提交更改: `git commit -am 'Add new feature'`
4. 推送分支: `git push origin feature/new-feature`
5. 创建Pull Request

## 支持

如遇问题，请：
1. 查看日志文件: `logs/qar_dev.log`
2. 运行测试脚本: `python test_local.py`
3. 检查环境配置: `conda list`
4. 提交Issue到项目仓库
