#!/usr/bin/env python3
"""
QAR数据分析系统 - 本地开发启动脚本
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    """主函数"""
    print("🚀 启动QAR数据分析系统 (本地开发模式)")
    print("=" * 50)
    
    # 设置项目根目录
    project_root = Path(__file__).parent
    os.chdir(project_root)
    
    # 设置Python路径
    src_path = project_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    
    # 设置环境变量
    os.environ["PYTHONPATH"] = str(src_path)
    os.environ["ENVIRONMENT"] = "development"
    os.environ["DEBUG"] = "True"
    
    print(f"📁 项目目录: {project_root}")
    print(f"🐍 Python路径: {sys.executable}")
    print(f"📦 Python版本: {sys.version}")
    
    # 检查conda环境
    conda_env = os.environ.get("CONDA_DEFAULT_ENV", "未知")
    print(f"🔧 Conda环境: {conda_env}")
    
    # 测试关键模块导入
    print("\n🔍 检查关键模块...")
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__} (CPU: {not torch.cuda.is_available()})")
    except ImportError:
        print("❌ PyTorch 未安装")
    
    try:
        import pandas as pd
        print(f"✅ Pandas: {pd.__version__}")
    except ImportError:
        print("❌ Pandas 未安装")
    
    try:
        import plotly
        print(f"✅ Plotly: {plotly.__version__}")
    except ImportError:
        print("❌ Plotly 未安装")
    
    try:
        import fastapi
        print(f"✅ FastAPI: {fastapi.__version__}")
    except ImportError:
        print("❌ FastAPI 未安装")
    
    # 测试应用导入
    print("\n📱 测试应用导入...")
    try:
        from api.main import app
        print("✅ QAR应用导入成功")
    except Exception as e:
        print(f"❌ QAR应用导入失败: {e}")
        return False
    
    # 启动开发服务器
    print("\n🌐 启动开发服务器...")
    print("访问地址: http://localhost:8000")
    print("API文档: http://localhost:8000/docs")
    print("按 Ctrl+C 停止服务器")
    print("-" * 50)
    
    try:
        # 使用uvicorn启动开发服务器
        import uvicorn
        uvicorn.run(
            "api.main:app",
            host="127.0.0.1",
            port=8000,
            reload=True,  # 开发模式自动重载
            log_level="info",
            access_log=True
        )
    except KeyboardInterrupt:
        print("\n👋 服务器已停止")
    except Exception as e:
        print(f"\n❌ 启动失败: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
