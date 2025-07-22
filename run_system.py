"""
QAR数据分析系统启动脚本
提供多种运行模式的便捷入口
"""

import argparse
import os
import sys
import subprocess
from pathlib import Path


def run_demo():
    """运行完整演示"""
    print("🚀 启动QAR数据分析系统完整演示...")
    try:
        subprocess.run([sys.executable, "main.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ 演示运行失败: {e}")
        return False
    return True


def run_web_server():
    """启动Web服务"""
    print("🌐 启动QAR数据分析系统Web服务...")
    api_path = Path("src/api/main.py")
    
    if not api_path.exists():
        print("❌ 找不到API服务文件")
        return False
    
    try:
        # 切换到API目录并启动服务
        os.chdir("src/api")
        subprocess.run([sys.executable, "main.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Web服务启动失败: {e}")
        return False
    except KeyboardInterrupt:
        print("\n🛑 Web服务已停止")
    return True


def run_tests():
    """运行测试套件"""
    print("🧪 运行QAR数据分析系统测试套件...")
    test_path = Path("tests/test_qar_system.py")
    
    if not test_path.exists():
        print("❌ 找不到测试文件")
        return False
    
    try:
        subprocess.run([sys.executable, str(test_path)], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ 测试运行失败: {e}")
        return False
    return True


def install_dependencies():
    """安装依赖包"""
    print("📦 安装QAR数据分析系统依赖包...")
    
    requirements_path = Path("requirements.txt")
    if not requirements_path.exists():
        print("❌ 找不到requirements.txt文件")
        return False
    
    try:
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ], check=True)
        print("✅ 依赖包安装完成")
    except subprocess.CalledProcessError as e:
        print(f"❌ 依赖包安装失败: {e}")
        return False
    return True


def check_environment():
    """检查运行环境"""
    print("🔍 检查运行环境...")
    
    # 检查Python版本
    python_version = sys.version_info
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
        print(f"❌ Python版本过低: {python_version.major}.{python_version.minor}")
        print("   需要Python 3.8或更高版本")
        return False
    
    print(f"✅ Python版本: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # 检查关键文件
    critical_files = [
        "requirements.txt",
        "main.py",
        "src/data_generation/qar_simulator.py",
        "src/visualization/qar_visualizer.py",
        "src/models/safety_models.py",
        "src/models/optimization_models.py",
        "src/api/main.py"
    ]
    
    missing_files = []
    for file_path in critical_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print("❌ 缺少关键文件:")
        for file_path in missing_files:
            print(f"   - {file_path}")
        return False
    
    print("✅ 所有关键文件存在")
    
    # 检查目录结构
    required_dirs = ["src", "data", "tests", "static"]
    for dir_name in required_dirs:
        Path(dir_name).mkdir(exist_ok=True)
    
    print("✅ 目录结构检查完成")
    return True


def show_system_info():
    """显示系统信息"""
    print("=" * 60)
    print("QAR数据分析系统信息")
    print("=" * 60)
    print(f"Python版本: {sys.version}")
    print(f"工作目录: {os.getcwd()}")
    print(f"系统平台: {sys.platform}")
    
    # 检查已安装的关键包
    try:
        import torch
        print(f"PyTorch版本: {torch.__version__}")
    except ImportError:
        print("PyTorch: 未安装")
    
    try:
        import pandas
        print(f"Pandas版本: {pandas.__version__}")
    except ImportError:
        print("Pandas: 未安装")
    
    try:
        import sklearn
        print(f"Scikit-learn版本: {sklearn.__version__}")
    except ImportError:
        print("Scikit-learn: 未安装")
    
    try:
        import fastapi
        print(f"FastAPI版本: {fastapi.__version__}")
    except ImportError:
        print("FastAPI: 未安装")
    
    print("=" * 60)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="QAR数据分析系统启动脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python run_system.py --demo          # 运行完整演示
  python run_system.py --web           # 启动Web服务
  python run_system.py --test          # 运行测试
  python run_system.py --install       # 安装依赖
  python run_system.py --check         # 检查环境
  python run_system.py --info          # 显示系统信息
        """
    )
    
    parser.add_argument("--demo", action="store_true", help="运行完整演示")
    parser.add_argument("--web", action="store_true", help="启动Web服务")
    parser.add_argument("--test", action="store_true", help="运行测试套件")
    parser.add_argument("--install", action="store_true", help="安装依赖包")
    parser.add_argument("--check", action="store_true", help="检查运行环境")
    parser.add_argument("--info", action="store_true", help="显示系统信息")
    
    args = parser.parse_args()
    
    # 如果没有参数，显示帮助
    if not any(vars(args).values()):
        parser.print_help()
        return
    
    success = True
    
    if args.info:
        show_system_info()
    
    if args.check:
        success &= check_environment()
    
    if args.install:
        success &= install_dependencies()
    
    if args.test:
        success &= run_tests()
    
    if args.demo:
        success &= run_demo()
    
    if args.web:
        success &= run_web_server()
    
    if success:
        print("\n✅ 所有操作完成成功")
    else:
        print("\n❌ 部分操作失败")
        sys.exit(1)


if __name__ == "__main__":
    main()
