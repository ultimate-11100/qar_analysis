"""
QAR数据分析系统配置文件
"""

import os
from pathlib import Path

# 基础配置
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
STATIC_DIR = BASE_DIR / "static"
MODELS_DIR = DATA_DIR / "models"
VISUALIZATIONS_DIR = STATIC_DIR / "visualizations"

# 数据配置
DEFAULT_FLIGHT_DURATION = 120  # 分钟
DEFAULT_SAMPLING_RATE = 1.0    # Hz
DEFAULT_ANOMALY_RATE = 0.05    # 5%异常率

# 模型配置
ANOMALY_DETECTION_CONFIG = {
    'contamination': 0.1,
    'n_estimators': 100,
    'random_state': 42
}

FUEL_OPTIMIZATION_CONFIG = {
    'n_estimators': 100,
    'learning_rate': 0.1,
    'max_depth': 6,
    'random_state': 42
}

MAINTENANCE_PREDICTION_CONFIG = {
    'n_estimators': 100,
    'random_state': 42
}

# 可视化配置
VISUALIZATION_CONFIG = {
    'style': 'seaborn-v0_8',
    'figure_size': (12, 8),
    'dpi': 300,
    'color_palette': 'husl'
}

# Web服务配置
WEB_CONFIG = {
    'host': '0.0.0.0',
    'port': 8000,
    'debug': True,
    'reload': True
}

# 安全阈值配置
SAFETY_THRESHOLDS = {
    'altitude_deviation': 500,    # ft
    'speed_deviation': 20,        # kt
    'engine_temp_high': 800,      # °C
    'fuel_flow_high': 3000,       # kg/h
    'pitch_extreme': 15,          # degrees
    'roll_extreme': 30,           # degrees
    'vertical_speed_extreme': 3000 # ft/min
}

# QAR参数配置
QAR_PARAMETERS = {
    # 姿态与运动参数
    'attitude': [
        'ATT_PITCH', 'ATT_ROLL', 'ATT_YAW', 
        'ATT_FPA', 'ATT_SIDESLIP'
    ],
    
    # 操纵面位置
    'control_surfaces': [
        'CTRL_AIL', 'CTRL_ELEV', 'CTRL_RUDDER', 'CTRL_SPOILER'
    ],
    
    # 飞行控制输入
    'flight_controls': [
        'CTRL_COLUMN', 'CTRL_PEDAL', 'AP_STATUS'
    ],
    
    # 发动机参数
    'engine': [
        'ENG_N1_1', 'ENG_N2_1', 'ENG_THRUST_1', 'ENG_N1_2', 'ENG_N2_2',
        'ENG_EGT_1', 'ENG_OIL_P_1', 'ENG_OIL_T_1', 
        'ENG_FUEL_FLOW_1', 'ENG_VIB_1'
    ],
    
    # 气动参数
    'aerodynamic': [
        'IAS', 'TAS', 'ALT_STD', 'ALT_GND', 'VS', 'FPA'
    ],
    
    # 导航参数
    'navigation': [
        'HDG', 'GS', 'LAT', 'LON', 'GPS_STATUS', 'IRS_HEADING'
    ],
    
    # 燃油参数
    'fuel': [
        'FUEL_QTY_1', 'FUEL_FLOW'
    ],
    
    # 环境参数
    'environment': [
        'WIND_SPEED', 'WIND_DIR', 'TURB_INT'
    ],
    
    # 系统事件
    'events': [
        'TOFF_FLAG', 'LAND_FLAG', 'ENG_FAIL', 'XPDR_CODE', 'TCAS_TA'
    ]
}

# 飞行阶段配置
FLIGHT_PHASES = {
    'taxi_out': {'color': '#FFA500', 'description': '滑行出港'},
    'takeoff': {'color': '#FF4500', 'description': '起飞'},
    'climb': {'color': '#32CD32', 'description': '爬升'},
    'cruise': {'color': '#4169E1', 'description': '巡航'},
    'descent': {'color': '#FFD700', 'description': '下降'},
    'approach': {'color': '#FF6347', 'description': '进近'},
    'landing': {'color': '#DC143C', 'description': '着陆'},
    'taxi_in': {'color': '#FFA500', 'description': '滑行进港'}
}

# 数据处理配置
DATA_PROCESSING_CONFIG = {
    'window_size': 100,
    'overlap_ratio': 0.5,
    'outlier_method': 'iqr',
    'outlier_factor': 1.5,
    'missing_value_method': 'forward_fill'
}

# 日志配置
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': 'qar_analysis.log'
}

# 创建必要的目录
def create_directories():
    """创建必要的目录结构"""
    directories = [
        DATA_DIR / "raw",
        DATA_DIR / "processed", 
        MODELS_DIR,
        VISUALIZATIONS_DIR,
        BASE_DIR / "logs"
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

# 获取配置函数
def get_config(section=None):
    """获取配置信息"""
    config = {
        'base_dir': BASE_DIR,
        'data_dir': DATA_DIR,
        'models_dir': MODELS_DIR,
        'visualizations_dir': VISUALIZATIONS_DIR,
        'flight_duration': DEFAULT_FLIGHT_DURATION,
        'sampling_rate': DEFAULT_SAMPLING_RATE,
        'anomaly_rate': DEFAULT_ANOMALY_RATE,
        'anomaly_detection': ANOMALY_DETECTION_CONFIG,
        'fuel_optimization': FUEL_OPTIMIZATION_CONFIG,
        'maintenance_prediction': MAINTENANCE_PREDICTION_CONFIG,
        'visualization': VISUALIZATION_CONFIG,
        'web': WEB_CONFIG,
        'safety_thresholds': SAFETY_THRESHOLDS,
        'qar_parameters': QAR_PARAMETERS,
        'flight_phases': FLIGHT_PHASES,
        'data_processing': DATA_PROCESSING_CONFIG,
        'logging': LOGGING_CONFIG
    }
    
    if section:
        return config.get(section, {})
    return config

if __name__ == "__main__":
    # 创建目录结构
    create_directories()
    print("配置初始化完成，目录结构已创建")
