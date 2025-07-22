"""
数据处理工具函数
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings
warnings.filterwarnings('ignore')


class QARDataProcessor:
    """QAR数据处理器"""
    
    def __init__(self):
        self.scalers = {}
        
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """数据清洗"""
        df_clean = df.copy()
        
        # 处理缺失值
        numeric_columns = df_clean.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            # 使用前向填充和后向填充
            df_clean[col] = df_clean[col].fillna(method='ffill').fillna(method='bfill')
            
            # 如果仍有缺失值，使用均值填充
            if df_clean[col].isnull().any():
                df_clean[col] = df_clean[col].fillna(df_clean[col].mean())
        
        # 处理异常值（使用IQR方法）
        for col in numeric_columns:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            
            # 定义异常值边界
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # 将异常值替换为边界值
            df_clean[col] = df_clean[col].clip(lower=lower_bound, upper=upper_bound)
        
        return df_clean
    
    def add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加衍生特征"""
        df_enhanced = df.copy()
        
        # 时间相关特征
        if 'TIMESTAMP' in df_enhanced.columns:
            df_enhanced['HOUR'] = pd.to_datetime(df_enhanced['TIMESTAMP']).dt.hour
            df_enhanced['MINUTE'] = pd.to_datetime(df_enhanced['TIMESTAMP']).dt.minute
        
        # 飞行性能特征
        if all(col in df_enhanced.columns for col in ['ALT_STD', 'IAS', 'TAS']):
            # 爬升/下降率
            df_enhanced['CLIMB_RATE'] = df_enhanced['ALT_STD'].diff() / df_enhanced.index.to_series().diff()
            
            # 加速度
            df_enhanced['ACCELERATION'] = df_enhanced['IAS'].diff() / df_enhanced.index.to_series().diff()
            
            # 真空速与指示空速比值
            df_enhanced['TAS_IAS_RATIO'] = df_enhanced['TAS'] / (df_enhanced['IAS'] + 1e-6)
        
        # 发动机效率特征
        if all(col in df_enhanced.columns for col in ['ENG_N1_1', 'ENG_EGT_1', 'ENG_FUEL_FLOW_1']):
            # 燃油效率
            df_enhanced['FUEL_EFFICIENCY'] = df_enhanced['TAS'] / (df_enhanced['ENG_FUEL_FLOW_1'] + 1e-6)
            
            # 发动机温度效率
            df_enhanced['TEMP_EFFICIENCY'] = df_enhanced['ENG_EGT_1'] / (df_enhanced['ENG_N1_1'] + 1e-6)
        
        # 操纵特征
        if all(col in df_enhanced.columns for col in ['ATT_PITCH', 'ATT_ROLL', 'ATT_YAW']):
            # 姿态变化率
            df_enhanced['PITCH_RATE'] = df_enhanced['ATT_PITCH'].diff()
            df_enhanced['ROLL_RATE'] = df_enhanced['ATT_ROLL'].diff()
            df_enhanced['YAW_RATE'] = df_enhanced['ATT_YAW'].diff()
            
            # 姿态稳定性指标
            df_enhanced['ATTITUDE_STABILITY'] = np.sqrt(
                df_enhanced['PITCH_RATE']**2 + 
                df_enhanced['ROLL_RATE']**2 + 
                df_enhanced['YAW_RATE']**2
            )
        
        # 环境适应性特征
        if all(col in df_enhanced.columns for col in ['WIND_SPEED', 'WIND_DIR', 'HDG']):
            # 风向与航向的夹角
            wind_angle_diff = np.abs(df_enhanced['WIND_DIR'] - df_enhanced['HDG'])
            wind_angle_diff = np.minimum(wind_angle_diff, 360 - wind_angle_diff)
            df_enhanced['WIND_ANGLE_DIFF'] = wind_angle_diff
            
            # 逆风/顺风分量
            df_enhanced['HEADWIND_COMPONENT'] = df_enhanced['WIND_SPEED'] * np.cos(np.radians(wind_angle_diff))
            df_enhanced['CROSSWIND_COMPONENT'] = df_enhanced['WIND_SPEED'] * np.sin(np.radians(wind_angle_diff))
        
        return df_enhanced
    
    def normalize_features(self, df: pd.DataFrame, method: str = 'standard') -> pd.DataFrame:
        """特征标准化"""
        df_normalized = df.copy()
        numeric_columns = df_normalized.select_dtypes(include=[np.number]).columns
        
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        else:
            raise ValueError("method must be 'standard' or 'minmax'")
        
        # 标准化数值列
        df_normalized[numeric_columns] = scaler.fit_transform(df_normalized[numeric_columns])
        
        # 保存缩放器
        self.scalers[method] = scaler
        
        return df_normalized
    
    def create_time_windows(self, df: pd.DataFrame, window_size: int = 60) -> List[pd.DataFrame]:
        """创建时间窗口数据"""
        windows = []
        
        for i in range(0, len(df) - window_size + 1, window_size // 2):  # 50%重叠
            window = df.iloc[i:i + window_size].copy()
            windows.append(window)
        
        return windows
    
    def extract_flight_segments(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """提取飞行段数据"""
        segments = {}
        
        for phase in df['FLIGHT_PHASE'].unique():
            segments[phase] = df[df['FLIGHT_PHASE'] == phase].copy()
        
        return segments
    
    def calculate_statistical_features(self, df: pd.DataFrame, 
                                     columns: List[str], 
                                     window_size: int = 100) -> pd.DataFrame:
        """计算统计特征"""
        df_stats = df.copy()
        
        for col in columns:
            if col in df.columns:
                # 滚动统计
                df_stats[f'{col}_MEAN'] = df[col].rolling(window=window_size, min_periods=1).mean()
                df_stats[f'{col}_STD'] = df[col].rolling(window=window_size, min_periods=1).std()
                df_stats[f'{col}_MIN'] = df[col].rolling(window=window_size, min_periods=1).min()
                df_stats[f'{col}_MAX'] = df[col].rolling(window=window_size, min_periods=1).max()
                df_stats[f'{col}_MEDIAN'] = df[col].rolling(window=window_size, min_periods=1).median()
                
                # 趋势特征
                df_stats[f'{col}_TREND'] = df[col].diff().rolling(window=window_size//2, min_periods=1).mean()
                
                # 变异系数
                df_stats[f'{col}_CV'] = df_stats[f'{col}_STD'] / (df_stats[f'{col}_MEAN'] + 1e-6)
        
        return df_stats


class FlightPhaseDetector:
    """飞行阶段检测器"""
    
    def __init__(self):
        self.phase_rules = self._define_phase_rules()
    
    def _define_phase_rules(self) -> Dict:
        """定义飞行阶段规则"""
        return {
            'taxi_out': {
                'altitude': (0, 100),
                'speed': (0, 50),
                'engine_n1': (20, 40)
            },
            'takeoff': {
                'altitude': (0, 2000),
                'speed': (50, 200),
                'engine_n1': (80, 100),
                'vertical_speed': (500, 5000)
            },
            'climb': {
                'altitude': (1000, 40000),
                'vertical_speed': (100, 3000),
                'engine_n1': (70, 95)
            },
            'cruise': {
                'altitude': (25000, 45000),
                'vertical_speed': (-500, 500),
                'engine_n1': (60, 85)
            },
            'descent': {
                'altitude': (1000, 40000),
                'vertical_speed': (-3000, -100),
                'engine_n1': (30, 70)
            },
            'approach': {
                'altitude': (0, 5000),
                'speed': (120, 200),
                'vertical_speed': (-1500, 0)
            },
            'landing': {
                'altitude': (0, 500),
                'speed': (100, 160),
                'vertical_speed': (-1000, 100)
            },
            'taxi_in': {
                'altitude': (0, 100),
                'speed': (0, 50),
                'engine_n1': (20, 40)
            }
        }
    
    def detect_phases(self, df: pd.DataFrame) -> pd.Series:
        """基于规则检测飞行阶段"""
        phases = pd.Series(['unknown'] * len(df), index=df.index)
        
        for i, row in df.iterrows():
            altitude = row.get('ALT_STD', 0)
            speed = row.get('IAS', 0)
            vs = row.get('VS', 0)
            engine_n1 = row.get('ENG_N1_1', 0)
            
            # 按优先级检查阶段
            for phase, rules in self.phase_rules.items():
                match = True
                
                if 'altitude' in rules:
                    alt_min, alt_max = rules['altitude']
                    if not (alt_min <= altitude <= alt_max):
                        match = False
                
                if 'speed' in rules and match:
                    spd_min, spd_max = rules['speed']
                    if not (spd_min <= speed <= spd_max):
                        match = False
                
                if 'vertical_speed' in rules and match:
                    vs_min, vs_max = rules['vertical_speed']
                    if not (vs_min <= vs <= vs_max):
                        match = False
                
                if 'engine_n1' in rules and match:
                    n1_min, n1_max = rules['engine_n1']
                    if not (n1_min <= engine_n1 <= n1_max):
                        match = False
                
                if match:
                    phases.iloc[i] = phase
                    break
        
        # 后处理：平滑阶段转换
        phases = self._smooth_phase_transitions(phases)
        
        return phases
    
    def _smooth_phase_transitions(self, phases: pd.Series, min_duration: int = 30) -> pd.Series:
        """平滑阶段转换"""
        smoothed_phases = phases.copy()
        
        # 移除过短的阶段
        current_phase = phases.iloc[0]
        phase_start = 0
        
        for i in range(1, len(phases)):
            if phases.iloc[i] != current_phase:
                # 检查当前阶段持续时间
                if i - phase_start < min_duration:
                    # 如果太短，用前一个阶段填充
                    if phase_start > 0:
                        smoothed_phases.iloc[phase_start:i] = smoothed_phases.iloc[phase_start - 1]
                
                current_phase = phases.iloc[i]
                phase_start = i
        
        return smoothed_phases
