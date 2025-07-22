"""
QAR数据模拟器
模拟机载QAR数据集，包含飞行操纵、发动机性能、气动性能等参数
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import random


class QARDataSimulator:
    """QAR数据模拟器"""
    
    def __init__(self, flight_duration_minutes: int = 120, sampling_rate_hz: float = 1.0):
        """
        初始化QAR数据模拟器
        
        Args:
            flight_duration_minutes: 飞行时长（分钟）
            sampling_rate_hz: 采样频率（Hz）
        """
        self.flight_duration = flight_duration_minutes
        self.sampling_rate = sampling_rate_hz
        self.total_samples = int(flight_duration_minutes * 60 * sampling_rate_hz)
        self.time_vector = np.linspace(0, flight_duration_minutes * 60, self.total_samples)
        
        # 飞行阶段定义
        self.flight_phases = self._define_flight_phases()
        
    def _define_flight_phases(self) -> Dict[str, Tuple[float, float]]:
        """定义飞行阶段时间段"""
        total_time = self.flight_duration * 60
        return {
            'taxi_out': (0, total_time * 0.05),
            'takeoff': (total_time * 0.05, total_time * 0.08),
            'climb': (total_time * 0.08, total_time * 0.25),
            'cruise': (total_time * 0.25, total_time * 0.75),
            'descent': (total_time * 0.75, total_time * 0.92),
            'approach': (total_time * 0.92, total_time * 0.97),
            'landing': (total_time * 0.97, total_time * 0.99),
            'taxi_in': (total_time * 0.99, total_time)
        }
    
    def _get_flight_phase(self, time: float) -> str:
        """根据时间获取飞行阶段"""
        for phase, (start, end) in self.flight_phases.items():
            if start <= time < end:
                return phase
        return 'cruise'
    
    def _generate_altitude_profile(self) -> np.ndarray:
        """生成高度剖面"""
        altitude = np.zeros(self.total_samples)
        
        for i, t in enumerate(self.time_vector):
            phase = self._get_flight_phase(t)
            
            if phase == 'taxi_out':
                altitude[i] = 0 + np.random.normal(0, 5)
            elif phase == 'takeoff':
                # 起飞阶段快速爬升
                progress = (t - self.flight_phases['takeoff'][0]) / (
                    self.flight_phases['takeoff'][1] - self.flight_phases['takeoff'][0])
                altitude[i] = 1500 * progress + np.random.normal(0, 50)
            elif phase == 'climb':
                # 爬升阶段
                progress = (t - self.flight_phases['climb'][0]) / (
                    self.flight_phases['climb'][1] - self.flight_phases['climb'][0])
                altitude[i] = 1500 + 35000 * progress + np.random.normal(0, 100)
            elif phase == 'cruise':
                altitude[i] = 36000 + np.random.normal(0, 200)
            elif phase == 'descent':
                # 下降阶段
                progress = (t - self.flight_phases['descent'][0]) / (
                    self.flight_phases['descent'][1] - self.flight_phases['descent'][0])
                altitude[i] = 36000 * (1 - progress) + 1500 * progress + np.random.normal(0, 100)
            elif phase == 'approach':
                progress = (t - self.flight_phases['approach'][0]) / (
                    self.flight_phases['approach'][1] - self.flight_phases['approach'][0])
                altitude[i] = 1500 * (1 - progress) + np.random.normal(0, 50)
            else:  # landing, taxi_in
                altitude[i] = 0 + np.random.normal(0, 5)
                
        return np.maximum(altitude, 0)  # 确保高度不为负
    
    def _generate_airspeed_profile(self, altitude: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """生成空速剖面"""
        ias = np.zeros(self.total_samples)  # 指示空速
        tas = np.zeros(self.total_samples)  # 真空速
        
        for i, t in enumerate(self.time_vector):
            phase = self._get_flight_phase(t)
            
            if phase == 'taxi_out' or phase == 'taxi_in':
                ias[i] = np.random.uniform(0, 30)
            elif phase == 'takeoff':
                progress = (t - self.flight_phases['takeoff'][0]) / (
                    self.flight_phases['takeoff'][1] - self.flight_phases['takeoff'][0])
                ias[i] = 80 + 80 * progress + np.random.normal(0, 5)
            elif phase == 'climb':
                ias[i] = np.random.uniform(250, 280)
            elif phase == 'cruise':
                ias[i] = np.random.uniform(280, 320)
            elif phase == 'descent':
                ias[i] = np.random.uniform(250, 300)
            elif phase == 'approach':
                progress = (t - self.flight_phases['approach'][0]) / (
                    self.flight_phases['approach'][1] - self.flight_phases['approach'][0])
                ias[i] = 180 * (1 - progress) + 140 * progress + np.random.normal(0, 5)
            elif phase == 'landing':
                ias[i] = np.random.uniform(130, 150)
            
            # 计算真空速（简化模型）
            density_ratio = np.exp(-altitude[i] / 30000)  # 简化的大气密度模型
            tas[i] = ias[i] / np.sqrt(density_ratio)
            
        return ias, tas
    
    def _generate_engine_parameters(self, altitude: np.ndarray, ias: np.ndarray) -> Dict[str, np.ndarray]:
        """生成发动机参数"""
        engine_params = {}
        
        for i, t in enumerate(self.time_vector):
            phase = self._get_flight_phase(t)
            
            # 根据飞行阶段设置基础推力需求
            if phase == 'taxi_out' or phase == 'taxi_in':
                base_n1 = np.random.uniform(20, 30)
            elif phase == 'takeoff':
                base_n1 = np.random.uniform(95, 100)
            elif phase == 'climb':
                base_n1 = np.random.uniform(85, 95)
            elif phase == 'cruise':
                base_n1 = np.random.uniform(75, 85)
            elif phase == 'descent':
                base_n1 = np.random.uniform(40, 60)
            elif phase == 'approach':
                base_n1 = np.random.uniform(60, 75)
            elif phase == 'landing':
                base_n1 = np.random.uniform(40, 50)
            else:
                base_n1 = 50
            
            # 发动机1参数
            n1_1 = base_n1 + np.random.normal(0, 2)
            n2_1 = n1_1 * 1.2 + np.random.normal(0, 3)
            
            # 发动机2参数（略有差异）
            n1_2 = base_n1 + np.random.normal(0, 2)
            n2_2 = n1_2 * 1.2 + np.random.normal(0, 3)
            
            if i == 0:
                engine_params['ENG_N1_1'] = [n1_1]
                engine_params['ENG_N2_1'] = [n2_1]
                engine_params['ENG_N1_2'] = [n1_2]
                engine_params['ENG_N2_2'] = [n2_2]
                
                # 推力估算
                engine_params['ENG_THRUST_1'] = [n1_1 * 2.5 + np.random.normal(0, 5)]
                
                # 温度参数
                engine_params['ENG_EGT_1'] = [400 + n1_1 * 6 + np.random.normal(0, 10)]
                engine_params['ENG_OIL_T_1'] = [80 + n1_1 * 0.5 + np.random.normal(0, 5)]
                
                # 压力参数
                engine_params['ENG_OIL_P_1'] = [40 + n1_1 * 0.3 + np.random.normal(0, 2)]
                
                # 燃油流量
                engine_params['ENG_FUEL_FLOW_1'] = [n1_1 * 25 + np.random.normal(0, 50)]
                
                # 振动
                engine_params['ENG_VIB_1'] = [np.random.uniform(0.5, 2.0)]
            else:
                engine_params['ENG_N1_1'].append(n1_1)
                engine_params['ENG_N2_1'].append(n2_1)
                engine_params['ENG_N1_2'].append(n1_2)
                engine_params['ENG_N2_2'].append(n2_2)
                engine_params['ENG_THRUST_1'].append(n1_1 * 2.5 + np.random.normal(0, 5))
                engine_params['ENG_EGT_1'].append(400 + n1_1 * 6 + np.random.normal(0, 10))
                engine_params['ENG_OIL_T_1'].append(80 + n1_1 * 0.5 + np.random.normal(0, 5))
                engine_params['ENG_OIL_P_1'].append(40 + n1_1 * 0.3 + np.random.normal(0, 2))
                engine_params['ENG_FUEL_FLOW_1'].append(n1_1 * 25 + np.random.normal(0, 50))
                engine_params['ENG_VIB_1'].append(np.random.uniform(0.5, 2.0))
        
        # 转换为numpy数组
        for key in engine_params:
            engine_params[key] = np.array(engine_params[key])

        return engine_params

    def _generate_attitude_parameters(self, altitude: np.ndarray) -> Dict[str, np.ndarray]:
        """生成姿态参数"""
        attitude_params = {}

        # 初始化参数
        pitch = np.zeros(self.total_samples)
        roll = np.zeros(self.total_samples)
        yaw = np.zeros(self.total_samples)

        for i, t in enumerate(self.time_vector):
            phase = self._get_flight_phase(t)

            if phase == 'takeoff':
                pitch[i] = np.random.uniform(8, 15) + np.random.normal(0, 1)
            elif phase == 'climb':
                pitch[i] = np.random.uniform(5, 8) + np.random.normal(0, 0.5)
            elif phase == 'cruise':
                pitch[i] = np.random.uniform(-1, 3) + np.random.normal(0, 0.3)
            elif phase == 'descent':
                pitch[i] = np.random.uniform(-5, -1) + np.random.normal(0, 0.5)
            elif phase == 'approach':
                pitch[i] = np.random.uniform(-3, 0) + np.random.normal(0, 0.5)
            elif phase == 'landing':
                pitch[i] = np.random.uniform(2, 5) + np.random.normal(0, 1)
            else:
                pitch[i] = np.random.normal(0, 0.5)

            # 滚转角（转弯时会有变化）
            if np.random.random() < 0.1:  # 10%概率转弯
                roll[i] = np.random.uniform(-25, 25)
            else:
                roll[i] = np.random.normal(0, 2)

            # 偏航角（航向变化）
            if i == 0:
                yaw[i] = np.random.uniform(0, 360)
            else:
                yaw[i] = yaw[i-1] + np.random.normal(0, 0.5)
                if yaw[i] < 0:
                    yaw[i] += 360
                elif yaw[i] >= 360:
                    yaw[i] -= 360

        attitude_params['ATT_PITCH'] = pitch
        attitude_params['ATT_ROLL'] = roll
        attitude_params['ATT_YAW'] = yaw
        attitude_params['ATT_FPA'] = pitch + np.random.normal(0, 0.5, self.total_samples)
        attitude_params['ATT_SIDESLIP'] = np.random.normal(0, 1, self.total_samples)

        return attitude_params

    def _generate_control_parameters(self, attitude_params: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """生成操纵参数"""
        control_params = {}

        # 操纵面位置与姿态相关
        control_params['CTRL_AIL'] = attitude_params['ATT_ROLL'] * 0.3 + np.random.normal(0, 1, self.total_samples)
        control_params['CTRL_ELEV'] = attitude_params['ATT_PITCH'] * 0.5 + np.random.normal(0, 1, self.total_samples)
        control_params['CTRL_RUDDER'] = attitude_params['ATT_SIDESLIP'] * 0.8 + np.random.normal(0, 0.5, self.total_samples)

        # 扰流板位置
        spoiler = np.zeros(self.total_samples)
        for i, t in enumerate(self.time_vector):
            phase = self._get_flight_phase(t)
            if phase == 'descent' or phase == 'approach':
                spoiler[i] = np.random.uniform(0, 50)
            elif phase == 'landing':
                spoiler[i] = np.random.uniform(80, 100)
        control_params['CTRL_SPOILER'] = spoiler

        # 驾驶杆和脚蹬输入
        control_params['CTRL_COLUMN'] = attitude_params['ATT_PITCH'] * 2 + np.random.normal(0, 5, self.total_samples)
        control_params['CTRL_PEDAL'] = attitude_params['ATT_SIDESLIP'] * 3 + np.random.normal(0, 3, self.total_samples)

        # 自动驾驶仪状态
        ap_status = np.zeros(self.total_samples)
        for i, t in enumerate(self.time_vector):
            phase = self._get_flight_phase(t)
            if phase in ['climb', 'cruise', 'descent']:
                ap_status[i] = 1  # 自动驾驶开启
        control_params['AP_STATUS'] = ap_status

        return control_params

    def _generate_navigation_parameters(self) -> Dict[str, np.ndarray]:
        """生成导航参数"""
        nav_params = {}

        # 起始位置（北京首都机场示例）
        start_lat = 40.0801
        start_lon = 116.5844

        # 目标位置（上海浦东机场示例）
        end_lat = 31.1434
        end_lon = 121.8052

        # 生成航迹
        lat_track = np.linspace(start_lat, end_lat, self.total_samples)
        lon_track = np.linspace(start_lon, end_lon, self.total_samples)

        # 添加噪声和航路偏差
        lat_noise = np.random.normal(0, 0.01, self.total_samples)
        lon_noise = np.random.normal(0, 0.01, self.total_samples)

        nav_params['LAT'] = lat_track + lat_noise
        nav_params['LON'] = lon_track + lon_noise

        # 航向
        nav_params['HDG'] = np.arctan2(end_lon - start_lon, end_lat - start_lat) * 180 / np.pi + np.random.normal(0, 5, self.total_samples)

        # GPS状态
        nav_params['GPS_STATUS'] = np.ones(self.total_samples)  # 1表示正常

        # 惯性导航航向
        nav_params['IRS_HEADING'] = nav_params['HDG'] + np.random.normal(0, 1, self.total_samples)

        return nav_params

    def _generate_environmental_parameters(self) -> Dict[str, np.ndarray]:
        """生成环境参数"""
        env_params = {}

        # 风速和风向
        env_params['WIND_SPEED'] = np.random.uniform(5, 50, self.total_samples)
        env_params['WIND_DIR'] = np.random.uniform(0, 360, self.total_samples)

        # 湍流强度
        turb_intensity = np.zeros(self.total_samples)
        for i, t in enumerate(self.time_vector):
            phase = self._get_flight_phase(t)
            if phase in ['climb', 'cruise', 'descent']:
                turb_intensity[i] = np.random.choice([0, 1, 2, 3], p=[0.7, 0.2, 0.08, 0.02])  # 0-轻微, 1-轻度, 2-中度, 3-重度
            else:
                turb_intensity[i] = np.random.choice([0, 1], p=[0.9, 0.1])
        env_params['TURB_INT'] = turb_intensity

        return env_params

    def _generate_system_events(self) -> Dict[str, np.ndarray]:
        """生成系统事件参数"""
        events = {}

        # 飞行阶段标志
        takeoff_flag = np.zeros(self.total_samples)
        landing_flag = np.zeros(self.total_samples)

        for i, t in enumerate(self.time_vector):
            phase = self._get_flight_phase(t)
            if phase == 'takeoff':
                takeoff_flag[i] = 1
            elif phase == 'landing':
                landing_flag[i] = 1

        events['TOFF_FLAG'] = takeoff_flag
        events['LAND_FLAG'] = landing_flag

        # 发动机失效标志（低概率事件）
        eng_fail = np.zeros(self.total_samples)
        if np.random.random() < 0.01:  # 1%概率发生发动机故障
            fail_start = np.random.randint(0, self.total_samples - 100)
            fail_duration = np.random.randint(50, 200)
            eng_fail[fail_start:fail_start + fail_duration] = 1
        events['ENG_FAIL'] = eng_fail

        return events

    def _calculate_derived_parameters(self, altitude: np.ndarray, ias: np.ndarray, tas: np.ndarray) -> Dict[str, np.ndarray]:
        """计算衍生参数"""
        derived = {}

        # 垂直速度
        vs = np.zeros(self.total_samples)
        for i in range(1, self.total_samples):
            vs[i] = (altitude[i] - altitude[i-1]) / (self.time_vector[i] - self.time_vector[i-1]) * 60  # ft/min
        derived['VS'] = vs

        # 地速（简化计算）
        derived['GS'] = tas + np.random.normal(0, 10, self.total_samples)

        # 燃油量（递减）
        initial_fuel = 15000  # kg
        fuel_consumption_rate = np.random.uniform(2000, 4000)  # kg/h
        fuel_qty = np.maximum(0, initial_fuel - (self.time_vector / 3600) * fuel_consumption_rate)
        derived['FUEL_QTY_1'] = fuel_qty

        # 总燃油流量
        derived['FUEL_FLOW'] = np.gradient(fuel_qty, self.time_vector) * -3600  # kg/h

        return derived

    def generate_complete_dataset(self) -> pd.DataFrame:
        """生成完整的QAR数据集"""
        print("正在生成QAR数据集...")

        # 生成基础参数
        altitude = self._generate_altitude_profile()
        ias, tas = self._generate_airspeed_profile(altitude)

        # 生成各类参数
        engine_params = self._generate_engine_parameters(altitude, ias)
        attitude_params = self._generate_attitude_parameters(altitude)
        control_params = self._generate_control_parameters(attitude_params)
        nav_params = self._generate_navigation_parameters()
        env_params = self._generate_environmental_parameters()
        events = self._generate_system_events()
        derived_params = self._calculate_derived_parameters(altitude, ias, tas)

        # 合并所有参数
        data_dict = {
            'TIMESTAMP': [datetime.now() + timedelta(seconds=t) for t in self.time_vector],
            'ALT_STD': altitude,
            'IAS': ias,
            'TAS': tas,
        }

        # 添加所有参数组
        for param_group in [engine_params, attitude_params, control_params, nav_params, env_params, events, derived_params]:
            data_dict.update(param_group)

        # 添加飞行阶段标识
        flight_phases = [self._get_flight_phase(t) for t in self.time_vector]
        data_dict['FLIGHT_PHASE'] = flight_phases

        # 创建DataFrame
        df = pd.DataFrame(data_dict)

        # 添加一些离散参数
        df['XPDR_CODE'] = np.random.choice([1200, 2000, 7000], size=self.total_samples, p=[0.8, 0.15, 0.05])
        df['TCAS_TA'] = np.random.choice([0, 1], size=self.total_samples, p=[0.95, 0.05])

        print(f"QAR数据集生成完成，共{len(df)}条记录")
        return df

    def add_anomalies(self, df: pd.DataFrame, anomaly_rate: float = 0.05) -> pd.DataFrame:
        """向数据集中添加异常情况"""
        df_anomaly = df.copy()
        n_anomalies = int(len(df) * anomaly_rate)

        anomaly_indices = np.random.choice(len(df), n_anomalies, replace=False)

        for idx in anomaly_indices:
            # 随机选择异常类型
            anomaly_type = np.random.choice(['overspeed', 'altitude_deviation', 'engine_anomaly', 'control_anomaly'])

            if anomaly_type == 'overspeed':
                df_anomaly.loc[idx, 'IAS'] *= 1.2
            elif anomaly_type == 'altitude_deviation':
                df_anomaly.loc[idx, 'ALT_STD'] += np.random.uniform(-1000, 1000)
            elif anomaly_type == 'engine_anomaly':
                df_anomaly.loc[idx, 'ENG_N1_1'] *= 0.8
                df_anomaly.loc[idx, 'ENG_EGT_1'] *= 1.3
            elif anomaly_type == 'control_anomaly':
                df_anomaly.loc[idx, 'ATT_PITCH'] += np.random.uniform(-10, 10)

        return df_anomaly
