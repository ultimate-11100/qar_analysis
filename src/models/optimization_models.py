"""
成本优化相关的机器学习模型
包括燃油优化、维护预测、效率分析等
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
import joblib


class FuelOptimizationModel:
    """燃油优化模型"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.fuel_model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        )
        self.feature_names = None
        
    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """准备燃油预测特征"""
        # 燃油消耗相关特征
        fuel_features = [
            'ALT_STD', 'IAS', 'TAS', 'GS', 'VS',
            'ENG_N1_1', 'ENG_N2_1', 'ATT_PITCH',
            'WIND_SPEED', 'WIND_DIR'
        ]
        
        # 添加衍生特征
        df_features = df.copy()
        df_features['ALTITUDE_BAND'] = pd.cut(df_features['ALT_STD'], 
                                            bins=[0, 10000, 25000, 40000, 50000],
                                            labels=[0, 1, 2, 3])
        df_features['SPEED_EFFICIENCY'] = df_features['TAS'] / (df_features['ENG_N1_1'] + 1e-6)
        df_features['CLIMB_RATE_ABS'] = np.abs(df_features['VS'])
        
        fuel_features.extend(['ALTITUDE_BAND', 'SPEED_EFFICIENCY', 'CLIMB_RATE_ABS'])
        
        # 处理分类变量
        df_features['ALTITUDE_BAND'] = df_features['ALTITUDE_BAND'].astype(float)
        
        # 目标变量：燃油流量
        X = df_features[fuel_features].fillna(method='ffill').fillna(0)
        y = df_features['ENG_FUEL_FLOW_1'].fillna(method='ffill').fillna(0)
        
        self.feature_names = fuel_features
        return X.values, y.values
    
    def fit(self, df: pd.DataFrame):
        """训练燃油优化模型"""
        X, y = self.prepare_features(df)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        self.fuel_model.fit(X_train_scaled, y_train)
        
        # 评估模型
        y_pred = self.fuel_model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"燃油预测模型性能: MSE={mse:.2f}, R²={r2:.3f}")
        
        # 特征重要性
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.fuel_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("燃油消耗关键影响因素:")
        print(feature_importance.head())
        
    def predict_fuel_consumption(self, df: pd.DataFrame) -> np.ndarray:
        """预测燃油消耗"""
        X, _ = self.prepare_features(df)
        X_scaled = self.scaler.transform(X)
        return self.fuel_model.predict(X_scaled)
    
    def optimize_flight_profile(self, df: pd.DataFrame) -> Dict:
        """优化飞行剖面以降低燃油消耗"""
        # 分析不同飞行阶段的燃油效率
        efficiency_analysis = {}
        
        for phase in df['FLIGHT_PHASE'].unique():
            phase_data = df[df['FLIGHT_PHASE'] == phase]
            if len(phase_data) > 10:
                avg_fuel_flow = phase_data['ENG_FUEL_FLOW_1'].mean()
                avg_speed = phase_data['TAS'].mean()
                avg_altitude = phase_data['ALT_STD'].mean()
                
                # 燃油效率 = 速度 / 燃油流量
                fuel_efficiency = avg_speed / (avg_fuel_flow + 1e-6)
                
                efficiency_analysis[phase] = {
                    'avg_fuel_flow': avg_fuel_flow,
                    'avg_speed': avg_speed,
                    'avg_altitude': avg_altitude,
                    'fuel_efficiency': fuel_efficiency
                }
        
        # 找出最优和最差效率阶段
        if efficiency_analysis:
            best_phase = max(efficiency_analysis.keys(), 
                           key=lambda x: efficiency_analysis[x]['fuel_efficiency'])
            worst_phase = min(efficiency_analysis.keys(), 
                            key=lambda x: efficiency_analysis[x]['fuel_efficiency'])
            
            optimization_recommendations = {
                'efficiency_analysis': efficiency_analysis,
                'best_efficiency_phase': best_phase,
                'worst_efficiency_phase': worst_phase,
                'recommendations': self._generate_fuel_recommendations(efficiency_analysis)
            }
        else:
            optimization_recommendations = {'error': 'Insufficient data for analysis'}
        
        return optimization_recommendations
    
    def _generate_fuel_recommendations(self, efficiency_analysis: Dict) -> List[str]:
        """生成燃油优化建议"""
        recommendations = []
        
        # 分析爬升阶段
        if 'climb' in efficiency_analysis:
            climb_data = efficiency_analysis['climb']
            if climb_data['fuel_efficiency'] < 0.1:  # 阈值可调
                recommendations.append("建议优化爬升率，当前爬升阶段燃油效率较低")
        
        # 分析巡航阶段
        if 'cruise' in efficiency_analysis:
            cruise_data = efficiency_analysis['cruise']
            if cruise_data['avg_altitude'] < 30000:
                recommendations.append("建议提高巡航高度以改善燃油效率")
        
        # 分析下降阶段
        if 'descent' in efficiency_analysis:
            descent_data = efficiency_analysis['descent']
            if descent_data['avg_fuel_flow'] > 1000:  # 阈值可调
                recommendations.append("建议优化下降策略，减少发动机推力使用")
        
        return recommendations


class MaintenancePredictionModel:
    """维护预测模型"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.maintenance_model = RandomForestRegressor(
            n_estimators=100,
            random_state=42
        )
        self.component_models = {}
        
    def prepare_maintenance_features(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """准备维护预测特征"""
        # 发动机相关特征
        engine_features = [
            'ENG_N1_1', 'ENG_N2_1', 'ENG_EGT_1', 
            'ENG_OIL_P_1', 'ENG_OIL_T_1', 'ENG_VIB_1'
        ]
        
        # 计算累积使用时间和循环
        df_features = df.copy()
        df_features['FLIGHT_HOURS'] = df_features.index / 3600  # 转换为小时
        df_features['ENGINE_CYCLES'] = (df_features['TOFF_FLAG'].diff() > 0).cumsum()
        
        # 计算参数的统计特征
        for feature in engine_features:
            df_features[f'{feature}_MEAN'] = df_features[feature].rolling(window=100, min_periods=1).mean()
            df_features[f'{feature}_STD'] = df_features[feature].rolling(window=100, min_periods=1).std()
            df_features[f'{feature}_TREND'] = df_features[feature].diff().rolling(window=50, min_periods=1).mean()
        
        # 组件特定特征
        components = {
            'engine': engine_features + ['FLIGHT_HOURS', 'ENGINE_CYCLES'],
            'flight_controls': ['CTRL_AIL', 'CTRL_ELEV', 'CTRL_RUDDER', 'FLIGHT_HOURS'],
            'landing_gear': ['ALT_STD', 'IAS', 'LAND_FLAG', 'ENGINE_CYCLES']
        }
        
        component_data = {}
        for component, features in components.items():
            # 添加统计特征
            extended_features = features.copy()
            for base_feature in features[:3]:  # 只对前几个特征计算统计量
                if f'{base_feature}_MEAN' in df_features.columns:
                    extended_features.extend([f'{base_feature}_MEAN', f'{base_feature}_STD', f'{base_feature}_TREND'])
            
            component_data[component] = df_features[extended_features].fillna(method='ffill').fillna(0).values
        
        return component_data
    
    def simulate_maintenance_targets(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """模拟维护目标变量（实际应用中应使用历史维护记录）"""
        n_samples = len(df)
        
        # 模拟不同组件的剩余使用寿命（小时）
        targets = {
            'engine': np.random.exponential(scale=5000, size=n_samples),  # 发动机
            'flight_controls': np.random.exponential(scale=8000, size=n_samples),  # 飞控系统
            'landing_gear': np.random.exponential(scale=10000, size=n_samples)  # 起落架
        }
        
        # 添加基于实际参数的趋势
        engine_wear = (df['ENG_EGT_1'] - df['ENG_EGT_1'].min()) / (df['ENG_EGT_1'].max() - df['ENG_EGT_1'].min())
        targets['engine'] = targets['engine'] * (1 - engine_wear * 0.5)
        
        return targets
    
    def fit(self, df: pd.DataFrame):
        """训练维护预测模型"""
        component_features = self.prepare_maintenance_features(df)
        maintenance_targets = self.simulate_maintenance_targets(df)
        
        for component in component_features.keys():
            X = component_features[component]
            y = maintenance_targets[component]
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # 为每个组件创建独立的模型和缩放器
            scaler = StandardScaler()
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            model.fit(X_train_scaled, y_train)
            
            # 评估模型
            y_pred = model.predict(X_test_scaled)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            print(f"{component}维护预测模型性能: MSE={mse:.2f}, R²={r2:.3f}")
            
            # 保存模型和缩放器
            self.component_models[component] = {
                'model': model,
                'scaler': scaler
            }
    
    def predict_maintenance_needs(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """预测维护需求"""
        component_features = self.prepare_maintenance_features(df)
        predictions = {}
        
        for component, X in component_features.items():
            if component in self.component_models:
                model_data = self.component_models[component]
                X_scaled = model_data['scaler'].transform(X)
                predictions[component] = model_data['model'].predict(X_scaled)
        
        return predictions
    
    def generate_maintenance_schedule(self, predictions: Dict[str, np.ndarray], 
                                    current_time: float = 0) -> Dict:
        """生成维护计划"""
        schedule = {}
        
        for component, remaining_life in predictions.items():
            # 找出需要维护的时间点（剩余寿命小于阈值）
            maintenance_threshold = 500  # 小时
            urgent_maintenance = remaining_life < maintenance_threshold
            
            if np.any(urgent_maintenance):
                next_maintenance = np.argmax(urgent_maintenance)
                schedule[component] = {
                    'next_maintenance_in_hours': remaining_life[next_maintenance],
                    'urgency': 'HIGH' if remaining_life[next_maintenance] < 100 else 'MEDIUM',
                    'estimated_cost': self._estimate_maintenance_cost(component)
                }
            else:
                schedule[component] = {
                    'next_maintenance_in_hours': remaining_life.min(),
                    'urgency': 'LOW',
                    'estimated_cost': self._estimate_maintenance_cost(component)
                }
        
        return schedule
    
    def _estimate_maintenance_cost(self, component: str) -> float:
        """估算维护成本"""
        cost_estimates = {
            'engine': 50000,
            'flight_controls': 15000,
            'landing_gear': 25000
        }
        return cost_estimates.get(component, 10000)
