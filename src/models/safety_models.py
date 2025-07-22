"""
安全性相关的机器学习模型
包括异常检测、风险预警、操作合规性检查等
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
import joblib
import warnings
warnings.filterwarnings('ignore')


class AnomalyDetector:
    """异常检测模型"""
    
    def __init__(self, contamination: float = 0.1):
        """
        初始化异常检测器
        
        Args:
            contamination: 异常比例
        """
        self.contamination = contamination
        self.scaler = StandardScaler()
        self.isolation_forest = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100
        )
        self.feature_names = None
        
    def prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """准备特征数据"""
        # 选择关键安全参数
        safety_features = [
            'ALT_STD', 'IAS', 'TAS', 'VS', 'ATT_PITCH', 'ATT_ROLL',
            'ENG_N1_1', 'ENG_N2_1', 'ENG_EGT_1', 'ENG_FUEL_FLOW_1',
            'CTRL_ELEV', 'CTRL_AIL', 'WIND_SPEED'
        ]
        
        # 添加衍生特征
        df_features = df.copy()
        df_features['SPEED_CHANGE_RATE'] = df_features['IAS'].diff()
        df_features['ALT_CHANGE_RATE'] = df_features['ALT_STD'].diff()
        df_features['ENGINE_TEMP_RATIO'] = df_features['ENG_EGT_1'] / (df_features['ENG_N1_1'] + 1e-6)
        
        safety_features.extend(['SPEED_CHANGE_RATE', 'ALT_CHANGE_RATE', 'ENGINE_TEMP_RATIO'])
        
        # 处理缺失值
        feature_data = df_features[safety_features].fillna(method='ffill').fillna(0)
        self.feature_names = safety_features
        
        return feature_data.values
    
    def fit(self, df: pd.DataFrame):
        """训练异常检测模型"""
        X = self.prepare_features(df)
        X_scaled = self.scaler.fit_transform(X)
        self.isolation_forest.fit(X_scaled)
        
        print(f"异常检测模型训练完成，使用{len(self.feature_names)}个特征")
        
    def predict(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """预测异常"""
        X = self.prepare_features(df)
        X_scaled = self.scaler.transform(X)
        
        # 预测异常（-1为异常，1为正常）
        anomaly_labels = self.isolation_forest.predict(X_scaled)
        
        # 计算异常分数
        anomaly_scores = self.isolation_forest.decision_function(X_scaled)
        
        return anomaly_labels, anomaly_scores
    
    def save_model(self, filepath: str):
        """保存模型"""
        model_data = {
            'scaler': self.scaler,
            'isolation_forest': self.isolation_forest,
            'feature_names': self.feature_names,
            'contamination': self.contamination
        }
        joblib.dump(model_data, filepath)
        
    def load_model(self, filepath: str):
        """加载模型"""
        model_data = joblib.load(filepath)
        self.scaler = model_data['scaler']
        self.isolation_forest = model_data['isolation_forest']
        self.feature_names = model_data['feature_names']
        self.contamination = model_data['contamination']


class FlightPhaseClassifier:
    """飞行阶段分类器"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.classifier = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            max_depth=10
        )
        self.label_encoder = LabelEncoder()
        
    def prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """准备分类特征"""
        features = [
            'ALT_STD', 'IAS', 'VS', 'ATT_PITCH',
            'ENG_N1_1', 'ENG_FUEL_FLOW_1'
        ]
        
        # 添加时间特征
        df_features = df.copy()
        df_features['TIME_NORMALIZED'] = (df_features.index - df_features.index.min()) / len(df_features)
        features.append('TIME_NORMALIZED')
        
        return df_features[features].fillna(method='ffill').fillna(0).values
    
    def fit(self, df: pd.DataFrame):
        """训练飞行阶段分类器"""
        X = self.prepare_features(df)
        y = self.label_encoder.fit_transform(df['FLIGHT_PHASE'])
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        self.classifier.fit(X_train_scaled, y_train)
        
        # 评估模型
        y_pred = self.classifier.predict(X_test_scaled)
        print("飞行阶段分类器性能:")
        print(classification_report(y_test, y_pred, 
                                  target_names=self.label_encoder.classes_))
        
    def predict(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """预测飞行阶段"""
        X = self.prepare_features(df)
        X_scaled = self.scaler.transform(X)
        
        predictions = self.classifier.predict(X_scaled)
        probabilities = self.classifier.predict_proba(X_scaled)
        
        # 转换回原始标签
        phase_labels = self.label_encoder.inverse_transform(predictions)
        
        return phase_labels, probabilities


class RiskAssessmentModel:
    """风险评估模型"""
    
    def __init__(self):
        self.risk_thresholds = {
            'altitude_deviation': 500,  # ft
            'speed_deviation': 20,      # kt
            'engine_temp_high': 800,    # °C
            'fuel_flow_high': 3000,     # kg/h
            'pitch_extreme': 15,        # degrees
            'roll_extreme': 30          # degrees
        }
        
    def assess_safety_risks(self, df: pd.DataFrame) -> pd.DataFrame:
        """评估安全风险"""
        risk_df = df.copy()
        
        # 高度偏差风险
        cruise_data = df[df['FLIGHT_PHASE'] == 'cruise']
        if len(cruise_data) > 0:
            target_altitude = cruise_data['ALT_STD'].median()
            risk_df['ALTITUDE_RISK'] = np.abs(df['ALT_STD'] - target_altitude) > self.risk_thresholds['altitude_deviation']
        else:
            risk_df['ALTITUDE_RISK'] = False
        
        # 速度偏差风险
        risk_df['SPEED_RISK'] = False
        for phase in df['FLIGHT_PHASE'].unique():
            phase_data = df[df['FLIGHT_PHASE'] == phase]
            if len(phase_data) > 10:
                target_speed = phase_data['IAS'].median()
                speed_deviation = np.abs(phase_data['IAS'] - target_speed)
                risk_df.loc[df['FLIGHT_PHASE'] == phase, 'SPEED_RISK'] = \
                    speed_deviation > self.risk_thresholds['speed_deviation']
        
        # 发动机温度风险
        risk_df['ENGINE_TEMP_RISK'] = df['ENG_EGT_1'] > self.risk_thresholds['engine_temp_high']
        
        # 燃油流量风险
        risk_df['FUEL_FLOW_RISK'] = df['ENG_FUEL_FLOW_1'] > self.risk_thresholds['fuel_flow_high']
        
        # 姿态风险
        risk_df['PITCH_RISK'] = np.abs(df['ATT_PITCH']) > self.risk_thresholds['pitch_extreme']
        risk_df['ROLL_RISK'] = np.abs(df['ATT_ROLL']) > self.risk_thresholds['roll_extreme']
        
        # 综合风险评分
        risk_columns = ['ALTITUDE_RISK', 'SPEED_RISK', 'ENGINE_TEMP_RISK', 
                       'FUEL_FLOW_RISK', 'PITCH_RISK', 'ROLL_RISK']
        risk_df['TOTAL_RISK_SCORE'] = risk_df[risk_columns].sum(axis=1)
        
        # 风险等级
        risk_df['RISK_LEVEL'] = pd.cut(
            risk_df['TOTAL_RISK_SCORE'],
            bins=[-1, 0, 1, 2, 6],
            labels=['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']
        )
        
        return risk_df
    
    def generate_safety_report(self, risk_df: pd.DataFrame) -> Dict:
        """生成安全报告"""
        report = {
            'total_records': len(risk_df),
            'risk_summary': risk_df['RISK_LEVEL'].value_counts().to_dict(),
            'high_risk_periods': len(risk_df[risk_df['RISK_LEVEL'].isin(['HIGH', 'CRITICAL'])]),
            'risk_factors': {
                'altitude_violations': risk_df['ALTITUDE_RISK'].sum(),
                'speed_violations': risk_df['SPEED_RISK'].sum(),
                'engine_temp_violations': risk_df['ENGINE_TEMP_RISK'].sum(),
                'fuel_flow_violations': risk_df['FUEL_FLOW_RISK'].sum(),
                'pitch_violations': risk_df['PITCH_RISK'].sum(),
                'roll_violations': risk_df['ROLL_RISK'].sum()
            }
        }
        
        return report
