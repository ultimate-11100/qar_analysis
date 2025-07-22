"""
QAR数据分析系统测试
"""

import unittest
import sys
import os
import pandas as pd
import numpy as np

# 添加src目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_generation.qar_simulator import QARDataSimulator
from visualization.qar_visualizer import QARVisualizer
from models.safety_models import AnomalyDetector, FlightPhaseClassifier, RiskAssessmentModel
from models.optimization_models import FuelOptimizationModel, MaintenancePredictionModel
from utils.data_processing import QARDataProcessor, FlightPhaseDetector


class TestQARDataSimulator(unittest.TestCase):
    """测试QAR数据模拟器"""
    
    def setUp(self):
        self.simulator = QARDataSimulator(flight_duration_minutes=60, sampling_rate_hz=1.0)
    
    def test_data_generation(self):
        """测试数据生成"""
        df = self.simulator.generate_complete_dataset()
        
        # 检查数据形状
        self.assertEqual(len(df), 3600)  # 60分钟 * 60秒 * 1Hz
        
        # 检查必要的列
        required_columns = ['ALT_STD', 'IAS', 'TAS', 'ENG_N1_1', 'ATT_PITCH', 'FLIGHT_PHASE']
        for col in required_columns:
            self.assertIn(col, df.columns)
        
        # 检查数据范围
        self.assertTrue(df['ALT_STD'].min() >= 0)
        self.assertTrue(df['IAS'].min() >= 0)
        self.assertTrue(df['ENG_N1_1'].max() <= 100)
    
    def test_anomaly_addition(self):
        """测试异常数据添加"""
        df = self.simulator.generate_complete_dataset()
        df_with_anomalies = self.simulator.add_anomalies(df, anomaly_rate=0.1)
        
        # 数据长度应该相同
        self.assertEqual(len(df), len(df_with_anomalies))
        
        # 应该有一些差异
        self.assertFalse(df.equals(df_with_anomalies))


class TestQARVisualizer(unittest.TestCase):
    """测试QAR可视化器"""
    
    def setUp(self):
        self.visualizer = QARVisualizer()
        self.simulator = QARDataSimulator(flight_duration_minutes=30, sampling_rate_hz=1.0)
        self.test_data = self.simulator.generate_complete_dataset()
    
    def test_takeoff_parameters_plot(self):
        """测试起飞参数图"""
        fig = self.visualizer.plot_takeoff_parameters(self.test_data)
        self.assertIsNotNone(fig)
    
    def test_flight_phases_plot(self):
        """测试飞行阶段图"""
        fig = self.visualizer.plot_flight_phases(self.test_data)
        self.assertIsNotNone(fig)
    
    def test_parameter_distributions_plot(self):
        """测试参数分布图"""
        fig = self.visualizer.plot_parameter_distributions(self.test_data)
        self.assertIsNotNone(fig)
    
    def test_correlation_heatmap(self):
        """测试相关性热力图"""
        fig = self.visualizer.plot_correlation_heatmap(self.test_data)
        self.assertIsNotNone(fig)


class TestAnomalyDetector(unittest.TestCase):
    """测试异常检测器"""
    
    def setUp(self):
        self.detector = AnomalyDetector(contamination=0.1)
        self.simulator = QARDataSimulator(flight_duration_minutes=30, sampling_rate_hz=1.0)
        self.test_data = self.simulator.generate_complete_dataset()
    
    def test_feature_preparation(self):
        """测试特征准备"""
        features = self.detector.prepare_features(self.test_data)
        self.assertIsInstance(features, np.ndarray)
        self.assertEqual(len(features), len(self.test_data))
    
    def test_model_training(self):
        """测试模型训练"""
        self.detector.fit(self.test_data)
        self.assertIsNotNone(self.detector.isolation_forest)
        self.assertIsNotNone(self.detector.scaler)
    
    def test_anomaly_prediction(self):
        """测试异常预测"""
        self.detector.fit(self.test_data)
        labels, scores = self.detector.predict(self.test_data)
        
        self.assertEqual(len(labels), len(self.test_data))
        self.assertEqual(len(scores), len(self.test_data))
        self.assertTrue(np.all(np.isin(labels, [-1, 1])))


class TestFlightPhaseClassifier(unittest.TestCase):
    """测试飞行阶段分类器"""
    
    def setUp(self):
        self.classifier = FlightPhaseClassifier()
        self.simulator = QARDataSimulator(flight_duration_minutes=30, sampling_rate_hz=1.0)
        self.test_data = self.simulator.generate_complete_dataset()
    
    def test_feature_preparation(self):
        """测试特征准备"""
        features = self.classifier.prepare_features(self.test_data)
        self.assertIsInstance(features, np.ndarray)
        self.assertEqual(len(features), len(self.test_data))
    
    def test_model_training(self):
        """测试模型训练"""
        self.classifier.fit(self.test_data)
        self.assertIsNotNone(self.classifier.classifier)
        self.assertIsNotNone(self.classifier.label_encoder)
    
    def test_phase_prediction(self):
        """测试阶段预测"""
        self.classifier.fit(self.test_data)
        phases, probs = self.classifier.predict(self.test_data)
        
        self.assertEqual(len(phases), len(self.test_data))
        self.assertEqual(probs.shape[0], len(self.test_data))


class TestRiskAssessmentModel(unittest.TestCase):
    """测试风险评估模型"""
    
    def setUp(self):
        self.risk_assessor = RiskAssessmentModel()
        self.simulator = QARDataSimulator(flight_duration_minutes=30, sampling_rate_hz=1.0)
        self.test_data = self.simulator.generate_complete_dataset()
    
    def test_safety_risk_assessment(self):
        """测试安全风险评估"""
        risk_df = self.risk_assessor.assess_safety_risks(self.test_data)
        
        # 检查风险列是否添加
        risk_columns = ['ALTITUDE_RISK', 'SPEED_RISK', 'ENGINE_TEMP_RISK', 
                       'FUEL_FLOW_RISK', 'PITCH_RISK', 'ROLL_RISK']
        for col in risk_columns:
            self.assertIn(col, risk_df.columns)
        
        # 检查风险评分和等级
        self.assertIn('TOTAL_RISK_SCORE', risk_df.columns)
        self.assertIn('RISK_LEVEL', risk_df.columns)
    
    def test_safety_report_generation(self):
        """测试安全报告生成"""
        risk_df = self.risk_assessor.assess_safety_risks(self.test_data)
        report = self.risk_assessor.generate_safety_report(risk_df)
        
        self.assertIn('total_records', report)
        self.assertIn('risk_summary', report)
        self.assertIn('risk_factors', report)


class TestFuelOptimizationModel(unittest.TestCase):
    """测试燃油优化模型"""
    
    def setUp(self):
        self.fuel_optimizer = FuelOptimizationModel()
        self.simulator = QARDataSimulator(flight_duration_minutes=30, sampling_rate_hz=1.0)
        self.test_data = self.simulator.generate_complete_dataset()
    
    def test_feature_preparation(self):
        """测试特征准备"""
        X, y = self.fuel_optimizer.prepare_features(self.test_data)
        self.assertIsInstance(X, np.ndarray)
        self.assertIsInstance(y, np.ndarray)
        self.assertEqual(len(X), len(y))
    
    def test_model_training(self):
        """测试模型训练"""
        self.fuel_optimizer.fit(self.test_data)
        self.assertIsNotNone(self.fuel_optimizer.fuel_model)
        self.assertIsNotNone(self.fuel_optimizer.scaler)
    
    def test_fuel_prediction(self):
        """测试燃油预测"""
        self.fuel_optimizer.fit(self.test_data)
        predictions = self.fuel_optimizer.predict_fuel_consumption(self.test_data)
        
        self.assertEqual(len(predictions), len(self.test_data))
        self.assertTrue(np.all(predictions >= 0))
    
    def test_flight_profile_optimization(self):
        """测试飞行剖面优化"""
        self.fuel_optimizer.fit(self.test_data)
        optimization_results = self.fuel_optimizer.optimize_flight_profile(self.test_data)
        
        self.assertIsInstance(optimization_results, dict)
        if 'efficiency_analysis' in optimization_results:
            self.assertIsInstance(optimization_results['efficiency_analysis'], dict)


class TestQARDataProcessor(unittest.TestCase):
    """测试QAR数据处理器"""
    
    def setUp(self):
        self.processor = QARDataProcessor()
        self.simulator = QARDataSimulator(flight_duration_minutes=30, sampling_rate_hz=1.0)
        self.test_data = self.simulator.generate_complete_dataset()
    
    def test_data_cleaning(self):
        """测试数据清洗"""
        # 添加一些缺失值
        test_data_with_na = self.test_data.copy()
        test_data_with_na.loc[0:10, 'ALT_STD'] = np.nan
        
        cleaned_data = self.processor.clean_data(test_data_with_na)
        
        # 检查缺失值是否被处理
        self.assertFalse(cleaned_data['ALT_STD'].isnull().any())
    
    def test_derived_features(self):
        """测试衍生特征"""
        enhanced_data = self.processor.add_derived_features(self.test_data)
        
        # 检查是否添加了新特征
        self.assertGreater(len(enhanced_data.columns), len(self.test_data.columns))
        
        # 检查特定的衍生特征
        if 'CLIMB_RATE' in enhanced_data.columns:
            self.assertIn('CLIMB_RATE', enhanced_data.columns)
    
    def test_normalization(self):
        """测试特征标准化"""
        normalized_data = self.processor.normalize_features(self.test_data)
        
        # 检查数值列是否被标准化
        numeric_columns = self.test_data.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if col in normalized_data.columns:
                # 标准化后的数据应该接近0均值，1标准差
                self.assertAlmostEqual(normalized_data[col].mean(), 0, places=1)


if __name__ == '__main__':
    # 创建测试套件
    test_suite = unittest.TestSuite()
    
    # 添加测试类
    test_classes = [
        TestQARDataSimulator,
        TestQARVisualizer,
        TestAnomalyDetector,
        TestFlightPhaseClassifier,
        TestRiskAssessmentModel,
        TestFuelOptimizationModel,
        TestQARDataProcessor
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # 输出测试结果
    if result.wasSuccessful():
        print("\n" + "="*50)
        print("所有测试通过！")
        print("="*50)
    else:
        print("\n" + "="*50)
        print(f"测试失败: {len(result.failures)} 个失败, {len(result.errors)} 个错误")
        print("="*50)
