"""
QAR数据分析系统主程序
演示完整的数据分析流程
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 添加src目录到路径
sys.path.append('src')

from data_generation.qar_simulator import QARDataSimulator
from visualization.qar_visualizer import QARVisualizer
from models.safety_models import AnomalyDetector, FlightPhaseClassifier, RiskAssessmentModel
from models.optimization_models import FuelOptimizationModel, MaintenancePredictionModel
from utils.data_processing import QARDataProcessor, FlightPhaseDetector


def main():
    """主函数"""
    print("=" * 60)
    print("QAR数据分析系统演示")
    print("=" * 60)
    
    # 创建必要的目录
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('data/models', exist_ok=True)
    os.makedirs('static/visualizations', exist_ok=True)
    
    # 1. 数据生成
    print("\n1. 生成QAR模拟数据...")
    simulator = QARDataSimulator(flight_duration_minutes=120, sampling_rate_hz=1.0)
    qar_data = simulator.generate_complete_dataset()
    
    # 添加异常数据
    qar_data_with_anomalies = simulator.add_anomalies(qar_data, anomaly_rate=0.05)
    
    # 保存原始数据
    qar_data_with_anomalies.to_csv('data/raw/qar_simulation.csv', index=False)
    print(f"✓ 生成了{len(qar_data_with_anomalies)}条QAR数据记录")
    
    # 2. 数据处理
    print("\n2. 数据预处理...")
    processor = QARDataProcessor()
    
    # 数据清洗
    clean_data = processor.clean_data(qar_data_with_anomalies)
    
    # 添加衍生特征
    enhanced_data = processor.add_derived_features(clean_data)
    
    # 保存处理后的数据
    enhanced_data.to_csv('data/processed/qar_enhanced.csv', index=False)
    print(f"✓ 数据预处理完成，特征数量: {len(enhanced_data.columns)}")
    
    # 3. 可视化分析
    print("\n3. 生成可视化图表...")
    visualizer = QARVisualizer()
    
    try:
        # 生成完整的可视化仪表板
        dashboard_files = visualizer.create_dashboard(enhanced_data, 'static/visualizations/')
        print("✓ 可视化图表生成完成:")
        for name, path in dashboard_files.items():
            print(f"  - {name}: {path}")
    except Exception as e:
        print(f"⚠ 可视化生成部分失败: {e}")
    
    # 4. 安全性分析
    print("\n4. 安全性分析...")
    
    # 异常检测
    try:
        anomaly_detector = AnomalyDetector(contamination=0.1)
        anomaly_detector.fit(enhanced_data)
        anomaly_labels, anomaly_scores = anomaly_detector.predict(enhanced_data)
        
        n_anomalies = np.sum(anomaly_labels == -1)
        print(f"✓ 异常检测完成，发现{n_anomalies}个异常点 ({n_anomalies/len(enhanced_data)*100:.2f}%)")
        
        # 保存异常检测模型
        anomaly_detector.save_model('data/models/anomaly_detector.pkl')
    except Exception as e:
        print(f"⚠ 异常检测失败: {e}")
    
    # 飞行阶段分类
    try:
        phase_classifier = FlightPhaseClassifier()
        phase_classifier.fit(enhanced_data)
        predicted_phases, phase_probs = phase_classifier.predict(enhanced_data)
        print("✓ 飞行阶段分类模型训练完成")
    except Exception as e:
        print(f"⚠ 飞行阶段分类失败: {e}")
    
    # 风险评估
    try:
        risk_assessor = RiskAssessmentModel()
        risk_df = risk_assessor.assess_safety_risks(enhanced_data)
        safety_report = risk_assessor.generate_safety_report(risk_df)
        
        print("✓ 安全风险评估完成:")
        print(f"  - 总记录数: {safety_report['total_records']}")
        print(f"  - 高风险时段: {safety_report['high_risk_periods']}")
        print("  - 风险等级分布:", safety_report['risk_summary'])
        
        # 保存风险评估结果
        risk_df.to_csv('data/processed/risk_assessment.csv', index=False)
    except Exception as e:
        print(f"⚠ 风险评估失败: {e}")
    
    # 5. 成本优化分析
    print("\n5. 成本优化分析...")
    
    # 燃油优化
    try:
        fuel_optimizer = FuelOptimizationModel()
        fuel_optimizer.fit(enhanced_data)
        
        optimization_results = fuel_optimizer.optimize_flight_profile(enhanced_data)
        predicted_fuel = fuel_optimizer.predict_fuel_consumption(enhanced_data)
        
        actual_fuel = enhanced_data['ENG_FUEL_FLOW_1'].mean()
        predicted_fuel_avg = np.mean(predicted_fuel)
        potential_savings = actual_fuel - predicted_fuel_avg
        
        print("✓ 燃油优化分析完成:")
        print(f"  - 平均燃油流量: {actual_fuel:.2f} kg/h")
        print(f"  - 优化后预测: {predicted_fuel_avg:.2f} kg/h")
        print(f"  - 潜在节省: {potential_savings:.2f} kg/h")
        
        if 'recommendations' in optimization_results:
            print("  - 优化建议:")
            for rec in optimization_results['recommendations']:
                print(f"    • {rec}")
    except Exception as e:
        print(f"⚠ 燃油优化分析失败: {e}")
    
    # 维护预测
    try:
        maintenance_predictor = MaintenancePredictionModel()
        maintenance_predictor.fit(enhanced_data)
        
        maintenance_predictions = maintenance_predictor.predict_maintenance_needs(enhanced_data)
        maintenance_schedule = maintenance_predictor.generate_maintenance_schedule(maintenance_predictions)
        
        print("✓ 维护预测分析完成:")
        for component, schedule in maintenance_schedule.items():
            print(f"  - {component}: {schedule['next_maintenance_in_hours']:.0f}小时后维护 "
                  f"(紧急程度: {schedule['urgency']})")
    except Exception as e:
        print(f"⚠ 维护预测分析失败: {e}")
    
    # 6. 生成综合报告
    print("\n6. 生成分析报告...")
    
    report = {
        "analysis_timestamp": datetime.now().isoformat(),
        "data_summary": {
            "total_records": len(enhanced_data),
            "flight_duration_minutes": 120,
            "flight_phases": enhanced_data['FLIGHT_PHASE'].value_counts().to_dict()
        },
        "safety_analysis": {
            "anomalies_detected": int(n_anomalies) if 'n_anomalies' in locals() else 0,
            "safety_report": safety_report if 'safety_report' in locals() else {}
        },
        "optimization_analysis": {
            "fuel_savings_potential": float(potential_savings) if 'potential_savings' in locals() else 0,
            "maintenance_schedule": maintenance_schedule if 'maintenance_schedule' in locals() else {}
        }
    }
    
    # 保存报告
    import json
    with open('data/processed/analysis_report.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print("✓ 分析报告已保存到 data/processed/analysis_report.json")
    
    # 7. 启动Web服务提示
    print("\n" + "=" * 60)
    print("分析完成！")
    print("\n要启动Web服务，请运行:")
    print("cd src/api && python main.py")
    print("\n然后访问: http://localhost:8000")
    print("API文档: http://localhost:8000/docs")
    print("=" * 60)


if __name__ == "__main__":
    main()
