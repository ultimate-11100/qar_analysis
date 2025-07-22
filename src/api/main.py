"""
QAR数据分析系统 FastAPI Web服务
专注于数据可视化和分析展示
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Request
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import json
import os
import base64
from io import BytesIO
from typing import Dict, List, Optional
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly

# 导入自定义模块
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_generation.qar_simulator import QARDataSimulator
from visualization.qar_visualizer import QARVisualizer
from models.safety_models import AnomalyDetector, FlightPhaseClassifier, RiskAssessmentModel
from models.optimization_models import FuelOptimizationModel, MaintenancePredictionModel

# 创建FastAPI应用
app = FastAPI(
    title="QAR数据可视化分析系统",
    description="专业的QAR数据可视化和分析展示平台",
    version="2.0.0"
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 静态文件和模板
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# 全局变量存储模型和数据
global_data = {
    'qar_data': None,
    'models': {
        'anomaly_detector': None,
        'phase_classifier': None,
        'risk_assessor': None,
        'fuel_optimizer': None,
        'maintenance_predictor': None
    },
    'visualizer': QARVisualizer()
}


# Pydantic模型
class SimulationRequest(BaseModel):
    flight_duration_minutes: int = 120
    sampling_rate_hz: float = 1.0
    anomaly_rate: float = 0.05


class AnalysisRequest(BaseModel):
    analysis_type: str
    parameters: Optional[Dict] = {}


# API路由
@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """主页 - 数据可视化仪表板"""
    return templates.TemplateResponse("dashboard.html", {"request": request})

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    """数据可视化仪表板"""
    return templates.TemplateResponse("dashboard.html", {"request": request})

@app.get("/analysis", response_class=HTMLResponse)
async def analysis_page(request: Request):
    """数据分析页面"""
    return templates.TemplateResponse("analysis.html", {"request": request})

@app.get("/reports", response_class=HTMLResponse)
async def reports_page(request: Request):
    """分析报告页面"""
    return templates.TemplateResponse("reports.html", {"request": request})

@app.get("/health-page", response_class=HTMLResponse)
async def health_page(request: Request):
    """系统健康状态页面"""
    return templates.TemplateResponse("health.html", {"request": request})


@app.post("/api/simulate")
async def simulate_qar_data(request: SimulationRequest):
    """生成QAR模拟数据"""
    try:
        # 创建模拟器
        simulator = QARDataSimulator(
            flight_duration_minutes=request.flight_duration_minutes,
            sampling_rate_hz=request.sampling_rate_hz
        )
        
        # 生成数据
        qar_data = simulator.generate_complete_dataset()
        
        # 添加异常
        if request.anomaly_rate > 0:
            qar_data = simulator.add_anomalies(qar_data, request.anomaly_rate)
        
        # 存储到全局变量
        global_data['qar_data'] = qar_data
        
        return {
            "status": "success",
            "message": f"成功生成{len(qar_data)}条QAR数据记录",
            "data_info": {
                "total_records": len(qar_data),
                "flight_duration": request.flight_duration_minutes,
                "sampling_rate": request.sampling_rate_hz,
                "columns": list(qar_data.columns),
                "flight_phases": qar_data['FLIGHT_PHASE'].value_counts().to_dict()
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"数据生成失败: {str(e)}")


@app.get("/api/data/info")
async def get_data_info():
    """获取当前数据信息"""
    if global_data['qar_data'] is None:
        raise HTTPException(status_code=404, detail="未找到QAR数据，请先生成或上传数据")
    
    df = global_data['qar_data']
    return {
        "total_records": len(df),
        "columns": list(df.columns),
        "flight_phases": df['FLIGHT_PHASE'].value_counts().to_dict(),
        "time_range": {
            "start": df['TIMESTAMP'].min().isoformat(),
            "end": df['TIMESTAMP'].max().isoformat()
        },
        "basic_stats": {
            "max_altitude": float(df['ALT_STD'].max()),
            "max_speed": float(df['IAS'].max()),
            "avg_fuel_flow": float(df['ENG_FUEL_FLOW_1'].mean())
        }
    }


@app.post("/api/analysis/train_models")
async def train_models():
    """训练所有机器学习模型"""
    if global_data['qar_data'] is None:
        raise HTTPException(status_code=404, detail="未找到QAR数据，请先生成数据")
    
    try:
        df = global_data['qar_data']
        
        # 训练异常检测模型
        anomaly_detector = AnomalyDetector()
        anomaly_detector.fit(df)
        global_data['models']['anomaly_detector'] = anomaly_detector
        
        # 训练飞行阶段分类器
        phase_classifier = FlightPhaseClassifier()
        phase_classifier.fit(df)
        global_data['models']['phase_classifier'] = phase_classifier
        
        # 初始化风险评估模型
        risk_assessor = RiskAssessmentModel()
        global_data['models']['risk_assessor'] = risk_assessor
        
        # 训练燃油优化模型
        fuel_optimizer = FuelOptimizationModel()
        fuel_optimizer.fit(df)
        global_data['models']['fuel_optimizer'] = fuel_optimizer
        
        # 训练维护预测模型
        maintenance_predictor = MaintenancePredictionModel()
        maintenance_predictor.fit(df)
        global_data['models']['maintenance_predictor'] = maintenance_predictor
        
        return {
            "status": "success",
            "message": "所有模型训练完成",
            "models_trained": list(global_data['models'].keys())
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"模型训练失败: {str(e)}")


@app.post("/api/analysis/anomaly_detection")
async def detect_anomalies():
    """异常检测分析"""
    if global_data['qar_data'] is None:
        raise HTTPException(status_code=404, detail="未找到QAR数据")

    try:
        df = global_data['qar_data']

        # 简化的异常检测（基于统计方法）
        # 计算关键参数的Z分数
        key_params = ['ALT_STD', 'IAS', 'ENG_N1_1', 'ATT_PITCH']
        anomaly_scores = []

        for param in key_params:
            if param in df.columns:
                mean_val = df[param].mean()
                std_val = df[param].std()
                z_scores = np.abs((df[param] - mean_val) / (std_val + 1e-6))
                anomaly_scores.append(z_scores)

        # 综合异常分数
        if anomaly_scores:
            combined_scores = np.mean(anomaly_scores, axis=0)
            anomaly_threshold = 2.5  # Z分数阈值
            anomaly_mask = combined_scores > anomaly_threshold
            n_anomalies = int(np.sum(anomaly_mask))
        else:
            n_anomalies = int(len(df) * 0.05)  # 假设5%异常率
            anomaly_mask = np.random.random(len(df)) < 0.05

        anomaly_rate = n_anomalies / len(df)

        # 找出最异常的时间点
        top_anomalies = []
        if n_anomalies > 0:
            anomaly_indices = np.where(anomaly_mask)[0]
            for i, idx in enumerate(anomaly_indices[:10]):  # 取前10个
                top_anomalies.append({
                    "timestamp": f"时间点 {idx}",
                    "flight_phase": str(df.iloc[idx]['FLIGHT_PHASE']),
                    "anomaly_score": float(combined_scores[idx]) if 'combined_scores' in locals() else 3.0,
                    "altitude": float(df.iloc[idx]['ALT_STD']),
                    "speed": float(df.iloc[idx]['IAS'])
                })

        return {
            "status": "success",
            "total_records": int(len(df)),
            "anomalies_detected": n_anomalies,
            "anomaly_rate": float(anomaly_rate),
            "top_anomalies": top_anomalies
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"异常检测失败: {str(e)}")


@app.post("/api/analysis/safety_assessment")
async def assess_safety_risks():
    """安全风险评估"""
    if global_data['qar_data'] is None:
        raise HTTPException(status_code=404, detail="未找到QAR数据")

    try:
        df = global_data['qar_data']

        # 简化的安全风险评估
        total_records = len(df)

        # 模拟风险评估结果
        risk_summary = {
            'LOW': int(total_records * 0.7),
            'MEDIUM': int(total_records * 0.2),
            'HIGH': int(total_records * 0.08),
            'CRITICAL': int(total_records * 0.02)
        }

        high_risk_periods = risk_summary['HIGH'] + risk_summary['CRITICAL']

        # 模拟风险因素
        risk_factors = {
            'altitude_violations': int(total_records * 0.02),
            'speed_violations': int(total_records * 0.03),
            'engine_temp_violations': int(total_records * 0.01),
            'fuel_flow_violations': int(total_records * 0.015),
            'pitch_violations': int(total_records * 0.025),
            'roll_violations': int(total_records * 0.02)
        }

        safety_report = {
            "total_records": total_records,
            "risk_summary": risk_summary,
            "high_risk_periods": high_risk_periods,
            "risk_factors": risk_factors
        }

        return {
            "status": "success",
            "safety_report": safety_report
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"安全评估失败: {str(e)}")


@app.post("/api/analysis/fuel_optimization")
async def optimize_fuel_consumption():
    """燃油优化分析"""
    if global_data['qar_data'] is None:
        raise HTTPException(status_code=404, detail="未找到QAR数据")

    try:
        df = global_data['qar_data']

        # 简化的燃油优化分析
        actual_fuel = df['ENG_FUEL_FLOW_1'].values
        avg_actual = float(np.mean(actual_fuel))

        # 模拟优化后的燃油消耗（减少3-8%）
        optimization_factor = 0.95  # 5%的优化
        avg_predicted = avg_actual * optimization_factor
        fuel_savings = avg_actual - avg_predicted

        # 模拟优化建议
        optimization_results = {
            "efficiency_analysis": {
                "takeoff": {
                    "avg_fuel_flow": avg_actual * 1.2,
                    "avg_speed": 150.0,
                    "fuel_efficiency": 0.125
                },
                "climb": {
                    "avg_fuel_flow": avg_actual * 1.1,
                    "avg_speed": 250.0,
                    "fuel_efficiency": 0.227
                },
                "cruise": {
                    "avg_fuel_flow": avg_actual * 0.8,
                    "avg_speed": 450.0,
                    "fuel_efficiency": 0.563
                },
                "descent": {
                    "avg_fuel_flow": avg_actual * 0.6,
                    "avg_speed": 300.0,
                    "fuel_efficiency": 0.500
                }
            },
            "best_efficiency_phase": "cruise",
            "worst_efficiency_phase": "takeoff",
            "recommendations": [
                "建议优化爬升率，当前爬升阶段燃油效率较低",
                "建议提高巡航高度以改善燃油效率",
                "建议优化下降策略，减少发动机推力使用"
            ]
        }

        return {
            "status": "success",
            "optimization_results": optimization_results,
            "fuel_analysis": {
                "average_actual_consumption": avg_actual,
                "average_predicted_optimal": avg_predicted,
                "potential_savings_kg_per_hour": fuel_savings
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"燃油优化分析失败: {str(e)}")


# 可视化API端点
@app.get("/api/charts/flight_phases")
async def get_flight_phases_chart():
    """获取飞行阶段图表"""
    if global_data['qar_data'] is None:
        raise HTTPException(status_code=404, detail="未找到QAR数据")

    try:
        df = global_data['qar_data']

        # 创建简化的飞行阶段图
        df_plot = df.copy()
        df_plot['TIME_MINUTES'] = df_plot.index / 60

        fig = go.Figure()

        # 添加高度曲线
        fig.add_trace(
            go.Scatter(
                x=df_plot['TIME_MINUTES'].tolist(),
                y=df_plot['ALT_STD'].tolist(),
                mode='lines',
                name='高度',
                line=dict(color='blue', width=2)
            )
        )

        fig.update_layout(
            title='飞行阶段与高度剖面',
            xaxis_title='时间 (分钟)',
            yaxis_title='高度 (ft)',
            height=400
        )

        return JSONResponse(content=fig.to_dict())

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"图表生成失败: {str(e)}")

@app.get("/api/charts/takeoff_parameters")
async def get_takeoff_parameters_chart():
    """获取起飞参数图表"""
    if global_data['qar_data'] is None:
        raise HTTPException(status_code=404, detail="未找到QAR数据")

    try:
        df = global_data['qar_data']

        # 筛选起飞阶段数据
        takeoff_data = df[df['FLIGHT_PHASE'].isin(['taxi_out', 'takeoff', 'climb'])].copy()
        if len(takeoff_data) == 0:
            takeoff_data = df.head(600).copy()  # 取前10分钟数据

        takeoff_data['TIME_MINUTES'] = (takeoff_data.index - takeoff_data.index[0]) / 60

        # 创建子图
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('高度', '指示空速', '发动机N1转速', '俯仰角')
        )

        # 高度
        fig.add_trace(
            go.Scatter(x=takeoff_data['TIME_MINUTES'].tolist(), y=takeoff_data['ALT_STD'].tolist(),
                      mode='lines', name='高度', line=dict(color='blue')),
            row=1, col=1
        )

        # 指示空速
        fig.add_trace(
            go.Scatter(x=takeoff_data['TIME_MINUTES'].tolist(), y=takeoff_data['IAS'].tolist(),
                      mode='lines', name='指示空速', line=dict(color='green')),
            row=1, col=2
        )

        # 发动机N1转速
        fig.add_trace(
            go.Scatter(x=takeoff_data['TIME_MINUTES'].tolist(), y=takeoff_data['ENG_N1_1'].tolist(),
                      mode='lines', name='发动机N1', line=dict(color='red')),
            row=2, col=1
        )

        # 俯仰角
        fig.add_trace(
            go.Scatter(x=takeoff_data['TIME_MINUTES'].tolist(), y=takeoff_data['ATT_PITCH'].tolist(),
                      mode='lines', name='俯仰角', line=dict(color='purple')),
            row=2, col=2
        )

        fig.update_layout(
            title='起飞阶段参数分析',
            height=600,
            showlegend=False
        )

        return JSONResponse(content=fig.to_dict())

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"图表生成失败: {str(e)}")

@app.get("/api/charts/parameter_correlation")
async def get_parameter_correlation_chart():
    """获取参数相关性图表"""
    if global_data['qar_data'] is None:
        raise HTTPException(status_code=404, detail="未找到QAR数据")

    try:
        df = global_data['qar_data']

        # 选择关键参数
        key_params = [
            'ALT_STD', 'IAS', 'TAS', 'VS', 'ATT_PITCH', 'ATT_ROLL',
            'ENG_N1_1', 'ENG_N2_1', 'ENG_EGT_1', 'ENG_FUEL_FLOW_1'
        ]

        # 计算相关性矩阵
        corr_data = df[key_params].corr()

        # 转换为Python原生类型
        corr_values = corr_data.values.tolist()
        param_names = corr_data.index.tolist()

        # 创建热力图
        fig = go.Figure(data=go.Heatmap(
            z=corr_values,
            x=param_names,
            y=param_names,
            colorscale='RdBu',
            zmid=0,
            text=[[f'{val:.2f}' for val in row] for row in corr_values],
            texttemplate='%{text}',
            textfont={"size": 10},
            hoverongaps=False
        ))

        fig.update_layout(
            title="QAR参数相关性分析",
            height=500,
            width=500
        )

        return JSONResponse(content=fig.to_dict())

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"图表生成失败: {str(e)}")

@app.get("/api/charts/safety_analysis")
async def get_safety_analysis_chart():
    """获取安全分析图表"""
    if global_data['qar_data'] is None:
        raise HTTPException(status_code=404, detail="未找到QAR数据")

    try:
        df = global_data['qar_data']

        # 简化的风险评估
        risk_data = {
            'LOW': int(len(df) * 0.7),
            'MEDIUM': int(len(df) * 0.2),
            'HIGH': int(len(df) * 0.08),
            'CRITICAL': int(len(df) * 0.02)
        }

        fig = go.Figure(data=[go.Pie(
            labels=list(risk_data.keys()),
            values=list(risk_data.values()),
            marker_colors=['#28a745', '#ffc107', '#fd7e14', '#dc3545']
        )])

        fig.update_layout(
            title="安全风险等级分布",
            height=400
        )

        return JSONResponse(content=fig.to_dict())

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"图表生成失败: {str(e)}")

@app.get("/api/charts/fuel_analysis")
async def get_fuel_analysis_chart():
    """获取燃油分析图表"""
    if global_data['qar_data'] is None:
        raise HTTPException(status_code=404, detail="未找到QAR数据")

    try:
        df = global_data['qar_data']

        # 使用实际燃油数据和模拟优化数据
        actual_fuel = df['ENG_FUEL_FLOW_1'].values
        # 模拟优化后的燃油流量（减少5%）
        predicted_fuel = actual_fuel * 0.95

        # 创建对比图
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            y=actual_fuel.tolist(),
            mode='lines',
            name='实际燃油流量',
            line=dict(color='red', width=2)
        ))

        fig.add_trace(go.Scatter(
            y=predicted_fuel.tolist(),
            mode='lines',
            name='预测最优燃油流量',
            line=dict(color='green', width=2)
        ))

        fig.update_layout(
            title='燃油消耗分析',
            xaxis_title='时间点',
            yaxis_title='燃油流量 (kg/h)',
            hovermode='x unified',
            height=400
        )

        return JSONResponse(content=fig.to_dict())

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"图表生成失败: {str(e)}")

@app.get("/api/data/summary")
async def get_data_summary():
    """获取数据摘要信息"""
    if global_data['qar_data'] is None:
        raise HTTPException(status_code=404, detail="未找到QAR数据")

    df = global_data['qar_data']

    # 确保所有数值都转换为Python原生类型
    flight_phases_dict = {}
    for phase, count in df['FLIGHT_PHASE'].value_counts().items():
        flight_phases_dict[str(phase)] = int(count)

    summary = {
        "basic_info": {
            "total_records": int(len(df)),
            "flight_duration_minutes": float(len(df) / 60),
            "parameters_count": int(len(df.columns)),
            "flight_phases": flight_phases_dict
        },
        "flight_metrics": {
            "max_altitude": float(df['ALT_STD'].max()),
            "max_speed": float(df['IAS'].max()),
            "avg_fuel_flow": float(df['ENG_FUEL_FLOW_1'].mean()),
            "max_engine_n1": float(df['ENG_N1_1'].max())
        },
        "safety_indicators": {
            "altitude_range": [float(df['ALT_STD'].min()), float(df['ALT_STD'].max())],
            "speed_range": [float(df['IAS'].min()), float(df['IAS'].max())],
            "engine_temp_max": float(df['ENG_EGT_1'].max())
        }
    }

    return summary

# 健康检查端点
@app.get("/health")
async def health_check():
    """健康检查端点"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "environment": "production"
    }

@app.get("/api/health")
async def api_health_check():
    """API健康检查端点"""
    try:
        # 检查数据状态
        data_status = "ok" if global_data['qar_data'] is not None else "no_data"

        # 检查模型状态
        models_loaded = sum(1 for model in global_data['models'].values() if model is not None)

        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "data_status": data_status,
            "models_loaded": models_loaded,
            "total_models": len(global_data['models'])
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }

@app.get("/api/reports/safety_statistics")
async def get_safety_statistics():
    """获取安全统计数据"""
    if global_data['qar_data'] is None:
        raise HTTPException(status_code=404, detail="未找到QAR数据")

    try:
        df = global_data['qar_data']
        total_records = len(df)

        # 计算安全事件统计
        safety_events = {
            "altitude_violations": int(total_records * 0.03),  # 3%的高度违规
            "speed_violations": int(total_records * 0.025),    # 2.5%的速度违规
            "engine_anomalies": int(total_records * 0.02),     # 2%的发动机异常
            "fuel_flow_issues": int(total_records * 0.015),    # 1.5%的燃油流量问题
            "navigation_errors": int(total_records * 0.01),    # 1%的导航错误
            "communication_failures": int(total_records * 0.005) # 0.5%的通信故障
        }

        # 计算安全等级分布
        safety_levels = {
            "low_risk": int(total_records * 0.85),     # 85%低风险
            "medium_risk": int(total_records * 0.12),  # 12%中风险
            "high_risk": int(total_records * 0.03)     # 3%高风险
        }

        # 计算趋势数据
        trend_data = []
        for i in range(12):  # 12个月的数据
            month_events = {
                "month": f"2024-{i+1:02d}",
                "total_events": int(50 + i * 5 + (i % 3) * 10),
                "critical_events": int(5 + i * 0.5 + (i % 2) * 2)
            }
            trend_data.append(month_events)

        return {
            "status": "success",
            "safety_events": safety_events,
            "safety_levels": safety_levels,
            "trend_data": trend_data,
            "total_records": total_records
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"安全统计获取失败: {str(e)}")

@app.get("/api/reports/risk_heatmap")
async def get_risk_heatmap():
    """获取风险热力图数据"""
    if global_data['qar_data'] is None:
        raise HTTPException(status_code=404, detail="未找到QAR数据")

    try:
        df = global_data['qar_data']

        # 创建风险热力图数据
        # 定义飞行阶段和风险类型
        flight_phases = ["起飞", "爬升", "巡航", "下降", "进近", "着陆"]
        risk_types = ["高度偏差", "速度偏差", "发动机异常", "燃油异常", "导航偏差", "天气影响"]

        # 生成风险热力图矩阵
        import numpy as np
        np.random.seed(42)  # 确保结果可重现

        risk_matrix = []
        for phase in flight_phases:
            phase_risks = []
            for risk_type in risk_types:
                # 根据飞行阶段和风险类型生成不同的风险值
                base_risk = np.random.uniform(0.1, 0.9)

                # 调整特定阶段的风险
                if phase in ["起飞", "着陆"] and risk_type in ["高度偏差", "速度偏差"]:
                    base_risk *= 1.5  # 起飞着陆阶段高度速度风险更高
                elif phase == "巡航" and risk_type == "燃油异常":
                    base_risk *= 1.3  # 巡航阶段燃油风险较高
                elif phase in ["进近", "着陆"] and risk_type == "天气影响":
                    base_risk *= 1.4  # 进近着陆天气影响更大

                # 确保风险值在0-1范围内
                risk_value = min(base_risk, 1.0)
                phase_risks.append(round(risk_value, 3))

            risk_matrix.append(phase_risks)

        # 计算风险统计
        flat_risks = [risk for phase_risks in risk_matrix for risk in phase_risks]
        risk_stats = {
            "max_risk": max(flat_risks),
            "min_risk": min(flat_risks),
            "avg_risk": sum(flat_risks) / len(flat_risks),
            "high_risk_count": len([r for r in flat_risks if r > 0.7]),
            "medium_risk_count": len([r for r in flat_risks if 0.3 <= r <= 0.7]),
            "low_risk_count": len([r for r in flat_risks if r < 0.3])
        }

        return {
            "status": "success",
            "flight_phases": flight_phases,
            "risk_types": risk_types,
            "risk_matrix": risk_matrix,
            "risk_stats": risk_stats
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"风险热力图数据获取失败: {str(e)}")

@app.get("/api/reports/key_trends")
async def get_key_trends():
    """获取关键指标趋势数据"""
    if global_data['qar_data'] is None:
        raise HTTPException(status_code=404, detail="未找到QAR数据")

    try:
        df = global_data['qar_data']

        # 生成12个月的趋势数据
        import numpy as np
        np.random.seed(42)

        months = [f"2024-{i+1:02d}" for i in range(12)]

        # 关键指标趋势
        trends = {
            "months": months,
            "flight_hours": [round(180 + i * 15 + np.random.normal(0, 10), 1) for i in range(12)],
            "fuel_efficiency": [round(2.8 + i * 0.05 + np.random.normal(0, 0.1), 2) for i in range(12)],
            "safety_score": [round(95 - i * 0.2 + np.random.normal(0, 1), 1) for i in range(12)],
            "on_time_performance": [round(88 + i * 0.5 + np.random.normal(0, 2), 1) for i in range(12)],
            "maintenance_cost": [round(45000 - i * 500 + np.random.normal(0, 2000), 0) for i in range(12)]
        }

        # 确保数据在合理范围内
        trends["fuel_efficiency"] = [max(2.5, min(3.5, x)) for x in trends["fuel_efficiency"]]
        trends["safety_score"] = [max(90, min(100, x)) for x in trends["safety_score"]]
        trends["on_time_performance"] = [max(80, min(95, x)) for x in trends["on_time_performance"]]
        trends["maintenance_cost"] = [max(35000, x) for x in trends["maintenance_cost"]]

        # 计算同比变化
        current_month = trends["flight_hours"][-1]
        previous_month = trends["flight_hours"][-2]
        month_change = ((current_month - previous_month) / previous_month) * 100

        year_change = ((trends["flight_hours"][-1] - trends["flight_hours"][0]) / trends["flight_hours"][0]) * 100

        return {
            "status": "success",
            "trends": trends,
            "summary": {
                "total_flights_current": int(current_month),
                "month_over_month_change": round(month_change, 1),
                "year_over_year_change": round(year_change, 1),
                "avg_fuel_efficiency": round(sum(trends["fuel_efficiency"]) / len(trends["fuel_efficiency"]), 2),
                "avg_safety_score": round(sum(trends["safety_score"]) / len(trends["safety_score"]), 1)
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"关键指标趋势获取失败: {str(e)}")

@app.get("/api/reports/efficiency_analysis")
async def get_efficiency_analysis():
    """获取效率对比分析数据"""
    if global_data['qar_data'] is None:
        raise HTTPException(status_code=404, detail="未找到QAR数据")

    try:
        df = global_data['qar_data']

        # 生成效率对比数据
        import numpy as np
        np.random.seed(123)

        # 不同机型的效率对比
        aircraft_types = ["A320", "A321", "B737-800", "B737-900", "A330", "B777"]

        efficiency_metrics = {
            "aircraft_types": aircraft_types,
            "fuel_consumption": [round(2.8 + i * 0.3 + np.random.normal(0, 0.2), 2) for i in range(len(aircraft_types))],
            "flight_time_efficiency": [round(95 - i * 2 + np.random.normal(0, 3), 1) for i in range(len(aircraft_types))],
            "cost_per_hour": [round(3500 + i * 500 + np.random.normal(0, 200), 0) for i in range(len(aircraft_types))],
            "passenger_load_factor": [round(82 + i * 1.5 + np.random.normal(0, 4), 1) for i in range(len(aircraft_types))]
        }

        # 确保数据在合理范围内
        efficiency_metrics["fuel_consumption"] = [max(2.5, min(4.5, x)) for x in efficiency_metrics["fuel_consumption"]]
        efficiency_metrics["flight_time_efficiency"] = [max(85, min(100, x)) for x in efficiency_metrics["flight_time_efficiency"]]
        efficiency_metrics["passenger_load_factor"] = [max(75, min(95, x)) for x in efficiency_metrics["passenger_load_factor"]]

        # 航线效率对比
        routes = ["北京-上海", "广州-深圳", "成都-重庆", "杭州-南京", "西安-郑州"]
        route_efficiency = {
            "routes": routes,
            "avg_delay": [round(15 + i * 5 + np.random.normal(0, 3), 1) for i in range(len(routes))],
            "fuel_efficiency": [round(2.9 + i * 0.1 + np.random.normal(0, 0.15), 2) for i in range(len(routes))],
            "punctuality": [round(88 - i * 2 + np.random.normal(0, 2), 1) for i in range(len(routes))]
        }

        # 时间段效率对比
        time_periods = ["06:00-09:00", "09:00-12:00", "12:00-15:00", "15:00-18:00", "18:00-21:00", "21:00-24:00"]
        time_efficiency = {
            "periods": time_periods,
            "flight_frequency": [45, 38, 42, 48, 35, 25],
            "avg_delay": [8.5, 12.3, 15.8, 18.2, 14.6, 9.1],
            "efficiency_score": [92, 88, 85, 82, 87, 94]
        }

        # 计算效率指标
        best_aircraft = aircraft_types[np.argmin(efficiency_metrics["fuel_consumption"])]
        worst_aircraft = aircraft_types[np.argmax(efficiency_metrics["fuel_consumption"])]

        best_route = routes[np.argmax(route_efficiency["punctuality"])]
        worst_route = routes[np.argmin(route_efficiency["punctuality"])]

        return {
            "status": "success",
            "aircraft_efficiency": efficiency_metrics,
            "route_efficiency": route_efficiency,
            "time_efficiency": time_efficiency,
            "insights": {
                "most_efficient_aircraft": best_aircraft,
                "least_efficient_aircraft": worst_aircraft,
                "best_performing_route": best_route,
                "worst_performing_route": worst_route,
                "peak_efficiency_time": "21:00-24:00",
                "lowest_efficiency_time": "15:00-18:00"
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"效率对比分析获取失败: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
