"""
QAR数据可视化模块
提供各种QAR数据的可视化功能
"""

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import folium
from folium import plugins


class QARVisualizer:
    """QAR数据可视化器"""
    
    def __init__(self, style: str = 'seaborn-v0_8'):
        """
        初始化可视化器
        
        Args:
            style: matplotlib样式
        """
        plt.style.use(style)
        sns.set_palette("husl")
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 飞行阶段颜色映射
        self.phase_colors = {
            'taxi_out': '#FFA500',
            'takeoff': '#FF4500', 
            'climb': '#32CD32',
            'cruise': '#4169E1',
            'descent': '#FFD700',
            'approach': '#FF6347',
            'landing': '#DC143C',
            'taxi_in': '#FFA500'
        }
    
    def plot_takeoff_parameters(self, df: pd.DataFrame, save_path: Optional[str] = None) -> go.Figure:
        """
        绘制起飞参数趋势图
        
        Args:
            df: QAR数据
            save_path: 保存路径
        """
        # 筛选起飞阶段数据
        takeoff_data = df[df['FLIGHT_PHASE'].isin(['taxi_out', 'takeoff', 'climb'])].copy()
        takeoff_data['TIME_MINUTES'] = (takeoff_data.index - takeoff_data.index[0]) / 60
        
        # 创建子图
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('高度', '指示空速', '发动机N1转速', '俯仰角', '燃油流量', '垂直速度'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # 高度
        fig.add_trace(
            go.Scatter(x=takeoff_data['TIME_MINUTES'], y=takeoff_data['ALT_STD'],
                      mode='lines', name='高度', line=dict(color='blue', width=2)),
            row=1, col=1
        )
        
        # 指示空速
        fig.add_trace(
            go.Scatter(x=takeoff_data['TIME_MINUTES'], y=takeoff_data['IAS'],
                      mode='lines', name='指示空速', line=dict(color='green', width=2)),
            row=1, col=2
        )
        
        # 发动机N1转速
        fig.add_trace(
            go.Scatter(x=takeoff_data['TIME_MINUTES'], y=takeoff_data['ENG_N1_1'],
                      mode='lines', name='发动机1 N1', line=dict(color='red', width=2)),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=takeoff_data['TIME_MINUTES'], y=takeoff_data['ENG_N1_2'],
                      mode='lines', name='发动机2 N1', line=dict(color='orange', width=2)),
            row=2, col=1
        )
        
        # 俯仰角
        fig.add_trace(
            go.Scatter(x=takeoff_data['TIME_MINUTES'], y=takeoff_data['ATT_PITCH'],
                      mode='lines', name='俯仰角', line=dict(color='purple', width=2)),
            row=2, col=2
        )
        
        # 燃油流量
        fig.add_trace(
            go.Scatter(x=takeoff_data['TIME_MINUTES'], y=takeoff_data['ENG_FUEL_FLOW_1'],
                      mode='lines', name='燃油流量', line=dict(color='brown', width=2)),
            row=3, col=1
        )
        
        # 垂直速度
        fig.add_trace(
            go.Scatter(x=takeoff_data['TIME_MINUTES'], y=takeoff_data['VS'],
                      mode='lines', name='垂直速度', line=dict(color='navy', width=2)),
            row=3, col=2
        )
        
        # 更新布局
        fig.update_layout(
            title='起飞阶段参数趋势分析',
            height=800,
            showlegend=True
        )
        
        # 更新坐标轴标签
        fig.update_xaxes(title_text="时间 (分钟)")
        fig.update_yaxes(title_text="高度 (ft)", row=1, col=1)
        fig.update_yaxes(title_text="空速 (kt)", row=1, col=2)
        fig.update_yaxes(title_text="N1转速 (%)", row=2, col=1)
        fig.update_yaxes(title_text="俯仰角 (°)", row=2, col=2)
        fig.update_yaxes(title_text="燃油流量 (kg/h)", row=3, col=1)
        fig.update_yaxes(title_text="垂直速度 (ft/min)", row=3, col=2)
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def plot_flight_phases(self, df: pd.DataFrame, save_path: Optional[str] = None) -> go.Figure:
        """
        绘制飞行阶段划分图
        
        Args:
            df: QAR数据
            save_path: 保存路径
        """
        df_plot = df.copy()
        df_plot['TIME_MINUTES'] = df_plot.index / 60
        
        fig = go.Figure()
        
        # 添加高度曲线
        fig.add_trace(
            go.Scatter(
                x=df_plot['TIME_MINUTES'],
                y=df_plot['ALT_STD'],
                mode='lines',
                name='高度',
                line=dict(color='black', width=2)
            )
        )
        
        # 为每个飞行阶段添加背景色
        phases = df_plot['FLIGHT_PHASE'].unique()
        for phase in phases:
            phase_data = df_plot[df_plot['FLIGHT_PHASE'] == phase]
            if len(phase_data) > 0:
                fig.add_vrect(
                    x0=phase_data['TIME_MINUTES'].min(),
                    x1=phase_data['TIME_MINUTES'].max(),
                    fillcolor=self.phase_colors.get(phase, '#CCCCCC'),
                    opacity=0.3,
                    layer="below",
                    line_width=0,
                    annotation_text=phase,
                    annotation_position="top left"
                )
        
        fig.update_layout(
            title='飞行阶段划分与高度剖面',
            xaxis_title='时间 (分钟)',
            yaxis_title='高度 (ft)',
            height=500
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig

    def plot_parameter_distributions(self, df: pd.DataFrame, save_path: Optional[str] = None) -> plt.Figure:
        """
        绘制参数分布分析图

        Args:
            df: QAR数据
            save_path: 保存路径
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('QAR参数分布分析', fontsize=16, fontweight='bold')

        # 发动机参数箱线图
        engine_params = ['ENG_N1_1', 'ENG_N1_2', 'ENG_EGT_1', 'ENG_FUEL_FLOW_1']
        engine_data = df[engine_params]
        axes[0, 0].boxplot([engine_data[col].dropna() for col in engine_params],
                          labels=['N1_1', 'N1_2', 'EGT_1', 'FUEL_1'])
        axes[0, 0].set_title('发动机参数箱线图')
        axes[0, 0].set_ylabel('参数值')
        axes[0, 0].grid(True, alpha=0.3)

        # 燃油流量直方图
        axes[0, 1].hist(df['ENG_FUEL_FLOW_1'].dropna(), bins=50, alpha=0.7, color='orange', edgecolor='black')
        axes[0, 1].set_title('燃油流量分布')
        axes[0, 1].set_xlabel('燃油流量 (kg/h)')
        axes[0, 1].set_ylabel('频次')
        axes[0, 1].grid(True, alpha=0.3)

        # 垂直速度分布
        axes[1, 0].hist(df['VS'].dropna(), bins=50, alpha=0.7, color='green', edgecolor='black')
        axes[1, 0].set_title('垂直速度分布')
        axes[1, 0].set_xlabel('垂直速度 (ft/min)')
        axes[1, 0].set_ylabel('频次')
        axes[1, 0].grid(True, alpha=0.3)

        # 空速-高度关系散点图
        sample_data = df.sample(n=min(1000, len(df)))  # 采样以提高性能
        scatter = axes[1, 1].scatter(sample_data['IAS'], sample_data['ALT_STD'],
                                   c=sample_data.index, cmap='viridis', alpha=0.6)
        axes[1, 1].set_title('空速-高度关系')
        axes[1, 1].set_xlabel('指示空速 (kt)')
        axes[1, 1].set_ylabel('高度 (ft)')
        axes[1, 1].grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=axes[1, 1], label='时间序列')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def plot_correlation_heatmap(self, df: pd.DataFrame, save_path: Optional[str] = None) -> plt.Figure:
        """
        绘制参数相关性热力图

        Args:
            df: QAR数据
            save_path: 保存路径
        """
        # 选择关键数值参数
        key_params = [
            'ALT_STD', 'IAS', 'TAS', 'VS', 'ATT_PITCH', 'ATT_ROLL',
            'ENG_N1_1', 'ENG_N2_1', 'ENG_EGT_1', 'ENG_FUEL_FLOW_1',
            'CTRL_ELEV', 'CTRL_AIL', 'WIND_SPEED'
        ]

        # 计算相关性矩阵
        corr_data = df[key_params].corr()

        # 创建热力图
        fig, ax = plt.subplots(figsize=(12, 10))

        # 绘制热力图
        sns.heatmap(
            corr_data,
            annot=True,
            cmap='RdBu_r',
            center=0,
            square=True,
            fmt='.2f',
            cbar_kws={'label': '相关系数'},
            ax=ax
        )

        ax.set_title('QAR关键参数相关性分析', fontsize=16, fontweight='bold', pad=20)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def create_dashboard(self, df: pd.DataFrame, save_dir: str = './visualizations/') -> Dict[str, str]:
        """
        创建完整的可视化仪表板

        Args:
            df: QAR数据
            save_dir: 保存目录
        """
        import os
        os.makedirs(save_dir, exist_ok=True)

        dashboard_files = {}

        # 1. 起飞参数趋势图
        takeoff_fig = self.plot_takeoff_parameters(df)
        takeoff_path = os.path.join(save_dir, 'takeoff_parameters.html')
        takeoff_fig.write_html(takeoff_path)
        dashboard_files['takeoff_parameters'] = takeoff_path

        # 2. 飞行阶段划分图
        phases_fig = self.plot_flight_phases(df)
        phases_path = os.path.join(save_dir, 'flight_phases.html')
        phases_fig.write_html(phases_path)
        dashboard_files['flight_phases'] = phases_path

        # 3. 参数分布分析
        dist_fig = self.plot_parameter_distributions(df)
        dist_path = os.path.join(save_dir, 'parameter_distributions.png')
        dist_fig.savefig(dist_path, dpi=300, bbox_inches='tight')
        plt.close(dist_fig)
        dashboard_files['parameter_distributions'] = dist_path

        # 4. 相关性热力图
        corr_fig = self.plot_correlation_heatmap(df)
        corr_path = os.path.join(save_dir, 'correlation_heatmap.png')
        corr_fig.savefig(corr_path, dpi=300, bbox_inches='tight')
        plt.close(corr_fig)
        dashboard_files['correlation_heatmap'] = corr_path

        print(f"可视化仪表板已保存到: {save_dir}")
        return dashboard_files
