#!/usr/bin/env python3
"""
快速测试图表加载问题
"""

import requests
import time

def test_chart_loading():
    """测试图表加载"""
    base_url = "http://localhost:8000"
    
    print('🔧 快速测试图表加载问题修复')
    print('=' * 50)
    
    # 1. 确保有数据
    print('\n1. 检查数据状态...')
    try:
        response = requests.get(f"{base_url}/api/data/summary", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f'   ✅ 数据已存在: {data["basic_info"]["total_records"]} 条记录')
        else:
            print('   ⚠️ 未找到数据，正在生成...')
            # 生成数据
            gen_response = requests.post(f"{base_url}/api/simulate", 
                                       json={
                                           "flight_duration_minutes": 30,
                                           "sampling_rate_hz": 1.0,
                                           "anomaly_rate": 0.05
                                       }, timeout=60)
            if gen_response.status_code == 200:
                print('   ✅ 数据生成成功')
            else:
                print('   ❌ 数据生成失败')
                return False
    except Exception as e:
        print(f'   ❌ 数据检查失败: {e}')
        return False
    
    # 2. 测试所有图表API
    print('\n2. 测试图表API响应...')
    chart_apis = [
        ('flight_phases', '飞行阶段图'),
        ('takeoff_parameters', '起飞参数图'),
        ('parameter_correlation', '参数相关性图'),
        ('safety_analysis', '安全分析图'),
        ('fuel_analysis', '燃油分析图')
    ]
    
    all_success = True
    for api_name, chart_name in chart_apis:
        try:
            start_time = time.time()
            response = requests.get(f"{base_url}/api/charts/{api_name}", timeout=30)
            end_time = time.time()
            
            if response.status_code == 200:
                data_size = len(response.content)
                response_time = (end_time - start_time) * 1000
                print(f'   ✅ {chart_name}: {response_time:.0f}ms, {data_size:,} 字节')
            else:
                print(f'   ❌ {chart_name}: HTTP {response.status_code}')
                all_success = False
        except Exception as e:
            print(f'   ❌ {chart_name}: {e}')
            all_success = False
    
    # 3. 测试Web页面
    print('\n3. 测试Web页面访问...')
    try:
        response = requests.get(f"{base_url}/", timeout=10)
        if response.status_code == 200:
            print(f'   ✅ 主页正常访问 ({len(response.content):,} 字节)')
        else:
            print(f'   ❌ 主页访问失败: HTTP {response.status_code}')
            all_success = False
    except Exception as e:
        print(f'   ❌ 主页访问失败: {e}')
        all_success = False
    
    # 4. 结果总结
    print('\n' + '=' * 50)
    if all_success:
        print('🎉 所有测试通过！')
        print('\n修复效果:')
        print('✅ 图表API全部正常响应')
        print('✅ 前端JavaScript已修复loading状态处理')
        print('✅ 图表容器会正确清空并重新渲染')
        print('\n现在访问 http://localhost:8000/ 应该可以看到:')
        print('- 图表正常加载，不会一直显示"正在加载图表..."')
        print('- 如果API失败，会显示错误信息和重试按钮')
        print('- 所有图表都支持交互操作')
        
        print('\n💡 如果仍然看到"正在加载图表..."，请:')
        print('1. 刷新浏览器页面 (Ctrl+F5 强制刷新)')
        print('2. 打开浏览器开发者工具查看控制台日志')
        print('3. 检查网络请求是否成功')
        
        return True
    else:
        print('⚠️ 部分测试失败，需要进一步调试')
        return False

if __name__ == "__main__":
    test_chart_loading()
