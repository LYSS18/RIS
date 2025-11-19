"""
原始LSTM轨迹预测模型测试器
测试原始轨迹预测模型的性能并生成可视化结果
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import load_model

# 添加项目根目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

def load_test_data():
    """加载测试数据"""
    print("1. 加载测试数据...")
    
    # 加载数据
    data_path = os.path.join('..', '..', 'data', 'marathon_data_with_features.csv')
    if not os.path.exists(data_path):
        print(f"❌ 数据文件不存在: {data_path}")
        return None, None, None, None
    
    data = pd.read_csv(data_path)
    print(f"✅ 数据加载成功，共 {len(data)} 条记录")
    
    # 提取特征
    features = data[['latitude', 'longitude', 'speed', 'direction']].copy()
    
    # 数据归一化
    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(features)
    
    # 创建序列数据
    sequence_length = 10
    X, y = [], []
    
    for i in range(sequence_length, len(features_scaled)):
        X.append(features_scaled[i-sequence_length:i])
        y.append(features_scaled[i, :2])  # 只预测经纬度
    
    X, y = np.array(X), np.array(y)
    
    # 划分训练测试集
    train_size = int(len(X) * 0.8)
    testX = X[train_size:]
    testY = y[train_size:]
    
    print(f"✅ 测试数据准备完成")
    print(f"   测试集大小: {testX.shape}")
    
    return testX, testY, scaler, features

def smooth_predictions(predictions, window_size=5):
    """对预测结果进行移动平均平滑"""
    if len(predictions) < window_size:
        return predictions
    
    smoothed = np.copy(predictions)
    for i in range(window_size, len(predictions)):
        smoothed[i] = np.mean(predictions[i-window_size:i+1], axis=0)
    
    return smoothed

def calculate_metrics(true_coords, pred_coords):
    """计算评估指标"""
    # 计算欧几里得距离误差
    errors = []
    for i in range(len(true_coords)):
        lat_diff = (true_coords[i, 0] - pred_coords[i, 0]) * 111000  # 纬度转米
        lon_diff = (true_coords[i, 1] - pred_coords[i, 1]) * 111000 * np.cos(np.radians(true_coords[i, 0]))  # 经度转米
        error = np.sqrt(lat_diff**2 + lon_diff**2)
        errors.append(error)
    
    errors = np.array(errors)
    
    # 计算各种指标
    avg_error = np.mean(errors)
    median_error = np.median(errors)
    max_error = np.max(errors)
    std_error = np.std(errors)
    
    # 计算精度指标
    accuracy_30m = np.sum(errors <= 30) / len(errors) * 100
    accuracy_50m = np.sum(errors <= 50) / len(errors) * 100
    accuracy_80m = np.sum(errors <= 80) / len(errors) * 100
    accuracy_100m = np.sum(errors <= 100) / len(errors) * 100
    
    return {
        'errors': errors,
        'avg_error': avg_error,
        'median_error': median_error,
        'max_error': max_error,
        'std_error': std_error,
        'accuracy_30m': accuracy_30m,
        'accuracy_50m': accuracy_50m,
        'accuracy_80m': accuracy_80m,
        'accuracy_100m': accuracy_100m
    }

def calculate_direction_stability(coords):
    """计算方向稳定性"""
    if len(coords) < 3:
        return 0.0
    
    directions = []
    for i in range(1, len(coords)):
        lat_diff = coords[i, 0] - coords[i-1, 0]
        lon_diff = coords[i, 1] - coords[i-1, 1]
        direction = np.arctan2(lat_diff, lon_diff)
        directions.append(direction)
    
    # 计算方向变化的标准差
    direction_changes = []
    for i in range(1, len(directions)):
        change = abs(directions[i] - directions[i-1])
        # 处理角度跳跃
        if change > np.pi:
            change = 2*np.pi - change
        direction_changes.append(change)
    
    if len(direction_changes) == 0:
        return 1.0
    
    stability = 1.0 - (np.std(direction_changes) / np.pi)
    return max(0.0, stability)

def main():
    """主函数"""
    print("🔬 原始LSTM轨迹预测模型测试器")
    print("=" * 60)
    
    # 1. 加载测试数据
    testX, testY, scaler, features = load_test_data()
    if testX is None:
        return
    
    # 2. 加载原始模型
    print("\n2. 加载原始LSTM模型...")
    model_path = os.path.join('..', '..', 'models', 'traj_model_120.h5')
    
    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在: {model_path}")
        print("请先运行训练脚本生成模型文件")
        return
    
    try:
        model = load_model(model_path)
        print(f"✅ 原始模型加载成功: {model_path}")
        print(f"   模型结构: {model.summary()}")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return
    
    # 3. 模型预测
    print("\n3. 进行轨迹预测...")
    predicted = model.predict(testX, verbose=0)
    print(f"✅ 预测完成，预测了 {len(predicted)} 个点")
    
    # 4. 反归一化
    print("\n4. 反归一化处理...")
    coord_scaler = MinMaxScaler()
    coord_scaler.fit(features.iloc[:, :2])
    
    predicted_inverse = coord_scaler.inverse_transform(predicted)
    testY_inverse = coord_scaler.inverse_transform(testY)
    
    # 5. 应用平滑处理
    print("\n5. 应用轨迹平滑...")
    predicted_smooth = smooth_predictions(predicted_inverse, 5)
    
    # 6. 计算评估指标
    print("\n6. 计算评估指标...")
    
    # 原始预测指标
    metrics_orig = calculate_metrics(testY_inverse, predicted_inverse)
    
    # 平滑后指标
    metrics_smooth = calculate_metrics(testY_inverse, predicted_smooth)
    
    # 方向稳定性
    direction_stability_orig = calculate_direction_stability(predicted_inverse)
    direction_stability_smooth = calculate_direction_stability(predicted_smooth)
    
    # 7. 输出结果
    print("\n7. 测试结果分析:")
    print("=" * 60)
    
    print(f"\n📊 原始预测性能:")
    print(f"  - 平均误差: {metrics_orig['avg_error']:.2f}m")
    print(f"  - 中位数误差: {metrics_orig['median_error']:.2f}m")
    print(f"  - 最大误差: {metrics_orig['max_error']:.2f}m")
    print(f"  - 误差标准差: {metrics_orig['std_error']:.2f}m")
    print(f"  - 30m精度: {metrics_orig['accuracy_30m']:.2f}%")
    print(f"  - 50m精度: {metrics_orig['accuracy_50m']:.2f}%")
    print(f"  - 80m精度: {metrics_orig['accuracy_80m']:.2f}%")
    print(f"  - 100m精度: {metrics_orig['accuracy_100m']:.2f}%")
    print(f"  - 方向稳定性: {direction_stability_orig:.3f}")
    
    print(f"\n📈 平滑后性能:")
    print(f"  - 平均误差: {metrics_smooth['avg_error']:.2f}m")
    print(f"  - 中位数误差: {metrics_smooth['median_error']:.2f}m")
    print(f"  - 最大误差: {metrics_smooth['max_error']:.2f}m")
    print(f"  - 误差标准差: {metrics_smooth['std_error']:.2f}m")
    print(f"  - 30m精度: {metrics_smooth['accuracy_30m']:.2f}%")
    print(f"  - 50m精度: {metrics_smooth['accuracy_50m']:.2f}%")
    print(f"  - 80m精度: {metrics_smooth['accuracy_80m']:.2f}%")
    print(f"  - 100m精度: {metrics_smooth['accuracy_100m']:.2f}%")
    print(f"  - 方向稳定性: {direction_stability_smooth:.3f}")
    
    # 8. 性能评估
    print(f"\n🎯 性能评估:")
    if metrics_smooth['avg_error'] < 40:
        print("  - 高精度模型：误差小于40m ✅")
    elif metrics_smooth['avg_error'] < 60:
        print("  - 良好精度模型：误差小于60m")
    else:
        print("  - 需要改进：误差较大")
    
    if metrics_smooth['accuracy_50m'] > 80:
        print("  - 50m精度表现优秀 ✅")
    elif metrics_smooth['accuracy_50m'] > 70:
        print("  - 50m精度表现良好")
    
    if direction_stability_smooth > 0.8:
        print("  - 轨迹方向稳定性优秀 ✅")
    elif direction_stability_smooth > 0.7:
        print("  - 轨迹方向稳定性良好")
    
    # 9. 生成可视化结果
    print("\n9. 生成可视化结果...")
    
    # 创建轨迹对比图
    plt.figure(figsize=(12, 8))
    
    # 轨迹对比图 - 显示前1000个点以便清晰显示
    sample_size = min(1000, len(testY_inverse))
    plt.plot(testY_inverse[:sample_size, 1], testY_inverse[:sample_size, 0],
             'g-', label='真实轨迹', linewidth=3, alpha=0.8)
    plt.plot(predicted_smooth[:sample_size, 1], predicted_smooth[:sample_size, 0],
             'r-', label='预测轨迹', linewidth=2, alpha=0.8)
    
    # 添加起点和终点标记
    plt.plot(testY_inverse[0, 1], testY_inverse[0, 0], 'go', markersize=10, label='起点')
    plt.plot(testY_inverse[sample_size-1, 1], testY_inverse[sample_size-1, 0], 'rs', markersize=10, label='终点')
    
    plt.title(f'原始LSTM轨迹预测对比 (平均误差: {metrics_smooth["avg_error"]:.1f}m, 前{sample_size}点)', 
              fontsize=16, fontweight='bold')
    plt.xlabel('经度', fontsize=14)
    plt.ylabel('纬度', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    results_path = os.path.join('..', '..', 'results', 'original_model_results.png')
    plt.savefig(results_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ 测试完成！")
    print(f"📊 可视化结果已保存为 {results_path}")
    print("=" * 60)
    print("🎯 原始LSTM模型测试总结：")
    print(f"- 平均误差：{metrics_smooth['avg_error']:.2f}m")
    print(f"- 50m精度：{metrics_smooth['accuracy_50m']:.2f}%")
    print(f"- 80m精度：{metrics_smooth['accuracy_80m']:.2f}%")
    print(f"- 方向稳定性：{direction_stability_smooth:.3f}")
    print("=" * 60)
    
    if metrics_smooth['avg_error'] < 60 and metrics_smooth['accuracy_50m'] > 70:
        print("🏆 原始模型性能良好！")
    else:
        print("⚠️ 原始模型有改进空间")

if __name__ == "__main__":
    main()
