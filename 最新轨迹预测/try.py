
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
from haversine import haversine
import os

# 统一字体设置
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# 设置随机种子确保可复现性
np.random.seed(42)
import tensorflow as tf
tf.random.set_seed(42)

def refined_smooth_multipass(trajectory, passes=3, window=7):
    """多次平滑处理 - 核心平滑技术"""
    smoothed = np.copy(trajectory)

    for pass_num in range(passes):
        # 每次使用稍小的窗口：7→6→5
        current_window = max(3, window - pass_num)

        for i in range(len(smoothed)):
            start_idx = max(0, i - current_window // 2)
            end_idx = min(len(smoothed), i + current_window // 2 + 1)
            smoothed[i] = np.mean(smoothed[start_idx:end_idx], axis=0)

    return smoothed

def calculate_direction_stability(trajectory):
    """计算方向稳定性"""
    if len(trajectory) < 2:
        return 0

    directions = np.diff(trajectory, axis=0)
    direction_changes = []

    for i in range(len(directions) - 1):
        angle1 = np.arctan2(directions[i][1], directions[i][0])
        angle2 = np.arctan2(directions[i+1][1], directions[i+1][0])
        angle_diff = abs(angle2 - angle1)
        if angle_diff > np.pi:
            angle_diff = 2 * np.pi - angle_diff
        direction_changes.append(angle_diff)

    return np.mean(direction_changes) if direction_changes else 0

def main():

    # 1. 加载数据
    print("\n1. 加载数据...")
    data = pd.read_csv('..\\Processed\\Data\\001\\Trajectory\\20081024234405.csv', skiprows=1, header=None)
    data.columns = ['lat','lon','speed','angle']

    print("原始数据统计:")
    print(f"- 数据点数量: {len(data)}")
    print(f"- 纬度范围: {data['lat'].min():.6f} ~ {data['lat'].max():.6f}")
    print(f"- 经度范围: {data['lon'].min():.6f} ~ {data['lon'].max():.6f}")
    print(f"- 速度范围: {data['speed'].min():.2f} ~ {data['speed'].max():.2f} km/h")
    print(f"- 角度范围: {data['angle'].min():.2f} ~ {data['angle'].max():.2f} 度")

    # 2. 特征工程
    print("\n2. 特征工程...")
    # 速度异常值清理：使用95%分位数
    speed_95 = np.percentile(data['speed'], 95)
    data_clean = data.copy()
    data_clean['speed'] = np.clip(data_clean['speed'], 0, speed_95)

    # 构建特征矩阵
    features = data_clean[['lat', 'lon', 'speed', 'angle']]
    print(f'特征数据形状：{features.shape}')
    print("确认使用所有4个特征：")
    print("  - lat: 纬度坐标")
    print("  - lon: 经度坐标")
    print("  - speed: 清理后的速度")
    print("  - angle: 原始角度信息")

    # 3. 数据预处理
    print("\n3. 数据预处理...")

    # 归一化
    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(features)

    # 构造数据集（时间窗口4）
    time_step = 4  # 关键参数

    def create_dataset(dataset, time_step):
        X, Y = [], []
        for i in range(len(dataset) - time_step):
            X.append(dataset[i:(i + time_step), :])
            Y.append(dataset[i + time_step, :2])
        return np.array(X), np.array(Y)

    X, Y = create_dataset(features_scaled, time_step)

    testX = X  # 使用全部数据
    testY = Y  # 使用全部数据

    # 4. 加载训练好的模型
    print("\n4. 加载训练好的模型...")
    try:
        model = load_model('model.keras')
        print("精细平滑4特征双向LSTM模型加载成功")
    except Exception as e:
        print(f"模型加载失败: {e}")
        print("请先运行 lstm.py 训练模型")
        return

    # 5. 进行预测
    print("\n5. 进行预测...")
    predicted = model.predict(testX, verbose=0)
    print(f'预测结果形状：{predicted.shape}')

    # 6. 反归一化
    print("6. 反归一化...")
    coord_scaler = MinMaxScaler()
    coord_scaler.fit(features.iloc[:, :2])

    predicted_inverse = coord_scaler.inverse_transform(predicted)
    testY_inverse = coord_scaler.inverse_transform(testY)

    print(f'反归一化后形状: 预测={predicted_inverse.shape}, 真实={testY_inverse.shape}')

    # 7. 应用精细多次平滑
    print("\n7. 应用精细多次平滑...")
    predicted_smooth = refined_smooth_multipass(predicted_inverse, passes=3, window=7)
    print("多次平滑处理完成（3次迭代，递减窗口7→6→5）")

    # 8. 计算评估指标
    print("\n8. 计算评估指标...")

    # 平滑后预测误差
    errors_smooth = []
    for i in range(len(predicted_smooth)):
        pred_point = (predicted_smooth[i][0], predicted_smooth[i][1])
        true_point = (testY_inverse[i][0], testY_inverse[i][1])
        error = haversine(pred_point, true_point) * 1000
        errors_smooth.append(error)

    # 统计指标
    avg_error_smooth = np.mean(errors_smooth)
    median_error_smooth = np.median(errors_smooth)
    std_error_smooth = np.std(errors_smooth)
    max_error_smooth = np.max(errors_smooth)
    min_error_smooth = np.min(errors_smooth)

    accuracy_30m_smooth = sum(1 for e in errors_smooth if e <= 30) / len(errors_smooth) * 100
    accuracy_40m_smooth = sum(1 for e in errors_smooth if e <= 40) / len(errors_smooth) * 100
    accuracy_50m_smooth = sum(1 for e in errors_smooth if e <= 50) / len(errors_smooth) * 100
    accuracy_80m_smooth = sum(1 for e in errors_smooth if e <= 80) / len(errors_smooth) * 100
    accuracy_100m_smooth = sum(1 for e in errors_smooth if e <= 100) / len(errors_smooth) * 100

    # 方向稳定性
    direction_stability_smooth = calculate_direction_stability(predicted_smooth)

    # 9. 输出结果
    print("\n" + "="*80)
    print("轨迹预测模型测试结果")
    print("="*80)
    print(f"测试样本数量: {len(errors_smooth)}")

    print(f"\n精细平滑预测结果:")
    print(f"平均误差: {avg_error_smooth:.2f} 米")
    print(f"中位数误差: {median_error_smooth:.2f} 米")
    print(f"标准差: {std_error_smooth:.2f} 米")
    print(f"最大误差: {max_error_smooth:.2f} 米")
    print(f"最小误差: {min_error_smooth:.2f} 米")
    print(f"方向稳定性: {direction_stability_smooth:.4f}")

    print(f"\n不同阈值下的预测精度:")
    print(f"{'阈值':<10} {'精度':<10}")
    print("-" * 25)

    thresholds = [30, 40, 50, 80, 100]
    smooth_accuracies = [accuracy_30m_smooth, accuracy_40m_smooth, accuracy_50m_smooth, accuracy_80m_smooth, accuracy_100m_smooth]

    for i, threshold in enumerate(thresholds):
        smooth_acc = smooth_accuracies[i]
        print(f"{threshold}米内     {smooth_acc:<10.2f}%")


    # 10. 生成可视化结果
    print("\n10. 生成可视化结果...")

    # 创建图形
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # 轨迹对比图 - 显示所有点
    axes[0, 0].plot(testY_inverse[:, 1], testY_inverse[:, 0],
                    'g-', label='Ground Truth Trajectory', linewidth=3, alpha=0.8)
    axes[0, 0].plot(predicted_smooth[:, 1], predicted_smooth[:, 0],
                    'r-', label='Predicted Trajectory', linewidth=2, alpha=0.8)
    axes[0, 0].set_title(f'Trajectory Comparison', fontsize=14)
    axes[0, 0].set_xlabel('Longitude')
    axes[0, 0].set_ylabel('Latitude')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 误差分布图
    axes[0, 1].hist(errors_smooth, bins=50, alpha=0.7, color='lightblue', edgecolor='black')
    axes[0, 1].axvline(avg_error_smooth, color='red', linestyle='--', linewidth=2,
                       label=f'平均误差: {avg_error_smooth:.1f}m')
    axes[0, 1].axvline(50, color='green', linestyle='--', linewidth=2, label='50m目标')
    axes[0, 1].axvline(30, color='orange', linestyle='--', linewidth=2, label='30m阈值')
    axes[0, 1].set_title('预测误差分布', fontsize=14)
    axes[0, 1].set_xlabel('误差 (米)')
    axes[0, 1].set_ylabel('频次')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 精度柱状图
    colors = ['red', 'orange', 'lightgreen', 'green', 'darkgreen']
    bars = axes[0, 2].bar([f'{t}m' for t in thresholds], smooth_accuracies, color=colors, alpha=0.7)

    axes[0, 2].set_title(f'预测精度表现\n平均误差: {avg_error_smooth:.1f}m', fontsize=14)
    axes[0, 2].set_xlabel('误差阈值')
    axes[0, 2].set_ylabel('精度 (%)')
    axes[0, 2].set_ylim(0, 100)
    axes[0, 2].grid(True, alpha=0.3, axis='y')

    # 添加数值标签
    for bar, value in zip(bars, smooth_accuracies):
        height = bar.get_height()
        axes[0, 2].text(bar.get_x() + bar.get_width()/2., height + 1,
                       f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')

    # 方向变化图
    def calculate_direction_changes_plot(trajectory):
        if len(trajectory) < 2:
            return []
        directions = np.diff(trajectory, axis=0)
        changes = []
        for i in range(len(directions) - 1):
            angle1 = np.arctan2(directions[i][1], directions[i][0])
            angle2 = np.arctan2(directions[i+1][1], directions[i+1][0])
            angle_diff = abs(angle2 - angle1)
            if angle_diff > np.pi:
                angle_diff = 2 * np.pi - angle_diff
            changes.append(np.degrees(angle_diff))
        return changes

    # 使用所有点进行方向变化分析
    smooth_changes = calculate_direction_changes_plot(predicted_smooth)

    axes[1, 0].plot(smooth_changes, 'r-', label=f'方向稳定性: {direction_stability_smooth:.4f}', alpha=0.8)
    axes[1, 0].set_title(f'方向变化分析', fontsize=14)
    axes[1, 0].set_xlabel('时间步')
    axes[1, 0].set_ylabel('方向变化 (度)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 累积误差图
    cumulative_smooth = np.cumsum(errors_smooth)

    axes[1, 1].plot(cumulative_smooth, 'r-', label='累积误差', alpha=0.8)
    axes[1, 1].set_title('累积误差分析', fontsize=14)
    axes[1, 1].set_xlabel('预测点')
    axes[1, 1].set_ylabel('累积误差 (米)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    # 精细成果总结
    axes[1, 2].axis('off')
    summary_text = f"""

核心技术:
• 时间窗口4 + 多次平滑
• 全4特征利用
• 精细调优架构
    """

    axes[1, 2].text(0.05, 0.95, summary_text, transform=axes[1, 2].transAxes,
                    fontsize=11, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="gold", alpha=0.8))

    plt.tight_layout()
    plt.savefig('results.png', dpi=300, bbox_inches='tight')

    # 单独保存轨迹对比图（英语标签）
    plt.figure(figsize=(12, 8))

    # 显示所有真实轨迹点
    plt.plot(testY_inverse[:, 1], testY_inverse[:, 0],
             label='True Trajectory', color='green', linewidth=3, alpha=0.8)
    # 显示所有预测轨迹点
    plt.plot(predicted_smooth[:, 1], predicted_smooth[:, 0],
             label='Predicted Trajectory', color='red', linewidth=2, alpha=0.8)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.legend()
    plt.title('True Trajectory vs Predicted Trajectory')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('trajectory_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    plt.show()

    print("测试完成！")
    print("可视化结果已保存:")


if __name__ == "__main__":
    main()
