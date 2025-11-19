
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys
import json
from datetime import datetime


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


try:
    from utils.distance_calculator import DistanceCalculator
except ImportError:

    class DistanceCalculator:
        def calculate_distance(self, lat1, lon1, lat2, lon2):
            # 简单的欧几里得距离计算（米）
            return ((lat1-lat2)**2 + (lon1-lon2)**2)**0.5 * 111000

class PredictionErrorAnalyzer:
    def __init__(self):
        """
        初始化轨迹预测误差分析器
        """
        self.distance_calc = DistanceCalculator()

        # 关键参数 - Critical Time Windows for Life Safety
        self.critical_times = {
            'cardiac': 2.0,      # T_critical,cardiac = 2秒
            'thermal': 3.0,      # T_critical,thermal = 3秒
            'arrhythmia': 1.0    # T_critical,arrhythmia = 1秒
        }

        # 风险模型参数 - Life Safety Risk Quantification
        self.lambda_cardiac = 0.2  # λ_cardiac = 0.2 s^-1 (心脏事件恶化率参数)
        self.thermal_risk_coeff = 0.05  # 热风险系数 (每秒5%增加)

        # 权重分配 - Comprehensive Safety Risk Index
        self.weights = {
            'cardiac': 0.7,    # w_cardiac = 0.7
            'thermal': 0.3     # w_thermal = 0.3
        }

        # 通信参数
        self.sinr_threshold = 10.0  # SINR_threshold (dB)
        self.base_sampling_period = 1.0  # T_base (秒)
        self.max_retransmissions = 3     # N_retrans

        # 路径损耗模型参数
        self.path_loss_exponent = 2.0  # α (路径损耗指数)
        self.base_distance_ub = 100.0   # d_UB 基准距离 (米)
        self.base_distance_ur = 50.0    # d_UR 基准距离 (米)
        self.distance_rb = 200.0        # d_RB 距离 (米)

        # 功率参数 (调整以在更大误差范围内体现变化趋势)
        self.gain_constant_a = 1e-6     # A (增益常数) - 增大以产生更明显的SINR变化
        self.gain_constant_b = 1e-14    # B (增益常数) - 调整干扰级别
        self.proportionality_k = 1e-12  # K (比例常数) - 增大角度影响
        
    def calculate_power_change_due_to_prediction_error(self, delta_d1, delta_d2):
        """
        计算由于预测误差导致的功率变化
        """
        alpha = self.path_loss_exponent  # α
        A = self.gain_constant_a         # 增益常数A
        B = self.gain_constant_b         # 增益常数B
        d_UB = self.base_distance_ub     # 用户到基站距离
        d_UR = self.base_distance_ur     # 用户到RIS距离
        d_RB = self.distance_rb          # RIS到基站距离

        # 原始功率
        # P_sig = A / d_UB^α
        # P_int = B / (d_UR · d_RB)^α
        P_sig_original = A / (d_UB ** alpha)
        P_int_original = B / ((d_UR * d_RB) ** alpha)

        # P_total = P_sig + P_int
        P_total_original = P_sig_original + P_int_original

        # 预测误差导致的距离变化
        # d_UB' = d_UB + Δd_1
        # d_UR' = d_UR + Δd_2
        d_UB_predicted = d_UB + delta_d1
        d_UR_predicted = d_UR + delta_d2

        # 确保距离为正值
        d_UB_predicted = max(d_UB_predicted, 1.0)
        d_UR_predicted = max(d_UR_predicted, 1.0)

        #  预测后的功率
        P_sig_predicted = A / (d_UB_predicted ** alpha)
        P_int_predicted = B / ((d_UR_predicted * d_RB) ** alpha)
        P_total_predicted = P_sig_predicted + P_int_predicted

        # 功率变化
        # ΔP_total = P_total' - P_total
        delta_P_total = P_total_predicted - P_total_original
        delta_P_sig = P_sig_predicted - P_sig_original
        delta_P_int = P_int_predicted - P_int_original

        return {
            'P_sig_original': P_sig_original,
            'P_int_original': P_int_original,
            'P_total_original': P_total_original,
            'P_sig_predicted': P_sig_predicted,
            'P_int_predicted': P_int_predicted,
            'P_total_predicted': P_total_predicted,
            'delta_P_sig': delta_P_sig,
            'delta_P_int': delta_P_int,
            'delta_P_total': delta_P_total,
            'd_UB_original': d_UB,
            'd_UR_original': d_UR,
            'd_UB_predicted': d_UB_predicted,
            'd_UR_predicted': d_UR_predicted
        }

    def calculate_angular_interference_error(self, theta_original, delta_theta):
        """
        计算角度预测误差对干扰的影响
        """
        # 确保角度为正值，避免除零错误
        theta_original = max(theta_original, 0.1)  # 最小0.1弧度 ≈ 5.7度
        theta_predicted = max(theta_original + delta_theta, 0.1)

        #  f(θ) = 10/θ
        f_original = 10.0 / theta_original
        f_predicted = 10.0 / theta_predicted

        #  Δf = f(θ + Δθ) - f(θ)
        delta_f = f_predicted - f_original

        # ΔP_int = K × Δf
        delta_P_int = self.proportionality_k * delta_f

        return {
            'theta_original': theta_original,
            'theta_predicted': theta_predicted,
            'f_original': f_original,
            'f_predicted': f_predicted,
            'delta_f': delta_f,
            'delta_P_int': delta_P_int
        }

    def calculate_sinr_degradation(self, distance_error, angle_error_deg=5.0):
        """
        计算SINR降级
        """
        if distance_error == 0 and angle_error_deg == 0:
            return 0.0

        # 预测误差导致的距离变化
        delta_d1 = distance_error * 0.6  # d_UB的误差分量
        delta_d2 = distance_error * 0.4  # d_UR的误差分量

        # 计算功率变化
        power_result = self.calculate_power_change_due_to_prediction_error(delta_d1, delta_d2)

        # 计算角度干扰误差
        theta_original = 30.0  # 假设原始角度30度
        delta_theta = angle_error_deg  # 角度误差（度）
        angular_result = self.calculate_angular_interference_error(theta_original, delta_theta)

        # 噪声功率
        P_noise = 1e-12  # 1 pW

        # 原始SINR = P_sig / (P_int + P_noise)
        P_sig_original = power_result['P_sig_original']
        P_int_original = power_result['P_int_original']
        sinr_original = P_sig_original / (P_int_original + P_noise)

        # 预测后的SINR (包含角度误差影响)
        # P_sig' = A / (d_UB + Δd_1)^α
        P_sig_predicted = power_result['P_sig_predicted']
        # P_int' = P_int + ΔP_int
        # 注意：如果ΔP_int为负值，说明角度误差实际上减少了干扰
        # 但从物理角度，角度误差通常应该增加干扰，所以我们使用绝对值
        P_int_predicted = power_result['P_int_predicted'] + abs(angular_result['delta_P_int'])
        sinr_predicted = P_sig_predicted / (P_int_predicted + P_noise)

        # 公式: ΔSINR = SINR' - SINR
        if sinr_predicted > 0 and sinr_original > 0:
            if sinr_original > sinr_predicted:
                # SINR下降，计算降级 (正值表示性能下降)
                sinr_degradation_db = 10 * np.log10(sinr_original / sinr_predicted)
            else:
                # SINR提升，降级为0
                sinr_degradation_db = 0.0
        else:
            # 备用计算方法：基于距离误差的路径损耗模型
            if distance_error > 0:
                alpha = self.path_loss_exponent
                base_distance = self.base_distance_ub
                distance_ratio = (base_distance + distance_error) / base_distance
                path_loss_change = distance_ratio ** alpha
                angle_factor = 1.0 + (angle_error_deg / 180.0) * 0.1
                sinr_degradation_db = 10 * np.log10(path_loss_change * angle_factor)
            else:
                sinr_degradation_db = 0.0

        # 确保结果在扩展的合理范围内 (0-50 dB) - 允许更大变化以体现趋势
        sinr_degradation_db = max(0.0, min(sinr_degradation_db, 50.0))

        # 确保结果是有限的
        if not np.isfinite(sinr_degradation_db):
            sinr_degradation_db = 0.0

        return sinr_degradation_db
    
    def sinr_to_packet_error_rate(self, sinr_degradation):
        """
        基于SINR计算数据包错误率
        公式: PER(SINR) = 0.5 × exp(-ln(2) × SINR/SINR_threshold)
        注意：这里的SINR应该是实际的SINR值，不是降级值
        """
        # 计算实际SINR (dB) = 基准SINR - SINR降级
        # 假设基准SINR为SINR_threshold
        actual_sinr_db = self.sinr_threshold - sinr_degradation

        if actual_sinr_db <= 0:
            return 0.5  # 最大错误率

        # PER公式: PER(SINR) = 0.5 × exp(-ln(2) × SINR/SINR_threshold)
        per = 0.5 * np.exp(-np.log(2) * actual_sinr_db / self.sinr_threshold)
        return min(per, 0.5)
    
    def calculate_monitoring_delay(self, per):
        """
        计算生理监测延迟
        公式: T_delay = T_base × (1 + N_retrans × PER)
        """
        delay = self.base_sampling_period * (1 + self.max_retransmissions * per)
        return delay
    
    def calculate_cardiac_risk(self, delay):
        """
        计算心脏事件风险
        公式: P_cardiac,miss(T_delay) = 1 - exp(-λ_cardiac × ΔT_cardiac)
        其中: ΔT_cardiac = max(0, T_delay - T_critical,cardiac)
        """
        delta_t_cardiac = max(0, delay - self.critical_times['cardiac'])
        risk = 1 - np.exp(-self.lambda_cardiac * delta_t_cardiac)
        return risk

    def calculate_thermal_risk(self, delay):
        """
        计算热相关风险
        公式: P_thermal,death(T_delay) = 0.05 × max(0, T_delay - T_critical,thermal)
        """
        delta_t_thermal = max(0, delay - self.critical_times['thermal'])
        risk = self.thermal_risk_coeff * delta_t_thermal
        return min(risk, 1.0)  # 风险不超过100%

    def calculate_comprehensive_safety_risk(self, cardiac_risk, thermal_risk):
        """
        计算综合安全风险指数
        公式: R_safety(ΔSINR) = w_cardiac × P1 + w_thermal × P2
        其中: P1 = P_cardiac,miss(ΔSINR), P2 = P_thermal,death(ΔSINR)
        """
        safety_risk = (self.weights['cardiac'] * cardiac_risk +
                      self.weights['thermal'] * thermal_risk)
        return safety_risk
    
    def analyze_prediction_error_impact(self, distance_errors, angle_errors=None):
        """
        分析轨迹预测误差对健康风险的影响
        """
        if angle_errors is None:
            angle_errors = [5.0] * len(distance_errors)  # 默认5度角度误差

        results = {
            'distance_errors': distance_errors,
            'angle_errors': angle_errors,
            'sinr_degradations': [],
            'packet_error_rates': [],
            'monitoring_delays': [],
            'cardiac_risks': [],
            'thermal_risks': [],
            'comprehensive_risks': [],
            'power_analysis': []
        }

        for i, error in enumerate(distance_errors):
            angle_error = angle_errors[i] if i < len(angle_errors) else 5.0

            # 1. 轨迹预测误差 → SINR降级
            sinr_deg = self.calculate_sinr_degradation(error, angle_error)
            results['sinr_degradations'].append(sinr_deg)

            # 2. SINR降级 → 数据包错误率
            per = self.sinr_to_packet_error_rate(sinr_deg)
            results['packet_error_rates'].append(per)

            # 3. 数据包错误率 → 生理监测延迟
            delay = self.calculate_monitoring_delay(per)
            results['monitoring_delays'].append(delay)

            # 4. 监测延迟 → 生命安全风险
            cardiac_risk = self.calculate_cardiac_risk(delay)
            thermal_risk = self.calculate_thermal_risk(delay)
            comprehensive_risk = self.calculate_comprehensive_safety_risk(cardiac_risk, thermal_risk)

            results['cardiac_risks'].append(cardiac_risk)
            results['thermal_risks'].append(thermal_risk)
            results['comprehensive_risks'].append(comprehensive_risk)

            # 5. 功率分析详情 (用于调试和验证)
            power_result = self.calculate_power_change_due_to_prediction_error(
                error * 0.6, error * 0.4
            )
            results['power_analysis'].append(power_result)

        return results

def create_prediction_error_analysis_plots(results, save_dir):
    """
    创建轨迹预测误差分析图表
    基于Error Analysis部分的误差传播分析
    """

    # 设置英文字体
    plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial']
    plt.rcParams['axes.unicode_minus'] = False

    # 创建自定义布局：左边一个大图，右边两个小图
    fig = plt.figure(figsize=(18, 10))

    # 主标题
    fig.suptitle('Impact Analysis of Trajectory Prediction Errors on Athlete Health Indicators',
                 fontsize=16, fontweight='bold')

    distance_errors = results['distance_errors']

    # 左边的合并图 - 显示四个指标
    ax_left = plt.subplot2grid((2, 3), (0, 0), colspan=2, rowspan=2)

    # 绘制四条曲线在同一个图上
    ax1 = ax_left
    ax2 = ax1.twinx()
    ax3 = ax1.twinx()
    ax4 = ax1.twinx()

    # 调整右侧轴的位置
    ax3.spines['right'].set_position(('outward', 60))
    ax4.spines['right'].set_position(('outward', 120))

    # 绘制四条曲线，使用不同线型避免重合
    line1 = ax1.plot(distance_errors, results['sinr_degradations'], 'b-', linewidth=2, marker='o', markersize=4, label='SINR Degradation (dB)')
    line2 = ax2.plot(distance_errors, np.array(results['packet_error_rates'])*100, 'r--', linewidth=2, marker='s', markersize=4, label='Packet Error Rate (%)')
    line3 = ax3.plot(distance_errors, results['monitoring_delays'], 'g:', linewidth=3, marker='^', markersize=4, label='Monitoring Delay (s)')
    line4 = ax4.plot(distance_errors, np.array(results['cardiac_risks'])*100, color='purple', linestyle='-.', linewidth=2, marker='d', markersize=4, label='Cardiac Risk (%)')

    # 设置轴标签和颜色
    ax1.set_xlabel('Trajectory Prediction Distance Error (m)', fontsize=12)
    ax1.set_ylabel('SINR Degradation (dB)', color='b', fontsize=12)
    ax2.set_ylabel('Packet Error Rate (%)', color='r', fontsize=12)
    ax3.set_ylabel('Monitoring Delay (s)', color='g', fontsize=12)
    ax4.set_ylabel('Cardiac Risk (%)', color='purple', fontsize=12)

    # 设置轴颜色
    ax1.tick_params(axis='y', labelcolor='b')
    ax2.tick_params(axis='y', labelcolor='r')
    ax3.tick_params(axis='y', labelcolor='g')
    ax4.tick_params(axis='y', labelcolor='purple')

    # 设置不同的Y轴范围以更好地区分曲线
    ax1.set_ylim(0, 15)  # SINR降级: 0-15 dB
    ax2.set_ylim(20, 55)  # PER: 20-55%
    ax3.set_ylim(1.5, 2.8)  # 延迟: 1.5-2.8s
    ax4.set_ylim(0, 12)  # 心脏风险: 0-12%

    ax1.set_title('Error Propagation Chain: Distance Error → SINR → PER → Delay → Risk', fontsize=14, pad=20)
    ax1.grid(True, alpha=0.3)

    # 合并图例
    lines = line1 + line2 + line3 + line4
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left', bbox_to_anchor=(0, 1))

    # 右上图 - 生理监测延迟（带阈值线）
    ax_top_right = plt.subplot2grid((2, 3), (0, 2))
    ax_top_right.plot(distance_errors, results['monitoring_delays'], 'g-', linewidth=2, marker='^')
    ax_top_right.axhline(y=2.0, color='red', linestyle='--', alpha=0.7, label='Cardiac Critical Time = 2s')
    ax_top_right.axhline(y=3.0, color='orange', linestyle='--', alpha=0.7, label='Thermal Critical Time = 3s')
    ax_top_right.set_xlabel('Trajectory Prediction Distance Error (m)')
    ax_top_right.set_ylabel('Physiological Monitoring Delay (s)')
    ax_top_right.set_title('Impact of Data Transmission Delay on Physiological Monitoring')
    ax_top_right.legend(fontsize=10)
    ax_top_right.grid(True, alpha=0.3)

    # 右下图 - 综合安全风险指数（带安全阈值线）
    ax_bottom_right = plt.subplot2grid((2, 3), (1, 2))
    ax_bottom_right.plot(distance_errors, np.array(results['comprehensive_risks'])*100, 'red', linewidth=3, marker='o')
    ax_bottom_right.axhline(y=5.0, color='red', linestyle='--', alpha=0.7, label='Safety Threshold = 5%')
    ax_bottom_right.set_xlabel('Trajectory Prediction Distance Error (m)')
    ax_bottom_right.set_ylabel('Comprehensive Safety Risk Index (%)')
    ax_bottom_right.set_title('Impact of Trajectory Prediction Errors on Athlete Life Safety')
    ax_bottom_right.legend(fontsize=10)
    ax_bottom_right.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图表
    plot_path = os.path.join(save_dir, 'prediction_error_analysis.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f" 轨迹预测误差分析图表已保存: {plot_path}")

    return plot_path

def main():

    # 创建保存目录
    save_dir = "."  # 当前目录就是test_exp
    os.makedirs(save_dir, exist_ok=True)

    # 初始化分析器
    analyzer = PredictionErrorAnalyzer()

    # 定义轨迹预测误差范围 - 扩大范围以体现变化趋势
    distance_errors = np.linspace(0, 500, 101)  # 0到500米，101个点
    angle_errors = np.linspace(1, 30, 101)      # 1到30度角度误差

    print(f" 分析轨迹预测距离误差范围: 0-500米 ({len(distance_errors)}个采样点)")
    print(f" 分析轨迹预测角度误差范围: 1-30度 ({len(angle_errors)}个采样点)")
    print(f" 扩大误差范围以充分体现误差传播趋势")

    # 进行误差分析
    print(" 正在计算误差传播链条...")
    print("   距离/角度误差 → SINR降级 → PER → 监测延迟 → 生命安全风险")
    results = analyzer.analyze_prediction_error_impact(distance_errors, angle_errors)

    # 创建图表
    print(" 正在生成误差分析图表...")
    plot_path = create_prediction_error_analysis_plots(results, save_dir)
    
    # 保存数值结果
    results_data = {
        'analysis_time': datetime.now().isoformat(),
        'analysis_type': 'trajectory_prediction_error_analysis',
        'paper_section': 'Error Analysis',
        'parameters': {
            'critical_times': analyzer.critical_times,
            'lambda_cardiac': analyzer.lambda_cardiac,
            'thermal_risk_coeff': analyzer.thermal_risk_coeff,
            'weights': analyzer.weights,
            'sinr_threshold': analyzer.sinr_threshold,
            'path_loss_exponent': analyzer.path_loss_exponent,
            'base_distances': {
                'UB': analyzer.base_distance_ub,
                'UR': analyzer.base_distance_ur,
                'RB': analyzer.distance_rb
            }
        },
        'results': {k: [float(v) if isinstance(v, (int, float, np.number)) else v
                       for v in values] for k, values in results.items()
                   if k != 'power_analysis'}  # 排除复杂的功率分析数据
    }

    results_path = os.path.join(save_dir, 'prediction_error_analysis_results.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, indent=2, ensure_ascii=False)

    print(f"  误差分析数值结果已保存: {results_path}")
    
    # 关键发现
    print("\n" + "=" * 70)
    print(" 误差分析关键发现:")

    # 找到5%风险阈值对应的距离误差 (安全阈值)
    acceptable_risk = 0.05  # R_acceptable = 5%
    comprehensive_risks = np.array(results['comprehensive_risks'])

    # 找到第一个超过5%风险的点
    exceed_indices = np.where(comprehensive_risks > acceptable_risk)[0]
    if len(exceed_indices) > 0:
        critical_distance = distance_errors[exceed_indices[0]]
        print(f" 当轨迹预测距离误差超过 {critical_distance:.1f}米 时，")
        print(f"   综合健康风险超过设定的5%安全阈值 (R_acceptable)")
    else:
        print(" 在测试范围内，所有轨迹预测误差的健康风险均在可接受范围内")

    # 显示关键数值
    max_risk_idx = np.argmax(comprehensive_risks)
    max_risk = comprehensive_risks[max_risk_idx] * 100
    max_risk_distance = distance_errors[max_risk_idx]
    max_sinr_degradation = max(results['sinr_degradations'])
    max_delay = max(results['monitoring_delays'])

    print(f"\n 误差分析统计:")
    print(f"   最大SINR降级: {max_sinr_degradation:.2f} dB (距离误差: {max_risk_distance:.1f}米)")
    print(f"   最大监测延迟: {max_delay:.2f}秒")
    print(f"   最大心脏风险: {max(results['cardiac_risks'])*100:.2f}%")
    print(f"   最大热风险: {max(results['thermal_risks'])*100:.2f}%")
    print(f"   最大综合风险: {max_risk:.2f}% (R_safety)")


if __name__ == "__main__":
    main()
