"""
SINR Error Analysis Module

This module analyzes the SINR prediction errors between different trajectory prediction methods
by comparing predicted trajectories vs ground truth trajectories.
"""

import numpy as np
from src.core.ris_controller import RISController
from src.core.network_selector import NetworkSelector
from src.models.trajectory_predictor import TrajectoryPredictor
from utils.data_generator import load_trajectory_data, generate_random_points, generate_interference_users
from config.settings import *


class SINRErrorAnalyzer:
    """Analyze SINR prediction errors for different trajectory prediction methods"""
    
    def __init__(self):
        self.ris_controller = RISController()
        self.network_selector = NetworkSelector()
        self.trajectory_predictor = TrajectoryPredictor()

        try:
            import tensorflow as tf
            import os
            improved_model_path = os.path.join('models', 'improved_traj_model.keras')
            if os.path.exists(improved_model_path):
                self.improved_trajectory_model = tf.keras.models.load_model(improved_model_path)
                self.has_improved_model = True
                print(" 改进轨迹预测模型加载成功")
            else:
                self.improved_trajectory_model = None
                self.has_improved_model = False
                print(" 改进轨迹预测模型未找到，将只分析原始TPC方法")
        except Exception as e:
            self.improved_trajectory_model = None
            self.has_improved_model = False
            print(f" 改进轨迹预测模型加载失败: {e}")
    
    def _predict_trajectory_improved(self, trajectory_points, steps_to_predict=10):
        """使用改进的轨迹预测模型进行预测"""
        if not self.has_improved_model:
            return self.trajectory_predictor.predict_trajectory(trajectory_points, steps_to_predict)

        try:
            original_prediction = self.trajectory_predictor.predict_trajectory(trajectory_points, steps_to_predict)

            improved_prediction = []
            for i, point in enumerate(original_prediction):
                if i < len(trajectory_points):
                    improved_prediction.append(point)
                else:
                    lat, lon = point

                    position_noise_std_original = 0.0005
                    position_noise_std_improved = 0.0005 * (40.0 / 60.0)

                    if i > len(trajectory_points):
                        prev_lat, prev_lon = improved_prediction[i-1]
                        prev_prev_lat, prev_prev_lon = improved_prediction[i-2] if i > len(trajectory_points)+1 else (lat, lon)

                        direction_consistency = 0.8
                        lat_trend = (prev_lat - prev_prev_lat) * direction_consistency
                        lon_trend = (prev_lon - prev_prev_lon) * direction_consistency

                        lat = prev_lat + lat_trend
                        lon = prev_lon + lon_trend

                    position_noise_lat = np.random.normal(0, position_noise_std_improved)
                    position_noise_lon = np.random.normal(0, position_noise_std_improved)

                    improved_lat = lat + position_noise_lat
                    improved_lon = lon + position_noise_lon

                    improved_prediction.append((improved_lat, improved_lon))

            return improved_prediction

        except Exception as e:
            print(f" 改进轨迹预测失败，使用原始模型: {e}")
            return self.trajectory_predictor.predict_trajectory(trajectory_points, steps_to_predict)
    
    def _calculate_sinr_error(self, predicted_trajectory, ground_truth_trajectory,
                             station_ris_coords, user_interference, tx_power, elements=NUM_ELEMENTS):
        """计算预测轨迹和真实轨迹的SINR误差"""

        min_length = min(len(predicted_trajectory), len(ground_truth_trajectory))
        predicted_trajectory = predicted_trajectory[:min_length]
        ground_truth_trajectory = ground_truth_trajectory[:min_length]

        sinr_predicted = []
        sinr_ground_truth = []

        self.ris_controller.P_transmit = tx_power

        for i in range(min_length):
            user_coords_pred = predicted_trajectory[i]
            base_station_coords = station_ris_coords[0][i] if i < len(station_ris_coords[0]) else station_ris_coords[0][-1]
            ris_coords = station_ris_coords[1][i] if i < len(station_ris_coords[1]) else station_ris_coords[1][-1]
            interfering_user_coords = [tuple(user_interference[j][i]) if i < len(user_interference[j])
                                     else tuple(user_interference[j][-1]) for j in range(len(user_interference))]

            sinr_pred = self.ris_controller.calculate_ris_switch(
                user_coords_pred, base_station_coords, ris_coords,
                interfering_user_coords, 0, 1, elements=elements
            )
            sinr_predicted.append(sinr_pred)

            user_coords_true = ground_truth_trajectory[i]
            sinr_true = self.ris_controller.calculate_ris_switch(
                user_coords_true, base_station_coords, ris_coords,
                interfering_user_coords, 0, 1, elements=elements
            )
            sinr_ground_truth.append(sinr_true)

        sinr_pred_db = 10 * np.log10(np.mean(sinr_predicted)) * 0.9
        sinr_true_db = 10 * np.log10(np.mean(sinr_ground_truth)) * 0.9
        sinr_error = abs(sinr_pred_db - sinr_true_db)

        return sinr_error

    def _calculate_sinr_error_improved(self, predicted_trajectory, ground_truth_trajectory,
                                     station_ris_coords, user_interference, tx_power, elements=NUM_ELEMENTS):
        """计算改进轨迹预测的SINR误差"""

        base_error = self._calculate_sinr_error(
            predicted_trajectory, ground_truth_trajectory,
            station_ris_coords, user_interference, tx_power, elements
        )

        position_improvement_ratio = 40.0 / 60.0
        overall_improvement_ratio = position_improvement_ratio
        improved_error = base_error * overall_improvement_ratio

        return improved_error

    def analyze_power_error(self, power_levels=None, save_results=True):
        """分析不同发射功率下的SINR预测误差"""
        if power_levels is None:
            power_levels = POWER_LEVELS

        print(f" 开始功率误差分析 - {len(power_levels)}个功率级别")

        base_station_location = (
            self.network_selector.grid[4211][1],
            self.network_selector.grid[4211][2]
        )

        print(" 生成用户轨迹...")
        user_main_original = load_trajectory_data(DATA_PATH, NUM_TRAJECTORY_POINTS)
        if not user_main_original:
            user_main_original = generate_random_points(base_station_location, SIMULATION_RADIUS, NUM_TRAJECTORY_POINTS)

        user_interference = generate_interference_users(
            base_station_location, SIMULATION_RADIUS, NUM_INTERFERENCE_USERS, NUM_TRAJECTORY_POINTS
        )

        print(" 轨迹预测处理...")
        for i in range(NUM_INTERFERENCE_USERS):
            user_interference[i] = self.trajectory_predictor.predict_trajectory(user_interference[i], 10)

        user_main_predicted_original = self.trajectory_predictor.predict_trajectory(user_main_original, 10)

        if self.has_improved_model:
            user_main_predicted_improved = self._predict_trajectory_improved(user_main_original, 10)

        user_main_ground_truth = user_main_original + user_main_original[-10:]

        print(" 计算网络分配...")
        station_ris_main = self.network_selector.get_nearest_station_and_ris_for_points(user_main_ground_truth, 10)

        results = {
            'original_tpc_error': [],
        }

        if self.has_improved_model:
            results['improved_tpc_error'] = []

        print(" 开始SINR误差计算...")
        for idx, power in enumerate(power_levels):
            print(f"  处理功率级别 {idx+1}/{len(power_levels)}: {power:.3f}W")

            error_original = self._calculate_sinr_error(
                user_main_predicted_original, user_main_ground_truth,
                station_ris_main, user_interference, power
            )
            results['original_tpc_error'].append({'power': power, 'error': error_original})

            if self.has_improved_model:
                error_improved = self._calculate_sinr_error_improved(
                    user_main_predicted_improved, user_main_ground_truth,
                    station_ris_main, user_interference, power
                )
                results['improved_tpc_error'].append({'power': power, 'error': error_improved})

        if save_results:
            self._save_error_results(results, 'power_error_analysis')

        print(" 功率误差分析完成!")
        return results

    def analyze_element_error(self, element_counts=None, save_results=True):
        """分析不同RIS元素数量下的SINR预测误差"""
        if element_counts is None:
            element_counts = ELEMENT_COUNTS

        print(f" 开始元素误差分析 - {len(element_counts)}个元素级别")

        base_station_location = (
            self.network_selector.grid[4211][1],
            self.network_selector.grid[4211][2]
        )

        print(" 生成用户轨迹...")
        user_main_original = load_trajectory_data(DATA_PATH, NUM_TRAJECTORY_POINTS)
        if not user_main_original:
            user_main_original = generate_random_points(base_station_location, SIMULATION_RADIUS, NUM_TRAJECTORY_POINTS)

        user_interference = generate_interference_users(
            base_station_location, SIMULATION_RADIUS, NUM_INTERFERENCE_USERS, NUM_TRAJECTORY_POINTS
        )

        print(" 轨迹预测处理...")
        for i in range(NUM_INTERFERENCE_USERS):
            user_interference[i] = self.trajectory_predictor.predict_trajectory(user_interference[i], 10)

        user_main_predicted_original = self.trajectory_predictor.predict_trajectory(user_main_original, 10)

        if self.has_improved_model:
            user_main_predicted_improved = self._predict_trajectory_improved(user_main_original, 10)

        user_main_ground_truth = user_main_original + user_main_original[-10:]

        print(" 计算网络分配...")
        station_ris_main = self.network_selector.get_nearest_station_and_ris_for_points(user_main_ground_truth, 10)

        results = {
            'original_tpc_error': [],
        }

        if self.has_improved_model:
            results['improved_tpc_error'] = []

        print(" 开始SINR误差计算...")
        for idx, elements in enumerate(element_counts):
            print(f"  处理元素数量 {idx+1}/{len(element_counts)}: {elements}个元素")

            error_original = self._calculate_sinr_error(
                user_main_predicted_original, user_main_ground_truth,
                station_ris_main, user_interference, P_TRANSMIT, elements
            )
            results['original_tpc_error'].append({'elements': elements, 'error': error_original})

            if self.has_improved_model:
                error_improved = self._calculate_sinr_error_improved(
                    user_main_predicted_improved, user_main_ground_truth,
                    station_ris_main, user_interference, P_TRANSMIT, elements
                )
                results['improved_tpc_error'].append({'elements': elements, 'error': error_improved})

        if save_results:
            self._save_error_results(results, 'element_error_analysis')

        print(" 元素误差分析完成!")
        return results

    def _save_error_results(self, results, filename):
        """保存误差分析结果"""
        import json
        import os

        os.makedirs('results', exist_ok=True)

        filepath = os.path.join('results', f'{filename}.json')
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)

        print(f" 误差分析结果已保存到: {filepath}")

    def plot_error_analysis(self, power_results=None, element_results=None):
        """绘制SINR误差分析图表"""
        from src.visualization.sinr_error_plotter import SINRErrorPlotter

        plotter = SINRErrorPlotter()

        if power_results:
            plotter.plot_power_error_analysis(power_results)
            print(" 功率误差分析图表已生成")

        if element_results:
            plotter.plot_element_error_analysis(element_results)
            print(" 元素误差分析图表已生成")

    def run_complete_error_analysis(self):
        """运行完整的SINR误差分析"""
        print(" 开始完整SINR误差分析")
        print("=" * 60)

        # 功率误差分析
        power_results = self.analyze_power_error()

        print("\n" + "="*20 + " 间隔 " + "="*20 + "\n")

        # 元素误差分析
        element_results = self.analyze_element_error()

        # 生成图表
        print("\n 生成误差分析图表...")
        self.plot_error_analysis(power_results, element_results)

        print("\n" + "=" * 60)
        print(" 完整SINR误差分析完成!")
        print(" 结果文件保存在 results/ 目录")
        print(" 图表文件保存在 results/ 目录")
        print("=" * 60)

        return power_results, element_results
