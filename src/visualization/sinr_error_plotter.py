"""
SINR Error Visualization Module

This module provides visualization capabilities for SINR prediction error analysis,
comparing different trajectory prediction methods.
"""

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np
from config.settings import *


class SINRErrorPlotter:
    """Plotter for SINR prediction error analysis results"""
    
    def __init__(self):
        self.figure_size = FIGURE_SIZE
        self.font_size_title = FONT_SIZE_TITLE
        self.font_size_label = FONT_SIZE_LABEL
        self.font_size_legend = FONT_SIZE_LEGEND
        self.dpi = DPI

        plt.style.use('default')
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.grid'] = True
        plt.rcParams['grid.alpha'] = 0.3
    
    def plot_power_error_analysis(self, results, save_path="./results/power_error_analysis.png"):
        """绘制不同发射功率下的SINR预测误差对比"""

        power_values = [d["power"] for d in results['original_tpc_error']]
        error_original = [d["error"] for d in results['original_tpc_error']]

        has_improved = 'improved_tpc_error' in results and len(results['improved_tpc_error']) > 0
        if has_improved:
            error_improved = [d["error"] for d in results['improved_tpc_error']]

        fig, ax = plt.subplots(figsize=self.figure_size)

        ax.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))

        ax.plot(power_values, error_original,
                color='b', marker='o', markerfacecolor='none', markeredgecolor='b',
                markersize=8, linestyle='-', label="Original TPC Error")

        if has_improved:
            ax.plot(power_values, error_improved,
                    color='green', marker='d', label="Improved TPC Error",
                    markersize=6, linestyle='-')

        ax.set_xlabel('P_transmit(W)', fontsize=self.font_size_label)
        ax.set_ylabel('SINR Prediction Error(dB)', fontsize=self.font_size_label)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(fontsize=self.font_size_legend)
        plt.suptitle("SINR Prediction Error Across P_transmit", fontsize=self.font_size_title, y=0.95)

        plt.gcf().set_dpi(self.dpi)
        plt.savefig(save_path, dpi=self.dpi)
        plt.show()

        print(f"📊 功率误差分析图表已保存: {save_path}")
        return fig, ax
    
    def plot_element_error_analysis(self, results, save_path="./results/element_error_analysis.png"):
        """绘制不同RIS元素数量下的SINR预测误差对比"""

        element_values = [d["elements"] for d in results['original_tpc_error']]
        error_original = [d["error"] for d in results['original_tpc_error']]

        has_improved = 'improved_tpc_error' in results and len(results['improved_tpc_error']) > 0
        if has_improved:
            error_improved = [d["error"] for d in results['improved_tpc_error']]

        fig, ax = plt.subplots(figsize=self.figure_size)

        ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))

        ax.plot(element_values, error_original,
                color='b', marker='o', markerfacecolor='none', markeredgecolor='b',
                markersize=8, linestyle='-', label="Original TPC Error")

        if has_improved:
            ax.plot(element_values, error_improved,
                    color='green', marker='d', label="Improved TPC Error",
                    markersize=6, linestyle='-')

        ax.set_xlabel('Number of RIS Elements', fontsize=self.font_size_label)
        ax.set_ylabel('SINR Prediction Error(dB)', fontsize=self.font_size_label)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(fontsize=self.font_size_legend)
        plt.suptitle("SINR Prediction Error Across RIS Element Count", fontsize=self.font_size_title, y=0.95)

        plt.gcf().set_dpi(self.dpi)
        plt.savefig(save_path, dpi=self.dpi)
        plt.show()

        print(f"📊 元素误差分析图表已保存: {save_path}")
        return fig, ax
    
    def plot_combined_error_analysis(self, power_results, element_results, 
                                   save_path="./results/combined_error_analysis.png"):
        """绘制组合的SINR误差分析图表"""
        
        # 创建子图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # 功率误差分析子图
        power_values = [d["power"] for d in power_results['original_tpc_error']]
        error_original_power = [d["error"] for d in power_results['original_tpc_error']]
        
        ax1.plot(power_values, error_original_power, 
                color='blue', marker='o', markerfacecolor='none', markeredgecolor='blue',
                markersize=8, linestyle='-', linewidth=2, label="Original TPC Error")
        
        if 'improved_tpc_error' in power_results:
            error_improved_power = [d["error"] for d in power_results['improved_tpc_error']]
            ax1.plot(power_values, error_improved_power, 
                    color='green', marker='d', markerfacecolor='green', markeredgecolor='green',
                    markersize=6, linestyle='-', linewidth=2, label="Improved TPC Error")
        
        ax1.set_xlabel('Transmit Power (W)', fontsize=self.font_size_label)
        ax1.set_ylabel('SINR Prediction Error (dB)', fontsize=self.font_size_label)
        ax1.set_title('Error vs Transmit Power', fontsize=self.font_size_label)
        ax1.grid(True, linestyle='--', alpha=0.5)
        ax1.legend(fontsize=self.font_size_legend)
        
        # 元素误差分析子图
        element_values = [d["elements"] for d in element_results['original_tpc_error']]
        error_original_element = [d["error"] for d in element_results['original_tpc_error']]
        
        ax2.plot(element_values, error_original_element, 
                color='blue', marker='o', markerfacecolor='none', markeredgecolor='blue',
                markersize=8, linestyle='-', linewidth=2, label="Original TPC Error")
        
        if 'improved_tpc_error' in element_results:
            error_improved_element = [d["error"] for d in element_results['improved_tpc_error']]
            ax2.plot(element_values, error_improved_element, 
                    color='green', marker='d', markerfacecolor='green', markeredgecolor='green',
                    markersize=6, linestyle='-', linewidth=2, label="Improved TPC Error")
        
        ax2.set_xlabel('Number of RIS Elements', fontsize=self.font_size_label)
        ax2.set_ylabel('SINR Prediction Error (dB)', fontsize=self.font_size_label)
        ax2.set_title('Error vs RIS Element Count', fontsize=self.font_size_label)
        ax2.grid(True, linestyle='--', alpha=0.5)
        ax2.legend(fontsize=self.font_size_legend)
        
        # 设置总标题
        plt.suptitle("SINR Prediction Error Analysis Comparison", 
                    fontsize=self.font_size_title, y=0.98)
        
        # 保存和显示
        plt.gcf().set_dpi(self.dpi)
        plt.tight_layout()
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.show()
        
        print(f"📊 组合误差分析图表已保存: {save_path}")
        return fig, (ax1, ax2)
