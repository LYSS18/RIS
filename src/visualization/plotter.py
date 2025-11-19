"""
Visualization module for plotting analysis results
"""
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np
from config.settings import *

class ResultPlotter:
    """Handles plotting of analysis results"""
    
    def __init__(self):
        self.figure_size = FIGURE_SIZE
        self.dpi = DPI
        self.font_size_title = FONT_SIZE_TITLE
        self.font_size_label = FONT_SIZE_LABEL
        self.font_size_legend = FONT_SIZE_LEGEND
    
    def plot_power_analysis(self, results, save_path="./results/power_analysis.png"):
        """Plot SINR vs transmit power comparison"""
        # Extract data
        power_values = [d["power"] for d in results['proposed_tpc']]
        sinr_proposed = [d["sinr"] for d in results['proposed_tpc']]
        sinr_always_on = [d["sinr"] for d in results['ris_always_on']]
        sinr_isl_based = [d["sinr"] for d in results['isl_based']]

        # Check if improved TPC data exists
        has_improved = 'improved_tpc' in results and len(results['improved_tpc']) > 0
        if has_improved:
            sinr_improved = [d["sinr"] for d in results['improved_tpc']]

        # Create figure
        fig, ax = plt.subplots(figsize=self.figure_size)

        # Format axes
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))

        # Plot lines
        ax.plot(power_values, sinr_proposed,
                color='b', marker='o', markerfacecolor='none', markeredgecolor='b',
                markersize=8, linestyle='-', label="Proposed TPC")
        ax.plot(power_values, sinr_always_on,
                color='r', marker='s', label="RIS always on",
                linestyle='-', markersize=6)
        ax.plot(power_values, sinr_isl_based,
                color='purple', marker='x', label="ISL based control",
                linestyle='-', markersize=6)

        # Plot Improved TPC if available
        if has_improved:
            ax.plot(power_values, sinr_improved,
                    color='green', marker='d', label="Improved TPC",
                    linestyle='-', markersize=6)

        # Formatting
        ax.set_xlabel('P_transmit(W)', fontsize=self.font_size_label)
        ax.set_ylabel('SINR(dB)', fontsize=self.font_size_label)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(fontsize=self.font_size_legend)
        plt.suptitle("SINR Comparison Across P_transmit", fontsize=self.font_size_title, y=0.95)

        # Save and show
        plt.gcf().set_dpi(self.dpi)
        plt.savefig(save_path, dpi=self.dpi)
        plt.show()

        return fig, ax
    
    def plot_element_analysis(self, results, save_path="./results/element_analysis.png"):
        """Plot SINR vs RIS element count comparison"""
        # Extract data
        element_values = [d["elements"] for d in results['proposed_tpc']]
        sinr_proposed = [d["sinr"] for d in results['proposed_tpc']]
        sinr_always_on = [d["sinr"] for d in results['ris_always_on']]
        sinr_isl_based = [d["sinr"] for d in results['isl_based']]

        # Check if improved TPC data exists
        has_improved = 'improved_tpc' in results and len(results['improved_tpc']) > 0
        if has_improved:
            sinr_improved = [d["sinr"] for d in results['improved_tpc']]

        # Create figure
        fig, ax = plt.subplots(figsize=self.figure_size)

        # Format axes
        ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))

        # Plot lines
        ax.plot(element_values, sinr_proposed,
                color='b', marker='o', markerfacecolor='none', markeredgecolor='b',
                markersize=8, linestyle='-', label="Proposed TPC")
        ax.plot(element_values, sinr_always_on,
                color='r', marker='s', label="RIS always on",
                linestyle='-', markersize=6)
        ax.plot(element_values, sinr_isl_based,
                color='purple', marker='x', label="ISL based control",
                linestyle='-', markersize=6)

        # Plot Improved TPC if available
        if has_improved:
            ax.plot(element_values, sinr_improved,
                    color='green', marker='d', label="Improved TPC",
                    linestyle='-', markersize=6)

        # Formatting
        ax.set_xlabel('Number of RIS Elements', fontsize=self.font_size_label)
        ax.set_ylabel('SINR(dB)', fontsize=self.font_size_label)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(fontsize=self.font_size_legend)
        plt.suptitle("SINR Comparison Across RIS Element Count", fontsize=self.font_size_title, y=0.95)

        # Save and show
        plt.gcf().set_dpi(self.dpi)
        plt.savefig(save_path, dpi=self.dpi)
        plt.show()

        return fig, ax
    
    def plot_trajectory(self, trajectory_data, title="Trajectory Visualization", 
                       save_path="./results/trajectory.png"):
        """Plot trajectory data"""
        trajectory = np.array(trajectory_data)
        
        fig, ax = plt.subplots(figsize=self.figure_size)
        
        ax.plot(trajectory[:, 1], trajectory[:, 0], 'r-', label='Trajectory', linewidth=2)
        ax.scatter(trajectory[0, 1], trajectory[0, 0], color='green', s=100, label='Start', zorder=5)
        ax.scatter(trajectory[-1, 1], trajectory[-1, 0], color='red', s=100, label='End', zorder=5)
        
        ax.set_xlabel('Longitude', fontsize=self.font_size_label)
        ax.set_ylabel('Latitude', fontsize=self.font_size_label)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(fontsize=self.font_size_legend)
        plt.suptitle(title, fontsize=self.font_size_title, y=0.95)
        
        plt.gcf().set_dpi(self.dpi)
        plt.savefig(save_path, dpi=self.dpi)
        plt.show()
        
        return fig, ax
    
    def plot_network_deployment(self, base_stations, ris_locations, coverage_area, 
                               save_path="./results/network_deployment.png"):
        """Plot network deployment visualization"""
        fig, ax = plt.subplots(figsize=self.figure_size)
        
        # Plot base stations
        bs_lats = [bs[1] for bs in base_stations[:100]]  # Limit for visibility
        bs_lons = [bs[2] for bs in base_stations[:100]]
        ax.scatter(bs_lons, bs_lats, color='blue', s=50, label='Base Stations', alpha=0.7)
        
        # Plot RIS
        ris_lats = [ris['ris_latitude'] for ris in ris_locations[:500]]  # Limit for visibility
        ris_lons = [ris['ris_longitude'] for ris in ris_locations[:500]]
        ax.scatter(ris_lons, ris_lats, color='red', s=10, label='RIS', alpha=0.5)
        
        # Set coverage area bounds
        ax.set_xlim(coverage_area['lon_min'], coverage_area['lon_max'])
        ax.set_ylim(coverage_area['lat_min'], coverage_area['lat_max'])
        
        ax.set_xlabel('Longitude', fontsize=self.font_size_label)
        ax.set_ylabel('Latitude', fontsize=self.font_size_label)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(fontsize=self.font_size_legend)
        plt.suptitle("Network Deployment", fontsize=self.font_size_title, y=0.95)
        
        plt.gcf().set_dpi(self.dpi)
        plt.savefig(save_path, dpi=self.dpi)
        plt.show()
        
        return fig, ax
