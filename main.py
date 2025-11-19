"""
Main execution script for RIS-assisted wireless communication system
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.models.trajectory_predictor import TrajectoryPredictor
from src.models.interference_classifier import InterferenceClassifier
from src.analysis.performance_analyzer import PerformanceAnalyzer
from src.analysis.sinr_error_analyzer import SINRErrorAnalyzer
from src.visualization.plotter import ResultPlotter
from src.core.network_deployment import NetworkDeployment
from config.settings import *

def train_models():
    """Train LSTM trajectory prediction and CNN interference classification models"""
    print("Training trajectory prediction model...")
    trajectory_predictor = TrajectoryPredictor()
    trajectory_predictor.train()
    
    print("Training interference classification model...")
    interference_classifier = InterferenceClassifier()
    interference_classifier.train()
    
    print("Model training completed!")

def run_power_analysis():
    """Run analysis across different transmit power levels"""
    print("Running power level analysis...")
    analyzer = PerformanceAnalyzer()
    results = analyzer.analyze_power_levels()
    
    print("Plotting power analysis results...")
    plotter = ResultPlotter()
    plotter.plot_power_analysis(results)
    
    print("Power analysis completed!")
    return results

def run_element_analysis():
    """Run analysis across different RIS element counts"""
    print("Running RIS element count analysis...")
    analyzer = PerformanceAnalyzer()
    results = analyzer.analyze_element_counts()
    
    print("Plotting element analysis results...")
    plotter = ResultPlotter()
    plotter.plot_element_analysis(results)
    
    print("Element analysis completed!")
    return results

def visualize_network():
    """Visualize network deployment"""
    print("Visualizing network deployment...")
    deployment = NetworkDeployment()
    plotter = ResultPlotter()
    
    plotter.plot_network_deployment(
        deployment.get_base_stations(),
        deployment.get_ris_list(),
        deployment.coverage_area
    )
    
    print("Network visualization completed!")

def run_sinr_error_analysis():
    """Run SINR prediction error analysis"""
    print("Running SINR prediction error analysis...")
    analyzer = SINRErrorAnalyzer()
    power_results, element_results = analyzer.run_complete_error_analysis()

    print("SINR error analysis completed!")
    return power_results, element_results

def main():
    """Main execution function"""
    print("RIS-Assisted Wireless Communication System")
    print("=" * 50)
    
    # Create necessary directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    
    while True:
        print("\nSelect an option:")
        print("1. Train models")
        print("2. Run power analysis")
        print("3. Run element analysis")
        print("4. Visualize network deployment")
        print("5. Run SINR error analysis")
        print("6. Run all analyses")
        print("7. Exit")

        choice = input("Enter your choice (1-7): ").strip()
        
        if choice == '1':
            train_models()
        elif choice == '2':
            run_power_analysis()
        elif choice == '3':
            run_element_analysis()
        elif choice == '4':
            visualize_network()
        elif choice == '5':
            run_sinr_error_analysis()
        elif choice == '6':
            print("Running complete analysis suite...")
            train_models()
            run_power_analysis()
            run_element_analysis()
            visualize_network()
            run_sinr_error_analysis()
            print("Complete analysis finished!")
        elif choice == '7':
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
