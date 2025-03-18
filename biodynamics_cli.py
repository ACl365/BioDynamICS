#!/usr/bin/env python3
"""
BioDynamICS Command Line Interface

This script provides a command-line interface for the BioDynamICS system,
allowing users to interact with the system without writing code.

Usage:
    python biodynamics_cli.py [command] [options]

Commands:
    init            Initialize the system
    load            Load MIMIC data
    analyze         Analyze a patient
    visualize       Create visualizations
    batch           Process a batch of patients
    export          Export patient data
    help            Show help message

Author: Alexander Clarke
Date: March 18, 2025
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
import matplotlib.pyplot as plt

# Import the BioDynamicsSystem
from src.system_integration import BioDynamicsSystem

# Global variables
system = None
config_path = "config/biodynamics_config.json"

def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )

def init_system(args):
    """Initialize the BioDynamicsSystem."""
    global system, config_path
    
    # Use specified config path if provided
    if args.config:
        config_path = args.config
    
    # Check if config file exists
    if os.path.exists(config_path):
        print(f"Loading configuration from {config_path}")
        system = BioDynamicsSystem(config_path=config_path)
    else:
        # Create default configuration
        print("No configuration file found. Creating default configuration.")
        
        # Create config directory if it doesn't exist
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        # Default configuration
        config = {
            "data_path": args.data_path or "data/mimic-iii-clinical-database-demo-1.4",
            "results_path": args.results_path or "results",
            "cache_enabled": True,
            "cache_path": "cache",
            "parallel_processing": True,
            "max_workers": 4,
            "analysis_parameters": {
                "window_hours": 24,
                "step_hours": 6,
                "embedding_dimension": 3,
                "stability_threshold": 0.5
            },
            "visualization_settings": {
                "save_figures": True,
                "figure_format": "png",
                "figure_dpi": 300
            }
        }
        
        # Save configuration
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
        
        print(f"Default configuration saved to {config_path}")
        
        # Initialize system
        system = BioDynamicsSystem(config_path=config_path)
    
    print("BioDynamicsSystem initialized successfully")
    return True

def load_data(args):
    """Load MIMIC data."""
    global system
    
    # Ensure system is initialized
    if system is None:
        print("System not initialized. Initializing now...")
        init_system(args)
    
    # Parse tables to load
    tables = args.tables.split(',') if args.tables else None
    
    # Load data
    print(f"Loading MIMIC data tables: {tables or 'core tables'}")
    success = system.load_mimic_data(tables, force_reload=args.force)
    
    if success:
        print("Data loaded successfully")
    else:
        print("Error loading data")
    
    # Load CHARTEVENTS if requested
    if args.chartevents:
        print("Loading CHARTEVENTS table...")
        success = system.load_chartevents(chunk_size=args.chunk_size)
        
        if success:
            print("CHARTEVENTS loaded successfully")
        else:
            print("Error loading CHARTEVENTS")
    
    return success

def analyze_patient(args):
    """Analyze a patient."""
    global system
    
    # Ensure system is initialized
    if system is None:
        print("System not initialized. Initializing now...")
        init_system(args)
    
    # Parse subject ID
    subject_id = int(args.subject_id)
    
    # Parse analyses to perform
    analyses = args.analyses.split(',') if args.analyses else None
    
    # Analyze patient
    print(f"Analyzing patient {subject_id}...")
    results = system.analyze_patient(subject_id, analyses=analyses)
    
    if results:
        print("Analysis completed successfully")
        
        # Print patient summary
        summary = system.get_patient_summary(subject_id)
        if summary:
            print("\nPatient Summary:")
            print(f"  Patient ID: {summary['patient_id']}")
            print(f"  Gender: {summary['gender']}")
            print(f"  Stability Status: {summary['stability_status']}")
            print(f"  Allostatic Load: {summary['allostatic_load']}")
            print(f"  Critical Systems: {', '.join(summary['critical_systems']) if summary['critical_systems'] else 'None'}")
            print(f"  Approaching Transition: {'Yes' if summary['approaching_transition'] else 'No'}")
        
        # Print treatment recommendations if requested
        if args.treatment:
            recommendations = system.get_treatment_recommendations(subject_id)
            if recommendations:
                print("\nTreatment Recommendations:")
                print(f"  Infection Type: {recommendations['infection_type']}")
                print(f"  Recommended Antibiotic: {recommendations['antibiotic']}")
                print(f"  Optimal Dose: {recommendations['optimal_dose']} mg")
                print(f"  Optimal Interval: {recommendations['optimal_interval']} hours")
    else:
        print("Error analyzing patient")
    
    return results is not None

def visualize_patient(args):
    """Create visualizations for a patient."""
    global system
    
    # Ensure system is initialized
    if system is None:
        print("System not initialized. Initializing now...")
        init_system(args)
    
    # Parse subject ID
    subject_id = int(args.subject_id)
    
    # Create results directory
    results_dir = Path(system.config['results_path']) / f"patient_{subject_id}"
    results_dir.mkdir(exist_ok=True)
    
    # Create visualizations based on type
    if args.type == 'all' or args.type == 'vitals':
        print("Creating vital signs visualization...")
        vital_signs_path = results_dir / "vital_signs.png"
        fig = system.visualize_vital_signs(subject_id)
        if fig:
            fig.savefig(vital_signs_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"Vital signs visualization saved to {vital_signs_path}")
    
    if args.type == 'all' or args.type == 'organs':
        print("Creating organ system visualization...")
        organ_system_path = results_dir / "organ_system.png"
        fig = system.visualize_organ_system_status(subject_id)
        if fig:
            fig.savefig(organ_system_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"Organ system visualization saved to {organ_system_path}")
    
    if args.type == 'all' or args.type == 'allostatic':
        print("Creating allostatic load visualization...")
        allostatic_load_path = results_dir / "allostatic_load.png"
        fig = system.visualize_allostatic_load(subject_id)
        if fig:
            fig.savefig(allostatic_load_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"Allostatic load visualization saved to {allostatic_load_path}")
    
    if args.type == 'all' or args.type == 'phase':
        print("Creating phase portrait...")
        phase_portrait_path = results_dir / "phase_portrait.png"
        fig = system.visualize_phase_portrait(subject_id, 'heart_rate', 'sbp')
        if fig:
            fig.savefig(phase_portrait_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"Phase portrait saved to {phase_portrait_path}")
    
    if args.type == 'all' or args.type == 'dashboard':
        print("Creating patient dashboard...")
        dashboard_path = results_dir / "dashboard.png"
        fig = system.create_patient_dashboard(subject_id)
        if fig:
            fig.savefig(dashboard_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"Patient dashboard saved to {dashboard_path}")
    
    print("Visualization completed")
    return True

def process_batch(args):
    """Process a batch of patients."""
    global system
    
    # Ensure system is initialized
    if system is None:
        print("System not initialized. Initializing now...")
        init_system(args)
    
    # Parse subject IDs
    if args.file:
        # Load subject IDs from file
        try:
            with open(args.file, 'r') as f:
                subject_ids = [int(line.strip()) for line in f if line.strip()]
        except Exception as e:
            print(f"Error loading subject IDs from file: {e}")
            return False
    else:
        # Parse subject IDs from command line
        try:
            subject_ids = [int(id) for id in args.subject_ids.split(',')]
        except Exception as e:
            print(f"Error parsing subject IDs: {e}")
            return False
    
    # Parse analyses to perform
    analyses = args.analyses.split(',') if args.analyses else None
    
    # Process batch
    print(f"Processing batch of {len(subject_ids)} patients...")
    batch_results = system.process_patient_batch(
        subject_ids,
        analyses=analyses,
        parallel=not args.sequential
    )
    
    if batch_results:
        print("Batch processing completed successfully")
        print(f"Processed {batch_results['completed']} patients successfully")
        print(f"Failed to process {batch_results['failed']} patients")
        
        # Generate batch report
        batch_id = list(system.batch_results.keys())[-1]
        report_path = Path(system.config['results_path']) / f"batch_{batch_id}_report.json"
        report = system.generate_batch_report(batch_id, str(report_path))
        
        if report:
            print(f"Batch report saved to {report_path}")
    else:
        print("Error processing batch")
    
    return batch_results is not None

def export_data(args):
    """Export patient data."""
    global system
    
    # Ensure system is initialized
    if system is None:
        print("System not initialized. Initializing now...")
        init_system(args)
    
    # Parse subject ID
    subject_id = int(args.subject_id)
    
    # Create export directory
    export_dir = Path(args.output_dir or system.config['results_path'])
    export_dir.mkdir(exist_ok=True)
    
    # Export data
    export_path = export_dir / f"patient_{subject_id}_data.{args.format}"
    print(f"Exporting data for patient {subject_id} to {export_path}...")
    
    success = system.export_patient_data(subject_id, str(export_path), format=args.format)
    
    if success:
        print(f"Data exported successfully to {export_path}")
    else:
        print("Error exporting data")
    
    # Save system state if requested
    if args.save_state:
        state_path = export_dir / "system_state.pkl"
        print(f"Saving system state to {state_path}...")
        
        success = system.save_state(str(state_path))
        
        if success:
            print(f"System state saved to {state_path}")
        else:
            print("Error saving system state")
    
    return success

def main():
    """Main function."""
    # Set up logging
    setup_logging()
    
    # Create main parser
    parser = argparse.ArgumentParser(
        description='BioDynamICS Command Line Interface',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Add subparsers for commands
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Init command
    init_parser = subparsers.add_parser('init', help='Initialize the system')
    init_parser.add_argument('--config', help='Path to configuration file')
    init_parser.add_argument('--data-path', help='Path to MIMIC data')
    init_parser.add_argument('--results-path', help='Path to save results')
    
    # Load command
    load_parser = subparsers.add_parser('load', help='Load MIMIC data')
    load_parser.add_argument('--tables', help='Comma-separated list of tables to load')
    load_parser.add_argument('--chartevents', action='store_true', help='Load CHARTEVENTS table')
    load_parser.add_argument('--chunk-size', type=int, default=100000, help='Chunk size for loading CHARTEVENTS')
    load_parser.add_argument('--force', action='store_true', help='Force reload of tables')
    load_parser.add_argument('--config', help='Path to configuration file')
    load_parser.add_argument('--data-path', help='Path to MIMIC data')
    load_parser.add_argument('--results-path', help='Path to save results')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze a patient')
    analyze_parser.add_argument('subject_id', help='Patient subject ID')
    analyze_parser.add_argument('--analyses', help='Comma-separated list of analyses to perform')
    analyze_parser.add_argument('--treatment', action='store_true', help='Include treatment recommendations')
    analyze_parser.add_argument('--config', help='Path to configuration file')
    analyze_parser.add_argument('--data-path', help='Path to MIMIC data')
    analyze_parser.add_argument('--results-path', help='Path to save results')
    
    # Visualize command
    visualize_parser = subparsers.add_parser('visualize', help='Create visualizations')
    visualize_parser.add_argument('subject_id', help='Patient subject ID')
    visualize_parser.add_argument('--type', choices=['all', 'vitals', 'organs', 'allostatic', 'phase', 'dashboard'],
                                default='all', help='Type of visualization to create')
    visualize_parser.add_argument('--config', help='Path to configuration file')
    visualize_parser.add_argument('--data-path', help='Path to MIMIC data')
    visualize_parser.add_argument('--results-path', help='Path to save results')
    
    # Batch command
    batch_parser = subparsers.add_parser('batch', help='Process a batch of patients')
    batch_parser.add_argument('--subject-ids', help='Comma-separated list of patient subject IDs')
    batch_parser.add_argument('--file', help='File containing patient subject IDs (one per line)')
    batch_parser.add_argument('--analyses', help='Comma-separated list of analyses to perform')
    batch_parser.add_argument('--sequential', action='store_true', help='Process patients sequentially')
    batch_parser.add_argument('--config', help='Path to configuration file')
    batch_parser.add_argument('--data-path', help='Path to MIMIC data')
    batch_parser.add_argument('--results-path', help='Path to save results')
    
    # Export command
    export_parser = subparsers.add_parser('export', help='Export patient data')
    export_parser.add_argument('subject_id', help='Patient subject ID')
    export_parser.add_argument('--format', choices=['json', 'csv', 'pickle'], default='json',
                             help='Export format')
    export_parser.add_argument('--output-dir', help='Directory to save exported data')
    export_parser.add_argument('--save-state', action='store_true', help='Save system state')
    export_parser.add_argument('--config', help='Path to configuration file')
    export_parser.add_argument('--data-path', help='Path to MIMIC data')
    export_parser.add_argument('--results-path', help='Path to save results')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Execute command
    if args.command == 'init':
        init_system(args)
    elif args.command == 'load':
        load_data(args)
    elif args.command == 'analyze':
        analyze_patient(args)
    elif args.command == 'visualize':
        visualize_patient(args)
    elif args.command == 'batch':
        process_batch(args)
    elif args.command == 'export':
        export_data(args)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()