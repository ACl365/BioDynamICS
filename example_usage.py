"""
BioDynamICS - Example Usage Script

This script demonstrates how to use the BioDynamicsSystem for analyzing patient data.

Author: Alexander Clarke
Date: March 18, 2025
"""

import os
import sys
import json
from pathlib import Path
import matplotlib.pyplot as plt

# Import the BioDynamicsSystem
from src.system_integration import BioDynamicsSystem

def main():
    """Main function demonstrating BioDynamicsSystem usage."""
    print("BioDynamICS Example Usage")
    print("========================\n")
    
    # Create results directory if it doesn't exist
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    # Create a configuration dictionary
    config = {
        "data_path": "data/mimic-iii-clinical-database-demo-1.4",
        "results_path": "results",
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
    
    # Save configuration to file
    config_path = results_dir / "example_config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    
    print(f"Configuration saved to {config_path}")
    
    # Initialize the BioDynamicsSystem with the configuration
    print("\nInitializing BioDynamicsSystem...")
    system = BioDynamicsSystem(config_path=config_path)
    
    # Load MIMIC data
    print("\nLoading MIMIC data...")
    system.load_mimic_data(['PATIENTS', 'ADMISSIONS', 'ICUSTAYS', 'CHARTEVENTS'])
    
    # Select a patient for analysis
    # In a real application, you would select a patient based on specific criteria
    # For this example, we'll use the first patient in the dataset
    print("\nGetting patient ID...")
    patient_ids = system.data_integrator.tables['PATIENTS']['subject_id'].unique()
    if len(patient_ids) == 0:
        print("Error: No patients found in the dataset")
        return
    
    subject_id = patient_ids[0]
    print(f"Selected patient ID: {subject_id}")
    
    # Create patient timeline
    print("\nCreating patient timeline...")
    timeline = system.create_patient_timeline(subject_id)
    if timeline is None:
        print("Error: Failed to create patient timeline")
        return
    
    print(f"Timeline created with {len(timeline['timeline'])} events")
    
    # Process physiological signals
    print("\nProcessing physiological signals...")
    signals = system.process_patient_signals(subject_id)
    if signals is None:
        print("Warning: Failed to process physiological signals")
    else:
        print("Physiological signals processed successfully")
    
    # Analyze physiological stability
    print("\nAnalyzing physiological stability...")
    stability = system.analyze_physiological_stability(subject_id)
    if stability is None or stability.empty:
        print("Warning: Failed to analyze physiological stability")
    else:
        print(f"Stability analysis completed with {len(stability)} time windows")
    
    # Create stability report
    print("\nCreating stability report...")
    report = system.create_stability_report(subject_id)
    if report is None:
        print("Warning: Failed to create stability report")
    else:
        print("Stability report created successfully")
    
    # Analyze dynamical stability
    print("\nAnalyzing dynamical stability...")
    dynamical = system.analyze_dynamical_stability(subject_id)
    if dynamical is None:
        print("Warning: Failed to analyze dynamical stability")
    else:
        print("Dynamical stability analysis completed successfully")
    
    # Detect critical transitions
    print("\nDetecting critical transitions...")
    transitions = system.detect_critical_transitions(subject_id)
    if transitions is None:
        print("Warning: Failed to detect critical transitions")
    else:
        print("Critical transition detection completed successfully")
    
    # Analyze infection treatment
    print("\nAnalyzing infection treatment...")
    infection = system.analyze_infection_treatment(subject_id)
    if infection is None:
        print("Warning: Failed to analyze infection treatment")
    else:
        print("Infection treatment analysis completed successfully")
    
    # Create visualizations
    print("\nCreating visualizations...")
    
    # Vital signs
    print("  - Vital signs timeline...")
    vital_signs_fig = system.visualize_vital_signs(subject_id)
    if vital_signs_fig is not None:
        vital_signs_path = results_dir / f"patient_{subject_id}_vital_signs.png"
        vital_signs_fig.savefig(vital_signs_path, dpi=300, bbox_inches='tight')
        plt.close(vital_signs_fig)
        print(f"    Saved to {vital_signs_path}")
    
    # Organ system status
    print("  - Organ system status...")
    organ_system_fig = system.visualize_organ_system_status(subject_id)
    if organ_system_fig is not None:
        organ_system_path = results_dir / f"patient_{subject_id}_organ_system.png"
        organ_system_fig.savefig(organ_system_path, dpi=300, bbox_inches='tight')
        plt.close(organ_system_fig)
        print(f"    Saved to {organ_system_path}")
    
    # Allostatic load
    print("  - Allostatic load trend...")
    allostatic_load_fig = system.visualize_allostatic_load(subject_id)
    if allostatic_load_fig is not None:
        allostatic_load_path = results_dir / f"patient_{subject_id}_allostatic_load.png"
        allostatic_load_fig.savefig(allostatic_load_path, dpi=300, bbox_inches='tight')
        plt.close(allostatic_load_fig)
        print(f"    Saved to {allostatic_load_path}")
    
    # Phase portrait
    print("  - Phase portrait...")
    phase_portrait_fig = system.visualize_phase_portrait(subject_id, 'heart_rate', 'sbp')
    if phase_portrait_fig is not None:
        phase_portrait_path = results_dir / f"patient_{subject_id}_phase_portrait.png"
        phase_portrait_fig.savefig(phase_portrait_path, dpi=300, bbox_inches='tight')
        plt.close(phase_portrait_fig)
        print(f"    Saved to {phase_portrait_path}")
    
    # Patient dashboard
    print("  - Patient dashboard...")
    dashboard_fig = system.create_patient_dashboard(subject_id)
    if dashboard_fig is not None:
        dashboard_path = results_dir / f"patient_{subject_id}_dashboard.png"
        dashboard_fig.savefig(dashboard_path, dpi=300, bbox_inches='tight')
        plt.close(dashboard_fig)
        print(f"    Saved to {dashboard_path}")
    
    # Get patient summary
    print("\nGetting patient summary...")
    summary = system.get_patient_summary(subject_id)
    if summary is not None:
        print("Patient Summary:")
        print(f"  - Patient ID: {summary['patient_id']}")
        print(f"  - Gender: {summary['gender']}")
        print(f"  - Stability Status: {summary['stability_status']}")
        print(f"  - Allostatic Load: {summary['allostatic_load']}")
        print(f"  - Critical Systems: {', '.join(summary['critical_systems']) if summary['critical_systems'] else 'None'}")
        print(f"  - Approaching Transition: {'Yes' if summary['approaching_transition'] else 'No'}")
    
    # Get treatment recommendations
    print("\nGetting treatment recommendations...")
    recommendations = system.get_treatment_recommendations(subject_id)
    if recommendations is not None:
        print("Treatment Recommendations:")
        print(f"  - Infection Type: {recommendations['infection_type']}")
        print(f"  - Recommended Antibiotic: {recommendations['antibiotic']}")
        print(f"  - Optimal Dose: {recommendations['optimal_dose']} mg")
        print(f"  - Optimal Interval: {recommendations['optimal_interval']} hours")
    
    # Export patient data
    print("\nExporting patient data...")
    export_path = results_dir / f"patient_{subject_id}_data.json"
    success = system.export_patient_data(subject_id, export_path, format='json')
    if success:
        print(f"Patient data exported to {export_path}")
    
    # Save system state
    print("\nSaving system state...")
    state_path = results_dir / "system_state.pkl"
    success = system.save_state(state_path)
    if success:
        print(f"System state saved to {state_path}")
    
    # Process a batch of patients
    print("\nProcessing a batch of patients...")
    batch_subject_ids = patient_ids[:min(3, len(patient_ids))]
    batch_results = system.process_patient_batch(
        batch_subject_ids,
        analyses=['timeline', 'signals', 'stability']
    )
    
    # Generate batch report
    print("\nGenerating batch report...")
    batch_id = list(system.batch_results.keys())[0]
    report_path = results_dir / f"batch_{batch_id}_report.json"
    batch_report = system.generate_batch_report(batch_id, report_path)
    if batch_report is not None:
        print(f"Batch report generated and saved to {report_path}")
    
    print("\nExample completed successfully!")

if __name__ == "__main__":
    main()