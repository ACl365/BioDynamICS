# BioDynamICS: Infection Treatment Modeling Demo

This notebook demonstrates the infection treatment modeling capabilities of the BioDynamICS framework, which integrates pharmacokinetic/pharmacodynamic (PK/PD) models with infection dynamics to analyze and optimize antimicrobial therapy.

## 1. Setup and Imports

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import integrate, optimize
import warnings

# Add the project root to path
project_root = r"C:\Users\alex5\Documents\Projects\MIMIC_III\biodynamics"
if project_root not in sys.path:
    sys.path.append(project_root)

# Import our custom modules
from src.data_integration import MimicPatientIntegrator
from src.signal_processing import PhysiologicalSignalProcessor
from src.dynamical_modeling import DynamicalSystemsModeler
from src.infection_modeling import InfectionTreatmentModeler
from src.visualization import ClinicalVisualizer

# Configure visualization settings
plt.style.use('seaborn-v0_8-whitegrid')
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = [12, 6]
plt.rcParams['font.size'] = 12

# Suppress warnings
warnings.filterwarnings('ignore')

## 2. Initialize Infection Treatment Modeler

# Create the infection treatment modeler
infection_modeler = InfectionTreatmentModeler()

## 3. Basic Infection Dynamics

# Simulate bacterial growth for different pathogens
simulation_duration = 72  # hours
pathogens = ['e_coli', 's_aureus', 'p_aeruginosa', 'k_pneumoniae']

plt.figure(figsize=(14, 8))
for pathogen in pathogens:
    time_points, bacterial_load = infection_modeler.simulate_bacterial_growth(
        duration_hours=simulation_duration, 
        pathogen=pathogen
    )
    plt.semilogy(time_points, bacterial_load, label=pathogen.replace('_', ' ').title())

plt.title('Bacterial Growth Dynamics for Different Pathogens', fontsize=14)
plt.xlabel('Time (hours)', fontsize=12)
plt.ylabel('Bacterial Load (CFU/mL)', fontsize=12)
plt.legend()
plt.grid(True, which="both", ls="-", alpha=0.2)
plt.tight_layout()
plt.show()

# Examine growth parameters
print("\nBacterial Growth Parameters by Pathogen:")
for pathogen, params in infection_modeler.bacterial_growth_params.items():
    if pathogen != 'default':
        print(f"\n{pathogen.replace('_', ' ').title()}")
        for param, value in params.items():
            print(f"  {param.replace('_', ' ').title()}: {value}")

## 4. Pharmacokinetic/Pharmacodynamic (PK/PD) Modeling

# Simulate antibiotic concentration over time
simulation_duration = 24  # hours
antibiotics = ['vancomycin', 'ceftriaxone', 'ciprofloxacin', 'piperacillin']
dose = 1000  # mg

plt.figure(figsize=(14, 8))
for antibiotic in antibiotics:
    # Simulate single dose at t=0
    time_points = np.linspace(0, simulation_duration, 100)
    concentrations = [
        infection_modeler.antibiotic_concentration(t, dose, [0], antibiotic)
        for t in time_points
    ]
    
    plt.plot(time_points, concentrations, label=antibiotic.title())

plt.title('Antibiotic Pharmacokinetics After Single Dose', fontsize=14)
plt.xlabel('Time (hours)', fontsize=12)
plt.ylabel('Concentration (mg/L)', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.2)
plt.tight_layout()
plt.show()

# Multiple dosing simulation
vancomycin_dose = 1000  # mg
vancomycin_interval = 12  # hours
time_points = np.linspace(0, 48, 200)
dose_times = [0, 12, 24, 36]

concentrations = [
    infection_modeler.antibiotic_concentration(t, vancomycin_dose, dose_times, 'vancomycin')
    for t in time_points
]

plt.figure(figsize=(14, 6))
plt.plot(time_points, concentrations, 'b-', linewidth=2)

# Add MIC thresholds for different pathogens
for pathogen in ['s_aureus', 'e_coli', 'p_aeruginosa']:
    if isinstance(infection_modeler.antibiotic_params['vancomycin']['mic'], dict):
        mic = infection_modeler.antibiotic_params['vancomycin']['mic'].get(pathogen, 1.0)
        plt.axhline(y=mic, linestyle='--', alpha=0.7, 
                   label=f"MIC for {pathogen.replace('_', ' ').title()}")

# Add dose times markers
for dose_time in dose_times:
    plt.axvline(x=dose_time, color='r', linestyle=':', alpha=0.5)
    plt.scatter(dose_time, 0, color='r', s=50, zorder=5)

plt.title('Vancomycin Concentration with Multiple Dosing (1000mg q12h)', fontsize=14)
plt.xlabel('Time (hours)', fontsize=12)
plt.ylabel('Concentration (mg/L)', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.2)
plt.tight_layout()
plt.show()

## 5. Simulating Treatment Response

# Simulate treatment of S. aureus with vancomycin
simulation_results = infection_modeler.simulate_treatment(
    antibiotic='vancomycin',
    pathogen='s_aureus',
    dose=1000,
    dosing_interval=12,
    duration_hours=96,
    initial_delay=6  # 6-hour delay before treatment starts
)

# Extract simulation data
times = simulation_results['times']
bacterial_load = simulation_results['bacterial_load']
antibiotic_conc = simulation_results['antibiotic_concentration']
dose_times = simulation_results['dose_times']

# Create figure with two y-axes
fig, ax1 = plt.subplots(figsize=(14, 8))

# Plot bacterial load on log scale (left y-axis)
ax1.semilogy(times, bacterial_load, 'b-', label='Bacterial Load')
ax1.set_xlabel('Time (hours)', fontsize=12)
ax1.set_ylabel('Bacterial Load (CFU/mL)', color='b', fontsize=12)
ax1.tick_params(axis='y', labelcolor='b')

# Create second y-axis for antibiotic concentration
ax2 = ax1.twinx()
ax2.plot(times, antibiotic_conc, 'r-', label='Vancomycin Concentration')
ax2.set_ylabel('Antibiotic Concentration (mg/L)', color='r', fontsize=12)
ax2.tick_params(axis='y', labelcolor='r')

# Add MIC line
if isinstance(infection_modeler.antibiotic_params['vancomycin']['mic'], dict):
    mic = infection_modeler.antibiotic_params['vancomycin']['mic'].get('s_aureus', 1.0)
    ax2.axhline(y=mic, color='r', linestyle='--', alpha=0.7, label=f'MIC')

# Add markers for dose administration
for dose_time in dose_times:
    ax2.axvline(x=dose_time, color='r', linestyle=':', alpha=0.2)

# Add vertical line at treatment initiation
ax1.axvline(x=6, color='green', linestyle='-', alpha=0.7, label='Treatment Start')

# Add legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

plt.title('S. aureus Infection Treated with Vancomycin (1000mg q12h)', fontsize=14)
plt.grid(True, alpha=0.2)
plt.tight_layout()
plt.show()

# Print treatment metrics
print("\nTreatment Metrics:")
metrics = simulation_results['metrics']
for key, value in metrics.items():
    if key != 'time_to_reduction':  # Handle this separately
        print(f"{key.replace('_', ' ').title()}: {value}")

# Print time to reduction separately
if 'time_to_reduction' in metrics:
    print("\nTime to Bacterial Reduction:")
    for pct, time in metrics['time_to_reduction'].items():
        time_str = f"{time:.1f} hours" if time is not None else "Not achieved"
        print(f"  {pct} reduction: {time_str}")

## 6. Comparing Different Treatment Regimens

# Define several treatment regimens for comparison
regimens = [
    {'dose': 1000, 'interval': 12},  # Standard q12h
    {'dose': 1500, 'interval': 12},  # Higher dose q12h
    {'dose': 1000, 'interval': 8},   # Standard dose q8h
    {'dose': 750, 'interval': 6},    # Lower dose q6h
    {'dose': 2000, 'interval': 24}   # High dose q24h
]

# Compare the regimens
comparison = infection_modeler.evaluate_multiple_regimens(
    antibiotic='vancomycin',
    pathogen='s_aureus',
    regimens=regimens,
    duration_hours=120,
    initial_delay=6
)

# Create a visualization of bacterial load for each regimen
plt.figure(figsize=(14, 8))

for i, metrics in enumerate(comparison['all_regimens']):
    sim_results = metrics['simulation_results']
    times = sim_results['times']
    bacterial_load = sim_results['bacterial_load']
    
    dose = metrics['dose']
    interval = metrics['interval']
    label = f"Regimen {i+1}: {dose}mg q{interval}h"
    
    plt.semilogy(times, bacterial_load, label=label)

plt.title('Comparing Treatment Regimens for S. aureus Infection', fontsize=14)
plt.xlabel('Time (hours)', fontsize=12)
plt.ylabel('Bacterial Load (CFU/mL)', fontsize=12)
plt.legend()
plt.grid(True, which="both", ls="-", alpha=0.2)
plt.tight_layout()
plt.show()

# Print comparison results
print("\nRegimen Comparison Summary:")
for i, metrics in enumerate(comparison['all_regimens']):
    dose = metrics['dose']
    interval = metrics['interval']
    success = "Successful" if metrics['is_successful'] else "Unsuccessful"
    cleared = "Yes" if metrics['is_cleared'] else "No"
    if metrics['is_cleared']:
        time_to_clear = f"{metrics['time_to_clearance']:.1f} hours"
    else:
        time_to_clear = "Not cleared"
    
    print(f"\nRegimen {i+1}: {dose}mg q{interval}h")
    print(f"  Treatment outcome: {success}")
    print(f"  Infection cleared: {cleared}")
    print(f"  Time to clearance: {time_to_clear}")
    print(f"  Log reduction: {np.log10(max(metrics['load_reduction'], 1)):.2f}")
    print(f"  Time above MIC: {metrics['time_above_mic']:.1f}%")
    print(f"  AUC/MIC ratio: {metrics['auc_mic_ratio']:.1f}")

# Identify best regimens
best_regimens = comparison['best_regimens']
print("\nBest Regimens by Category:")
for criterion, idx in best_regimens.items():
    reg = comparison['all_regimens'][idx]
    print(f"  Best for {criterion}: Regimen {idx+1} ({reg['dose']}mg q{reg['interval']}h)")

## 7. Treatment Optimization

# Optimize dosing regimen for S. aureus infection
optimization_results = infection_modeler.optimize_dosing_regimen(
    antibiotic='vancomycin',
    pathogen='s_aureus',
    dose_range=(500, 2000),
    interval_range=(6, 24),
    duration_hours=120,
    initial_delay=6,
    objective='clearance'
)

# Print optimization results
print("\nOptimized Treatment Regimen:")
print(f"Optimal dose: {optimization_results['optimal_dose']:.1f} mg")
print(f"Optimal interval: {optimization_results['optimal_interval']:.1f} hours")
print(f"Optimization objective: Minimize time to clearance")
print(f"Optimization successful: {optimization_results['convergence']}")
print(f"Number of iterations: {optimization_results['iterations']}")

# Extract results from optimal treatment
optimal_results = optimization_results['treatment_results']
optimal_metrics = optimal_results['metrics']

# Print treatment metrics for optimal regimen
print("\nOptimal Treatment Metrics:")
for key, value in optimal_metrics.items():
    if key != 'time_to_reduction':  # Handle this separately
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            print(f"{key.replace('_', ' ').title()}: {value:.2f}")
        else:
            print(f"{key.replace('_', ' ').title()}: {value}")

# Visualize optimal treatment
times = optimal_results['times']
bacterial_load = optimal_results['bacterial_load']
antibiotic_conc = optimal_results['antibiotic_concentration']
dose_times = optimal_results['dose_times']

# Create figure with two y-axes
fig, ax1 = plt.subplots(figsize=(14, 8))

# Plot bacterial load on log scale (left y-axis)
ax1.semilogy(times, bacterial_load, 'b-', label='Bacterial Load')
ax1.set_xlabel('Time (hours)', fontsize=12)
ax1.set_ylabel('Bacterial Load (CFU/mL)', color='b', fontsize=12)
ax1.tick_params(axis='y', labelcolor='b')

# Create second y-axis for antibiotic concentration
ax2 = ax1.twinx()
ax2.plot(times, antibiotic_conc, 'r-', label='Vancomycin Concentration')
ax2.set_ylabel('Antibiotic Concentration (mg/L)', color='r', fontsize=12)
ax2.tick_params(axis='y', labelcolor='r')

# Add MIC line
if isinstance(infection_modeler.antibiotic_params['vancomycin']['mic'], dict):
    mic = infection_modeler.antibiotic_params['vancomycin']['mic'].get('s_aureus', 1.0)
    ax2.axhline(y=mic, color='r', linestyle='--', alpha=0.7, label=f'MIC')

# Add markers for dose administration
for dose_time in dose_times:
    ax2.axvline(x=dose_time, color='r', linestyle=':', alpha=0.2)

# Add vertical line at treatment initiation
ax1.axvline(x=6, color='green', linestyle='-', alpha=0.7, label='Treatment Start')

# Add legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

plt.title(f'Optimized Treatment: {optimization_results["optimal_dose"]:.1f}mg q{optimization_results["optimal_interval"]:.1f}h', fontsize=14)
plt.grid(True, alpha=0.2)
plt.tight_layout()
plt.show()

## 8. Integration with MIMIC-III Data Analysis

# Load MIMIC-III data
data_path = r"C:\Users\alex5\Documents\Projects\MIMIC_III\biodynamics\data\mimic-iii-clinical-database-demo-1.4"
mimic = MimicPatientIntegrator(data_path)

# Load required tables
print("Loading core tables...")
patient_stays = mimic.load_core_tables()

print("Loading microbiology data...")
micro = mimic.load_table("MICROBIOLOGYEVENTS")

print("Loading prescriptions...")
prescriptions = mimic.load_table("PRESCRIPTIONS")

if micro is not None and prescriptions is not None:
    # Find a patient with both microbiology data and antibiotic prescriptions
    print("\nAnalyzing microbiology and prescription data...")
    
    # Count microbiological cultures by organism
    if 'ORGANISM' in micro.columns:
        organism_counts = micro['ORGANISM'].value_counts()
        
        print("\nMost common organisms in microbiology data:")
        for organism, count in organism_counts.head(5).items():
            print(f"  {organism}: {count} cultures")
    
    # Count antibiotic prescriptions
    if 'DRUG' in prescriptions.columns:
        # Filter for common antibiotics (simplified example)
        common_antibiotics = [
            'VANCOMYCIN', 'CEFTRIAXONE', 'CIPROFLOXACIN', 'PIPERACILLIN',
            'MEROPENEM', 'LEVOFLOXACIN', 'AZITHROMYCIN'
        ]
        
        antibiotic_mask = prescriptions['DRUG'].str.contains('|'.join(common_antibiotics), 
                                                            case=False, 
                                                            na=False)
        antibiotic_prescriptions = prescriptions[antibiotic_mask]
        
        antibiotic_counts = antibiotic_prescriptions['DRUG'].value_counts()
        
        print("\nMost common antibiotic prescriptions:")
        for drug, count in antibiotic_counts.head(5).items():
            print(f"  {drug}: {count} prescriptions")
    
    # Create a synthetic case study based on available data
    print("\nCreating synthetic case study based on MIMIC-III data patterns...")
    
    case_study = {
        'patient_id': 'MIMIC-10001',
        'age': 65,
        'weight': 70,
        'infection_site': 'Bloodstream',
        'organism': 'Staphylococcus aureus',
        'antibiotic': 'vancomycin',
        'renal_function': 'Normal'
    }
    
    print("\nCase Study Details:")
    for key, value in case_study.items():
        print(f"  {key.replace('_', ' ').title()}: {value}")
    
    # Simulate treatment options for the case study
    print("\nSimulating treatment options for the case study...")
    
    # Define regimens based on clinical guidelines
    case_regimens = [
        {'dose': 1000, 'interval': 12},  # Standard
        {'dose': 1500, 'interval': 12},  # High dose
        {'dose': 1000, 'interval': 8},   # Frequent dosing
    ]
    
    case_comparison = infection_modeler.evaluate_multiple_regimens(
        antibiotic='vancomycin',
        pathogen='s_aureus',
        regimens=case_regimens,
        duration_hours=120
    )
    
    # Print case-specific recommendations
    best_idx = case_comparison['best_regimens']['overall']
    best_regimen = case_comparison['all_regimens'][best_idx]
    
    print("\nTreatment Recommendation:")
    print(f"  Recommended regimen: {best_regimen['dose']}mg q{best_regimen['interval']}h")
    if best_regimen['is_successful']:
        print(f"  Expected time to clearance: {best_regimen['time_to_clearance']:.1f} hours")
        print(f"  Expected bacterial reduction: {np.log10(best_regimen['load_reduction']):.1f} log")
    else:
        print("  Warning: Treatment may not be adequate for clearance")
        print("  Consider alternative antibiotics or combination therapy")
else:
    print("\nUnable to perform MIMIC-III data integration - required tables not loaded.")

## 9. Clinical Significance and Applications

print("\n=== Clinical Applications of Infection Treatment Modeling ===")
print("""
1. Personalized Dosing Regimens
   - Optimize antibiotic dosing based on patient-specific factors
   - Account for variations in pharmacokinetics due to age, weight, and organ function
   - Maximize efficacy while minimizing toxicity risk

2. Antimicrobial Stewardship
   - Identify minimum effective dosing to reduce antibiotic overuse
   - Model the impact of delayed therapy on treatment outcomes
   - Compare cost-effectiveness of different treatment strategies

3. Research and Education
   - Visualize complex PK/PD concepts for clinical training
   - Test hypotheses about novel dosing strategies
   - Explore treatment responses for rare pathogens or antibiotics

4. Clinical Decision Support
   - Provide guidance on optimal timing and dosing of antibiotics
   - Alert to potential treatment failures based on predicted responses
   - Suggest alternative regimens when standard approaches may be inadequate
""")

## 10. Summary

print("\n=== BioDynamICS Infection Treatment Modeling Summary ===")
print("""
The infection treatment modeling component of BioDynamICS demonstrates:

1. Integration of pharmacokinetic/pharmacodynamic principles with bacterial growth dynamics
2. Capability to simulate and predict treatment responses for various pathogens and antibiotics
3. Tools for comparing and optimizing antibiotic dosing regimens
4. Framework for analyzing treatment timing and effectiveness
5. Potential for clinical application in personalized medicine and antimicrobial stewardship

This component complements the physiological signal processing and dynamical systems modeling
components by adding the ability to analyze interventions and their effects on physiological
systems, specifically in the context of infection and antimicrobial therapy.

The modeling approach used here could be extended to other therapeutic interventions beyond
antibiotics, providing a general framework for intervention modeling within the BioDynamICS
ecosystem.
""")

print("\nBioDynamICS infection treatment modeling demonstration complete!")
