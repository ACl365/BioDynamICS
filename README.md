# BioDynamICS: Physiological Dynamics Analysis Framework

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## Overview

BioDynamICS is a comprehensive framework for analyzing physiological dynamics in critical care data. It integrates advanced signal processing, dynamical systems modeling, and machine learning techniques to extract meaningful insights from complex physiological time series data.

The framework is designed to help clinicians and researchers:

- Analyze physiological stability and allostatic load
- Detect critical transitions and early warning signs
- Model infection treatment dynamics
- Visualize complex physiological relationships
- Generate comprehensive patient reports

BioDynamICS leverages the MIMIC-III critical care database but can be adapted to work with other clinical datasets.

## Features

### Data Integration
- Seamless integration with MIMIC-III database
- Flexible data loading and preprocessing
- Patient timeline creation and management

### Signal Processing
- Advanced filtering and artifact removal
- Feature extraction from physiological signals
- Stability analysis and allostatic load calculation

### Dynamical Systems Modeling
- State space reconstruction
- Lyapunov exponent calculation
- Critical transition detection
- Phase portrait analysis

### Infection Treatment Modeling
- Pharmacokinetic/pharmacodynamic modeling
- Treatment optimization
- Antimicrobial resistance simulation

### Visualization
- Interactive vital sign timelines
- Organ system radar charts
- Phase portraits and state space visualizations
- Comprehensive patient dashboards
- Exportable HTML and PDF reports

### System Integration
- Unified API for all components
- Batch processing capabilities
- Configurable analysis parameters
- Performance optimization

## Installation

### Prerequisites
- Python 3.8+
- MIMIC-III database access (for full functionality)

### Basic Installation

```bash
# Clone the repository
git clone https://github.com/ACl365/BioDynamICS.git
cd BioDynamICS

# Create and activate virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### MIMIC-III Setup

To use BioDynamICS with the MIMIC-III database:

1. Obtain access to MIMIC-III through PhysioNet: [https://physionet.org/content/mimiciii/](https://physionet.org/content/mimiciii/)
2. Download the database files
3. Update the configuration file with the path to your MIMIC-III data

## Quick Start

### Basic Usage

```python
from src.system_integration import BioDynamicsSystem

# Initialize the system
system = BioDynamicsSystem()

# Load MIMIC data
system.load_mimic_data(['PATIENTS', 'ADMISSIONS', 'ICUSTAYS'])

# Create patient timeline
subject_id = 12345  # Replace with actual patient ID
timeline = system.create_patient_timeline(subject_id)

# Process signals
signals = system.process_patient_signals(subject_id)

# Analyze stability
stability = system.analyze_physiological_stability(subject_id)
report = system.create_stability_report(subject_id)

# Create visualizations
vital_signs_fig = system.visualize_vital_signs(subject_id)
organ_system_fig = system.visualize_organ_system_status(subject_id)
dashboard_fig = system.create_patient_dashboard(subject_id)

# Export results
system.export_patient_data(subject_id, "results/patient_data.json")
```

### Command Line Interface

BioDynamICS includes a command-line interface for common operations:

```bash
# Initialize the system
python biodynamics_cli.py init

# Load data
python biodynamics_cli.py load --tables PATIENTS,ADMISSIONS,ICUSTAYS

# Analyze a patient
python biodynamics_cli.py analyze 12345

# Create visualizations
python biodynamics_cli.py visualize 12345 --type all

# Process a batch of patients
python biodynamics_cli.py batch --subject-ids 12345,67890,54321

# Export patient data
python biodynamics_cli.py export 12345 --format json
```

### Enhanced Visualizations

For interactive visualizations:

```python
from src.visualization_enhancements import EnhancedVisualizer

# Create enhanced visualizer
visualizer = EnhancedVisualizer()

# Create interactive visualizations
interactive_vitals = visualizer.create_interactive_vital_signs(timeline['timeline'])
phase_portrait = visualizer.create_interactive_phase_portrait(
    timeline['timeline'], 'heart_rate', 'sbp'
)
dashboard = visualizer.create_interactive_dashboard(timeline, report)

# Export as HTML
visualizer.export_visualization(dashboard, "results/dashboard.html", format="html")

# Create comprehensive report
visualizer.create_html_report(timeline, report, "results/patient_report.html")
```

## Project Structure

```
BioDynamICS/
├── src/                      # Source code
│   ├── data_integration.py   # MIMIC-III data integration
│   ├── signal_processing.py  # Signal processing and feature extraction
│   ├── dynamical_modeling.py # Dynamical systems modeling
│   ├── infection_treatment.py # Infection treatment modeling
│   ├── visualization.py      # Basic visualization
│   ├── visualization_enhancements.py # Enhanced visualizations
│   ├── system_integration.py # System integration
│   ├── code_optimization.py  # Performance optimizations
│   ├── testing_validation.py # Testing and validation
│   └── github_integration.py # GitHub integration
├── tests/                    # Unit and integration tests
├── docs/                     # Documentation
├── biodynamics_cli.py        # Command-line interface
├── example_usage.py          # Example usage script
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Key Innovations

1. **Allostatic Load Quantification**: Integrates multiple physiological systems to provide a single metric of overall physiological stress.

2. **Critical Transition Detection**: Applies dynamical systems theory to detect early warning signs of clinical deterioration before traditional vital sign thresholds are crossed.

3. **Infection Treatment Optimization**: Simulates infection-treatment dynamics and optimizes dosing regimens for improved outcomes.

4. **Enhanced Interactive Visualization**: Creates interactive visualizations that allow clinicians to explore complex physiological data more effectively.

5. **Unified System Integration**: Integrates all components into a cohesive system with a consistent API.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- MIMIC-III database: Johnson, A., et al. (2016). MIMIC-III, a freely accessible critical care database. Scientific Data, 3, 160035.
- PhysioNet: Goldberger, A., et al. (2000). PhysioBank, PhysioToolkit, and PhysioNet: Components of a New Research Resource for Complex Physiologic Signals. Circulation, 101(23), e215-e220.

## Contact

Alexander Clarke - alexanderclarke365@gmail.com

Project Link: [https://github.com/ACl365/BioDynamICS](https://github.com/ACl365/BioDynamICS)