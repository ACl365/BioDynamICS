# BioDynamICS System Integration Module

## Overview

The System Integration Module is the central component of the BioDynamICS framework, unifying all other modules into a cohesive system for analyzing physiological dynamics in critical care. This module provides:

- A high-level API for clinical use
- Batch processing capabilities for multiple patients
- Configurable analysis parameters
- Comprehensive visualization tools
- Data export and persistence functionality

The module integrates the following components:
- Data Integration (loading and processing MIMIC-III data)
- Signal Processing (extracting features from physiological signals)
- Dynamical Systems Modeling (analyzing stability and detecting critical transitions)
- Infection Treatment Modeling (simulating and optimizing antimicrobial therapy)
- Clinical Visualization (creating informative visualizations of patient data)

## Features

### Unified System Architecture
- Seamless integration of all BioDynamICS components
- Consistent API for all functionality
- Centralized configuration management
- Comprehensive logging system

### High-Level Clinical API
- Patient-centric analysis functions
- Stability assessment and monitoring
- Treatment recommendations
- Comprehensive patient summaries

### Batch Processing
- Parallel processing of multiple patients
- Configurable analysis pipeline
- Batch reporting and statistics
- Progress tracking and error handling

### Visualization
- Vital sign timelines
- Organ system status radar charts
- Allostatic load trend analysis
- Phase portraits for dynamical analysis
- Comprehensive patient dashboards

### Data Management
- Export functionality in multiple formats (JSON, CSV, Pickle)
- System state persistence
- Caching for improved performance
- Configurable data paths and settings

## Installation

The System Integration Module is part of the BioDynamICS framework. To use it, ensure you have all the required dependencies installed:

```bash
pip install -r requirements.txt
```

## Usage

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

# Analyze patient
results = system.analyze_patient(subject_id)

# Get patient summary
summary = system.get_patient_summary(subject_id)

# Create visualizations
dashboard = system.create_patient_dashboard(subject_id)
```

### Configuration

The system can be configured using a JSON configuration file:

```python
# Create with custom configuration
config_path = "path/to/config.json"
system = BioDynamicsSystem(config_path=config_path)

# Update configuration
system.update_configuration({
    "parallel_processing": True,
    "max_workers": 8
})

# Save configuration
system.save_configuration("path/to/new_config.json")
```

Example configuration file:

```json
{
    "data_path": "data/mimic-iii-clinical-database-demo-1.4",
    "results_path": "results",
    "cache_enabled": true,
    "cache_path": "cache",
    "parallel_processing": true,
    "max_workers": 4,
    "analysis_parameters": {
        "window_hours": 24,
        "step_hours": 6,
        "embedding_dimension": 3,
        "stability_threshold": 0.5
    },
    "visualization_settings": {
        "save_figures": true,
        "figure_format": "png",
        "figure_dpi": 300
    }
}
```

### Batch Processing

```python
# Process multiple patients
subject_ids = [12345, 67890, 54321]
batch_results = system.process_patient_batch(
    subject_ids,
    analyses=['timeline', 'signals', 'stability', 'dynamical']
)

# Generate batch report
batch_id = list(system.batch_results.keys())[0]
report = system.generate_batch_report(batch_id, "path/to/report.json")
```

### Data Export and Persistence

```python
# Export patient data
system.export_patient_data(subject_id, "path/to/patient_data.json", format='json')

# Save system state
system.save_state("path/to/system_state.pkl")

# Load system state
system.load_state("path/to/system_state.pkl")
```

## API Documentation

### Initialization

```python
BioDynamicsSystem(config_path=None)
```

- `config_path`: Path to configuration file (JSON)

### Data Loading and Integration

```python
load_mimic_data(tables=None, force_reload=False)
```

- `tables`: List of table names to load (default: core tables)
- `force_reload`: Whether to force reload tables even if already loaded

```python
load_chartevents(chunk_size=100000)
```

- `chunk_size`: Size of chunks for processing large files

```python
create_patient_timeline(subject_id, force_recreate=False)
```

- `subject_id`: Patient subject ID
- `force_recreate`: Whether to force recreation of timeline even if it exists

```python
create_all_patient_timelines(max_patients=None)
```

- `max_patients`: Maximum number of patients to process

### Signal Processing and Analysis

```python
process_patient_signals(subject_id, force_reprocess=False)
```

- `subject_id`: Patient subject ID
- `force_reprocess`: Whether to force reprocessing even if results exist

```python
analyze_physiological_stability(subject_id, window_hours=None, step_hours=None, force_reanalyze=False)
```

- `subject_id`: Patient subject ID
- `window_hours`: Size of analysis window in hours
- `step_hours`: Step size for sliding window in hours
- `force_reanalyze`: Whether to force reanalysis even if results exist

```python
create_stability_report(subject_id, force_recreate=False)
```

- `subject_id`: Patient subject ID
- `force_recreate`: Whether to force recreation of report even if it exists

### Dynamical Systems Analysis

```python
analyze_dynamical_stability(subject_id, vital_signs=None, force_reanalyze=False)
```

- `subject_id`: Patient subject ID
- `vital_signs`: List of vital signs to analyze
- `force_reanalyze`: Whether to force reanalysis even if results exist

```python
detect_critical_transitions(subject_id, vital_signs=None, force_redetect=False)
```

- `subject_id`: Patient subject ID
- `vital_signs`: List of vital signs to analyze
- `force_redetect`: Whether to force redetection even if results exist

### Infection and Treatment Analysis

```python
analyze_infection_treatment(subject_id, antibiotic=None, pathogen=None, force_reanalyze=False)
```

- `subject_id`: Patient subject ID
- `antibiotic`: Type of antibiotic to analyze
- `pathogen`: Type of pathogen to analyze
- `force_reanalyze`: Whether to force reanalysis even if results exist

```python
compare_treatment_regimens(subject_id, antibiotic, pathogen, regimens)
```

- `subject_id`: Patient subject ID
- `antibiotic`: Type of antibiotic
- `pathogen`: Type of pathogen
- `regimens`: List of treatment regimens, each with 'dose' and 'interval' keys

### Visualization

```python
visualize_vital_signs(subject_id, vital_signs=None, save_path=None)
```

- `subject_id`: Patient subject ID
- `vital_signs`: List of vital signs to visualize
- `save_path`: Path to save the visualization

```python
visualize_organ_system_status(subject_id, save_path=None)
```

- `subject_id`: Patient subject ID
- `save_path`: Path to save the visualization

```python
visualize_allostatic_load(subject_id, save_path=None)
```

- `subject_id`: Patient subject ID
- `save_path`: Path to save the visualization

```python
visualize_phase_portrait(subject_id, x_measure, y_measure, z_measure=None, save_path=None)
```

- `subject_id`: Patient subject ID
- `x_measure`: Measurement for x-axis
- `y_measure`: Measurement for y-axis
- `z_measure`: Measurement for z-axis (if provided, makes a 3D plot)
- `save_path`: Path to save the visualization

```python
create_patient_dashboard(subject_id, save_path=None)
```

- `subject_id`: Patient subject ID
- `save_path`: Path to save the dashboard

### Batch Processing

```python
process_patient_batch(subject_ids, analyses=None, parallel=None)
```

- `subject_ids`: List of patient subject IDs
- `analyses`: List of analyses to perform (default: all)
- `parallel`: Whether to use parallel processing

```python
generate_batch_report(batch_id, report_path=None)
```

- `batch_id`: Batch ID
- `report_path`: Path to save the report

### Data Export and Persistence

```python
export_patient_data(subject_id, export_path, format='json')
```

- `subject_id`: Patient subject ID
- `export_path`: Path to export the data
- `format`: Export format ('json', 'csv', or 'pickle')

```python
save_state(save_path)
```

- `save_path`: Path to save the state

```python
load_state(load_path)
```

- `load_path`: Path to load the state from

### High-Level Clinical API

```python
analyze_patient(subject_id, analyses=None)
```

- `subject_id`: Patient subject ID
- `analyses`: List of analyses to perform (default: all)

```python
get_patient_stability_status(subject_id)
```

- `subject_id`: Patient subject ID

```python
get_treatment_recommendations(subject_id, infection_type=None)
```

- `subject_id`: Patient subject ID
- `infection_type`: Type of infection

```python
get_patient_summary(subject_id)
```

- `subject_id`: Patient subject ID

## Example Script

See `example_usage.py` for a complete example of how to use the BioDynamicsSystem.

## Configuration Options

### Data Paths
- `data_path`: Path to MIMIC-III data directory
- `results_path`: Path to save results
- `cache_path`: Path to cache directory

### Processing Options
- `cache_enabled`: Whether to enable caching
- `parallel_processing`: Whether to use parallel processing
- `max_workers`: Maximum number of worker threads for parallel processing

### Analysis Parameters
- `window_hours`: Size of analysis window in hours
- `step_hours`: Step size for sliding window in hours
- `embedding_dimension`: Embedding dimension for state space reconstruction
- `stability_threshold`: Threshold for stability classification

### Visualization Settings
- `save_figures`: Whether to save figures
- `figure_format`: Format for saved figures ('png', 'pdf', 'svg', etc.)
- `figure_dpi`: DPI for saved figures