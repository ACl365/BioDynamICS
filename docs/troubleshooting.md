# BioDynamICS Troubleshooting Guide

This guide addresses common issues that may arise when using the BioDynamICS system and provides solutions to help you resolve them.

## Table of Contents
1. [Installation and Setup Issues](#installation-and-setup-issues)
2. [Data Loading Problems](#data-loading-problems)
3. [Analysis Errors](#analysis-errors)
4. [Visualization Issues](#visualization-issues)
5. [Performance Problems](#performance-problems)
6. [Memory Management](#memory-management)
7. [Error Handling](#error-handling)
8. [Command Line Interface Issues](#command-line-interface-issues)

## Installation and Setup Issues

### ImportError: No module named 'src.system_integration'

**Problem**: Python cannot find the BioDynamICS modules.

**Solution**: 
- Ensure you're running the script from the project root directory.
- Add the project directory to your Python path:
  ```python
  import sys
  sys.path.append('/path/to/biodynamics')
  ```
- Create an empty `__init__.py` file in the `src` directory if it doesn't exist.

### ModuleNotFoundError for dependencies

**Problem**: Missing required dependencies.

**Solution**:
- Install all required dependencies:
  ```bash
  pip install -r requirements.txt
  ```
- If a specific dependency is causing issues, try installing it separately:
  ```bash
  pip install pandas numpy matplotlib scipy scikit-learn
  ```

### Configuration file not found

**Problem**: The system cannot find the configuration file.

**Solution**:
- Initialize the system with a default configuration:
  ```python
  system = BioDynamicsSystem()  # Uses default configuration
  ```
- Create a configuration file manually:
  ```python
  import json
  config = {
      "data_path": "data/mimic-iii-clinical-database-demo-1.4",
      "results_path": "results",
      "cache_enabled": True,
      "cache_path": "cache"
  }
  with open('config.json', 'w') as f:
      json.dump(config, f, indent=4)
  ```
- Use the CLI to initialize the system:
  ```bash
  python biodynamics_cli.py init
  ```

## Data Loading Problems

### FileNotFoundError when loading MIMIC data

**Problem**: The system cannot find the MIMIC data files.

**Solution**:
- Check that the data path in your configuration is correct.
- Verify that the MIMIC data files exist in the specified directory.
- Update the data path in your configuration:
  ```python
  system.update_configuration({"data_path": "correct/path/to/data"})
  ```

### Memory error when loading CHARTEVENTS

**Problem**: The CHARTEVENTS table is too large to load into memory.

**Solution**:
- Use chunked loading with a smaller chunk size:
  ```python
  system.load_chartevents(chunk_size=50000)
  ```
- Filter the data before loading:
  ```python
  # Load only specific patients
  patient_ids = [12345, 67890]
  for patient_id in patient_ids:
      system.create_patient_timeline(patient_id)
  ```
- Use a machine with more RAM or enable disk-based processing.

### Empty or incomplete patient timelines

**Problem**: Patient timelines are missing data or are empty.

**Solution**:
- Ensure all necessary tables are loaded:
  ```python
  system.load_mimic_data(['PATIENTS', 'ADMISSIONS', 'ICUSTAYS', 'CHARTEVENTS', 'LABEVENTS'])
  ```
- Check that the patient ID exists in the dataset:
  ```python
  if subject_id in system.data_integrator.tables['PATIENTS']['subject_id'].values:
      # Patient exists
      pass
  ```
- Verify that the patient has events in the relevant tables.

## Analysis Errors

### KeyError or AttributeError during analysis

**Problem**: Missing data or incorrect data structure.

**Solution**:
- Check that the patient timeline exists and has the expected structure:
  ```python
  if subject_id in system.patient_timelines:
      timeline = system.patient_timelines[subject_id]
      if 'timeline' in timeline and isinstance(timeline['timeline'], pd.DataFrame):
          # Timeline exists and has correct structure
          pass
  ```
- Ensure the timeline contains the necessary columns for the analysis.
- Use try-except blocks to handle potential errors:
  ```python
  try:
      results = system.analyze_physiological_stability(subject_id)
  except Exception as e:
      print(f"Error analyzing stability: {e}")
  ```

### NaN or infinite values in analysis results

**Problem**: Calculations producing invalid results due to missing or invalid data.

**Solution**:
- Preprocess the data to remove or replace NaN values:
  ```python
  # In your code or by modifying the system_integration.py file
  timeline_df = timeline_df.fillna(method='ffill')  # Forward fill
  timeline_df = timeline_df.fillna(method='bfill')  # Backward fill
  timeline_df = timeline_df.fillna(0)  # Replace with zeros
  ```
- Add validation checks before calculations:
  ```python
  if not np.isnan(value) and not np.isinf(value):
      # Proceed with calculation
      pass
  ```

### Incorrect or unexpected analysis results

**Problem**: Analysis producing results that don't match expectations.

**Solution**:
- Verify the input data is correct:
  ```python
  # Print summary statistics
  timeline_df = system.patient_timelines[subject_id]['timeline']
  print(timeline_df.describe())
  ```
- Check for outliers or anomalies in the data:
  ```python
  # Plot histograms of vital signs
  import matplotlib.pyplot as plt
  timeline_df['heart_rate'].hist()
  plt.show()
  ```
- Adjust analysis parameters:
  ```python
  system.update_configuration({
      "analysis_parameters": {
          "window_hours": 12,  # Try different window size
          "step_hours": 4,     # Try different step size
          "stability_threshold": 0.7  # Adjust threshold
      }
  })
  ```

## Visualization Issues

### Matplotlib errors or warnings

**Problem**: Issues with creating or displaying visualizations.

**Solution**:
- Ensure Matplotlib is properly installed:
  ```bash
  pip install matplotlib
  ```
- Use a different backend if needed:
  ```python
  import matplotlib
  matplotlib.use('Agg')  # Use non-interactive backend
  ```
- Close figures after saving to free up resources:
  ```python
  fig = system.visualize_vital_signs(subject_id)
  fig.savefig('vital_signs.png')
  plt.close(fig)
  ```

### Empty or incomplete visualizations

**Problem**: Visualizations missing data or components.

**Solution**:
- Check that the necessary data exists:
  ```python
  timeline_df = system.patient_timelines[subject_id]['timeline']
  if 'heart_rate' in timeline_df.columns and not timeline_df['heart_rate'].isna().all():
      # Data exists for visualization
      pass
  ```
- Ensure the analysis has been performed before visualization:
  ```python
  # Perform analysis before visualization
  system.create_stability_report(subject_id)
  system.visualize_organ_system_status(subject_id)
  ```
- Try different visualization parameters or types.

### Visualization quality issues

**Problem**: Low resolution or poor quality visualizations.

**Solution**:
- Increase DPI for saved figures:
  ```python
  system.update_configuration({
      "visualization_settings": {
          "figure_dpi": 300  # Higher DPI for better quality
      }
  })
  ```
- Use vector formats for better quality:
  ```python
  system.update_configuration({
      "visualization_settings": {
          "figure_format": "svg"  # Vector format
      }
  })
  ```
- Adjust figure size:
  ```python
  fig = system.visualize_vital_signs(subject_id)
  fig.set_size_inches(12, 8)  # Larger figure
  fig.savefig('vital_signs.png')
  ```

## Performance Problems

### Slow data loading

**Problem**: Loading MIMIC data takes too long.

**Solution**:
- Load only necessary tables:
  ```python
  system.load_mimic_data(['PATIENTS', 'ADMISSIONS', 'ICUSTAYS'])
  ```
- Use chunked loading for large tables:
  ```python
  system.load_chartevents(chunk_size=100000)
  ```
- Enable caching:
  ```python
  system.update_configuration({"cache_enabled": True})
  ```
- Consider preprocessing the data and saving it in a more efficient format (e.g., Parquet).

### Slow analysis

**Problem**: Analysis operations take too long.

**Solution**:
- Reduce the analysis window or increase the step size:
  ```python
  system.update_configuration({
      "analysis_parameters": {
          "window_hours": 12,  # Smaller window
          "step_hours": 6      # Larger step
      }
  })
  ```
- Enable parallel processing:
  ```python
  system.update_configuration({
      "parallel_processing": True,
      "max_workers": 8  # Adjust based on your CPU
  })
  ```
- Analyze only specific patients or time periods of interest.

### Slow batch processing

**Problem**: Processing multiple patients takes too long.

**Solution**:
- Enable parallel processing:
  ```python
  batch_results = system.process_patient_batch(
      subject_ids,
      analyses=['timeline', 'signals'],
      parallel=True
  )
  ```
- Limit the analyses performed:
  ```python
  batch_results = system.process_patient_batch(
      subject_ids,
      analyses=['timeline', 'signals']  # Only essential analyses
  )
  ```
- Process smaller batches of patients:
  ```python
  # Process in smaller batches
  for i in range(0, len(all_subject_ids), 10):
      batch = all_subject_ids[i:i+10]
      system.process_patient_batch(batch)
  ```

## Memory Management

### Out of memory errors

**Problem**: System runs out of memory during operation.

**Solution**:
- Process patients one at a time:
  ```python
  for subject_id in subject_ids:
      system.analyze_patient(subject_id)
      # Clear memory after each patient
      if subject_id in system.patient_timelines:
          del system.patient_timelines[subject_id]
      if subject_id in system.patient_analyses:
          del system.patient_analyses[subject_id]
  ```
- Reduce the amount of data loaded:
  ```python
  # Load only essential tables
  system.load_mimic_data(['PATIENTS', 'ADMISSIONS', 'ICUSTAYS'])
  
  # Load CHARTEVENTS only for specific patients
  for subject_id in subject_ids:
      # Create custom query to load only data for this patient
      pass
  ```
- Use disk-based storage for intermediate results:
  ```python
  # Export results to disk and clear memory
  system.export_patient_data(subject_id, f"results/patient_{subject_id}.json")
  if subject_id in system.patient_timelines:
      del system.patient_timelines[subject_id]
  ```
- Increase system swap space or use a machine with more RAM.

### Memory leaks

**Problem**: Memory usage increases over time even when processing the same amount of data.

**Solution**:
- Close matplotlib figures after use:
  ```python
  fig = system.visualize_vital_signs(subject_id)
  fig.savefig('vital_signs.png')
  plt.close(fig)  # Important to prevent memory leaks
  plt.close('all')  # Close all figures
  ```
- Clear caches periodically:
  ```python
  # Clear pandas cache
  import gc
  gc.collect()
  ```
- Restart the system for long-running processes:
  ```python
  # Save state
  system.save_state("system_state.pkl")
  
  # Create new system instance
  del system
  system = BioDynamicsSystem()
  system.load_state("system_state.pkl")
  ```

## Error Handling

### Uncaught exceptions

**Problem**: Exceptions causing the system to crash.

**Solution**:
- Use try-except blocks for error-prone operations:
  ```python
  try:
      results = system.analyze_patient(subject_id)
  except Exception as e:
      print(f"Error analyzing patient {subject_id}: {e}")
      # Continue with next patient or operation
  ```
- Add error handling to the system integration module:
  ```python
  # In your custom version of system_integration.py
  def analyze_patient(self, subject_id, analyses=None):
      try:
          # Existing code
          pass
      except Exception as e:
          self.logger.error(f"Error analyzing patient {subject_id}: {e}")
          return None
  ```
- Check inputs before operations:
  ```python
  if subject_id not in system.patient_timelines:
      print(f"Patient {subject_id} not found")
      return
  ```

### Logging issues

**Problem**: Missing or inadequate logging information.

**Solution**:
- Enable more detailed logging:
  ```python
  import logging
  logging.basicConfig(
      level=logging.DEBUG,
      format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
      filename='biodynamics.log'
  )
  ```
- Add custom logging to operations:
  ```python
  # In your code
  import logging
  logger = logging.getLogger("BioDynamics")
  
  logger.info(f"Starting analysis for patient {subject_id}")
  try:
      results = system.analyze_patient(subject_id)
      logger.info(f"Analysis completed for patient {subject_id}")
  except Exception as e:
      logger.error(f"Error analyzing patient {subject_id}: {e}", exc_info=True)
  ```
- Check log files for errors and warnings.

### Silent failures

**Problem**: Operations fail without errors or warnings.

**Solution**:
- Add validation checks with explicit messages:
  ```python
  if subject_id not in system.patient_timelines:
      print(f"Warning: Patient {subject_id} not found")
      return None
      
  timeline = system.patient_timelines[subject_id]
  if 'timeline' not in timeline or timeline['timeline'].empty:
      print(f"Warning: Empty timeline for patient {subject_id}")
      return None
  ```
- Return status codes or detailed results:
  ```python
  def my_function():
      success = False
      message = "Operation failed"
      result = None
      
      try:
          # Operation code
          result = system.analyze_patient(subject_id)
          success = result is not None
          message = "Operation succeeded" if success else "Operation failed"
      except Exception as e:
          message = f"Error: {str(e)}"
          
      return {
          "success": success,
          "message": message,
          "result": result
      }
  ```
- Enable debug logging to see more details.

## Command Line Interface Issues

### Command not found

**Problem**: The CLI command is not recognized.

**Solution**:
- Ensure you're running the command from the project root directory.
- Make the script executable:
  ```bash
  chmod +x biodynamics_cli.py
  ```
- Use python explicitly:
  ```bash
  python biodynamics_cli.py [command]
  ```
- Add the script to your PATH or create an alias.

### Invalid arguments

**Problem**: The CLI reports invalid arguments.

**Solution**:
- Check the command syntax:
  ```bash
  python biodynamics_cli.py help
  ```
- Use quotes for arguments with spaces:
  ```bash
  python biodynamics_cli.py analyze 12345 --analyses "timeline,signals,stability"
  ```
- Ensure numeric arguments are valid numbers.

### CLI hangs or crashes

**Problem**: The CLI hangs indefinitely or crashes.

**Solution**:
- Run with verbose output:
  ```bash
  python biodynamics_cli.py analyze 12345 --verbose
  ```
- Check log files for errors.
- Run the operation in a Python script to see detailed errors:
  ```python
  from src.system_integration import BioDynamicsSystem
  system = BioDynamicsSystem()
  system.analyze_patient(12345)
  ```
- Try running with a smaller dataset or simpler operation.

## Additional Resources

- [BioDynamICS Documentation](./README_system_integration.md)
- [Example Usage Script](../example_usage.py)
- [Unit Tests](../tests/test_system_integration.py)

If you encounter issues not covered in this guide, please check the logs for detailed error messages and consult the documentation or contact the development team for assistance.