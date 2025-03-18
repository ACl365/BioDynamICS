"""
BioDynamICS - Code Optimization Module

This module provides optimizations for the BioDynamICS system,
including improved error handling, data validation, and
performance enhancements for larger datasets.

Author: Alexander Clarke
Date: March 18, 2025
"""

import os
import pandas as pd
import numpy as np
import time
import logging
import psutil
import functools
import traceback
from typing import Dict, List, Union, Optional, Callable, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

# Import BioDynamICS modules
from src.system_integration import BioDynamicsSystem

class OptimizedBioDynamicsSystem(BioDynamicsSystem):
    """
    Optimized version of the BioDynamicsSystem with improved error handling,
    data validation, and performance optimizations.
    """
    
    def __init__(self, config_path=None):
        """
        Initialize the OptimizedBioDynamicsSystem.
        
        Parameters:
        -----------
        config_path : str, optional
            Path to configuration file (JSON)
        """
        # Call parent constructor
        super().__init__(config_path)
        
        # Set up additional logging
        self.performance_logger = logging.getLogger("BioDynamicsPerformance")
        
        # Initialize performance metrics
        self.performance_metrics = {}
        
        # Initialize validation errors
        self.validation_errors = {}
        
        # Set up memory monitoring
        self.memory_usage = []
        
        self.logger.info("Initialized OptimizedBioDynamicsSystem")
    
    # =========================================================================
    # Decorators for Optimization
    # =========================================================================
    
    @staticmethod
    def validate_input(validation_func):
        """
        Decorator for input validation.
        
        Parameters:
        -----------
        validation_func : callable
            Function that validates the inputs
            
        Returns:
        --------
        callable
            Decorated function
        """
        def decorator(func):
            @functools.wraps(func)
            def wrapper(self, *args, **kwargs):
                # Validate inputs
                is_valid, error_message = validation_func(self, *args, **kwargs)
                
                if not is_valid:
                    # Log validation error
                    self.logger.error(f"Validation error in {func.__name__}: {error_message}")
                    
                    # Store validation error
                    if func.__name__ not in self.validation_errors:
                        self.validation_errors[func.__name__] = []
                    self.validation_errors[func.__name__].append({
                        'args': args,
                        'kwargs': kwargs,
                        'error': error_message,
                        'timestamp': time.time()
                    })
                    
                    # Return None to indicate validation failure
                    return None
                
                # Call the original function
                return func(self, *args, **kwargs)
            return wrapper
        return decorator
    
    @staticmethod
    def measure_performance(func):
        """
        Decorator for measuring function performance.
        
        Parameters:
        -----------
        func : callable
            Function to measure
            
        Returns:
        --------
        callable
            Decorated function
        """
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            # Get function name
            func_name = func.__name__
            
            # Initialize performance metrics for this function
            if func_name not in self.performance_metrics:
                self.performance_metrics[func_name] = {
                    'calls': 0,
                    'total_time': 0,
                    'min_time': float('inf'),
                    'max_time': 0,
                    'avg_time': 0,
                    'memory_before': 0,
                    'memory_after': 0,
                    'memory_diff': 0
                }
            
            # Measure memory usage before
            memory_before = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)  # MB
            
            # Measure execution time
            start_time = time.time()
            
            # Call the original function
            result = func(self, *args, **kwargs)
            
            # Calculate execution time
            execution_time = time.time() - start_time
            
            # Measure memory usage after
            memory_after = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)  # MB
            memory_diff = memory_after - memory_before
            
            # Update performance metrics
            metrics = self.performance_metrics[func_name]
            metrics['calls'] += 1
            metrics['total_time'] += execution_time
            metrics['min_time'] = min(metrics['min_time'], execution_time)
            metrics['max_time'] = max(metrics['max_time'], execution_time)
            metrics['avg_time'] = metrics['total_time'] / metrics['calls']
            metrics['memory_before'] = memory_before
            metrics['memory_after'] = memory_after
            metrics['memory_diff'] = memory_diff
            
            # Log performance metrics
            self.performance_logger.info(
                f"{func_name}: time={execution_time:.4f}s, "
                f"memory_diff={memory_diff:.2f}MB, "
                f"calls={metrics['calls']}"
            )
            
            # Store memory usage
            self.memory_usage.append({
                'timestamp': time.time(),
                'function': func_name,
                'memory_mb': memory_after,
                'memory_diff_mb': memory_diff
            })
            
            return result
        return wrapper
    
    @staticmethod
    def robust_error_handling(func):
        """
        Decorator for robust error handling.
        
        Parameters:
        -----------
        func : callable
            Function to handle errors for
            
        Returns:
        --------
        callable
            Decorated function
        """
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            try:
                # Call the original function
                return func(self, *args, **kwargs)
            except Exception as e:
                # Get function name
                func_name = func.__name__
                
                # Get traceback
                tb = traceback.format_exc()
                
                # Log error with traceback
                self.logger.error(
                    f"Error in {func_name}: {str(e)}\n"
                    f"Args: {args}\n"
                    f"Kwargs: {kwargs}\n"
                    f"Traceback: {tb}"
                )
                
                # Return None to indicate error
                return None
        return wrapper
    
    # =========================================================================
    # Data Validation Functions
    # =========================================================================
    
    def validate_patient_id(self, subject_id):
        """
        Validate a patient subject ID.
        
        Parameters:
        -----------
        subject_id : int
            Patient subject ID
            
        Returns:
        --------
        tuple
            (is_valid, error_message)
        """
        # Check if subject_id is an integer
        if not isinstance(subject_id, int):
            return False, f"Subject ID must be an integer, got {type(subject_id)}"
        
        # Check if subject_id is positive
        if subject_id <= 0:
            return False, f"Subject ID must be positive, got {subject_id}"
        
        # Check if subject_id exists in the dataset
        if 'PATIENTS' in self.data_integrator.tables:
            if subject_id not in self.data_integrator.tables['PATIENTS']['subject_id'].values:
                return False, f"Subject ID {subject_id} not found in PATIENTS table"
        
        return True, ""
    
    def validate_timeline(self, timeline):
        """
        Validate a patient timeline.
        
        Parameters:
        -----------
        timeline : dict
            Patient timeline dictionary
            
        Returns:
        --------
        tuple
            (is_valid, error_message)
        """
        # Check if timeline is a dictionary
        if not isinstance(timeline, dict):
            return False, f"Timeline must be a dictionary, got {type(timeline)}"
        
        # Check if timeline has required keys
        required_keys = ['info', 'timeline']
        for key in required_keys:
            if key not in timeline:
                return False, f"Timeline missing required key: {key}"
        
        # Check if info is a dictionary
        if not isinstance(timeline['info'], dict):
            return False, f"Timeline info must be a dictionary, got {type(timeline['info'])}"
        
        # Check if timeline is a DataFrame
        if not isinstance(timeline['timeline'], pd.DataFrame):
            return False, f"Timeline data must be a DataFrame, got {type(timeline['timeline'])}"
        
        # Check if timeline DataFrame has required columns
        required_columns = ['measurement_time']
        for column in required_columns:
            if column not in timeline['timeline'].columns:
                return False, f"Timeline DataFrame missing required column: {column}"
        
        # Check if timeline DataFrame has data
        if len(timeline['timeline']) == 0:
            return False, "Timeline DataFrame is empty"
        
        return True, ""
    
    def validate_analysis_parameters(self, window_hours, step_hours):
        """
        Validate analysis parameters.
        
        Parameters:
        -----------
        window_hours : int or float
            Size of analysis window in hours
        step_hours : int or float
            Step size for sliding window in hours
            
        Returns:
        --------
        tuple
            (is_valid, error_message)
        """
        # Check if window_hours is a number
        if not isinstance(window_hours, (int, float)):
            return False, f"Window hours must be a number, got {type(window_hours)}"
        
        # Check if step_hours is a number
        if not isinstance(step_hours, (int, float)):
            return False, f"Step hours must be a number, got {type(step_hours)}"
        
        # Check if window_hours is positive
        if window_hours <= 0:
            return False, f"Window hours must be positive, got {window_hours}"
        
        # Check if step_hours is positive
        if step_hours <= 0:
            return False, f"Step hours must be positive, got {step_hours}"
        
        # Check if step_hours is less than or equal to window_hours
        if step_hours > window_hours:
            return False, f"Step hours ({step_hours}) must be less than or equal to window hours ({window_hours})"
        
        return True, ""
    
    # =========================================================================
    # Optimized Data Loading and Integration
    # =========================================================================
    
    @measure_performance
    @robust_error_handling
    def load_mimic_data(self, tables=None, force_reload=False):
        """
        Optimized version of load_mimic_data.
        
        Parameters:
        -----------
        tables : list, optional
            List of table names to load (default: core tables)
        force_reload : bool, optional
            Whether to force reload tables even if already loaded
            
        Returns:
        --------
        bool
            Success status
        """
        # Call parent method
        return super().load_mimic_data(tables, force_reload)
    
    @measure_performance
    @robust_error_handling
    def load_chartevents(self, chunk_size=100000):
        """
        Optimized version of load_chartevents.
        
        Parameters:
        -----------
        chunk_size : int, optional
            Size of chunks for processing large files
            
        Returns:
        --------
        bool
            Success status
        """
        # Call parent method
        return super().load_chartevents(chunk_size)
    
    @measure_performance
    @robust_error_handling
    @validate_input(lambda self, subject_id, force_recreate: self.validate_patient_id(subject_id))
    def create_patient_timeline(self, subject_id, force_recreate=False):
        """
        Optimized version of create_patient_timeline.
        
        Parameters:
        -----------
        subject_id : int
            Patient subject ID
        force_recreate : bool, optional
            Whether to force recreation of timeline even if it exists
            
        Returns:
        --------
        dict
            Patient timeline dictionary
        """
        # Call parent method
        return super().create_patient_timeline(subject_id, force_recreate)
    
    # =========================================================================
    # Optimized Signal Processing and Analysis
    # =========================================================================
    
    @measure_performance
    @robust_error_handling
    @validate_input(lambda self, subject_id, force_reprocess: self.validate_patient_id(subject_id))
    def process_patient_signals(self, subject_id, force_reprocess=False):
        """
        Optimized version of process_patient_signals.
        
        Parameters:
        -----------
        subject_id : int
            Patient subject ID
        force_reprocess : bool, optional
            Whether to force reprocessing even if results exist
            
        Returns:
        --------
        dict
            Signal processing results
        """
        # Call parent method
        return super().process_patient_signals(subject_id, force_reprocess)
    
    @measure_performance
    @robust_error_handling
    @validate_input(lambda self, subject_id, window_hours, step_hours, force_reanalyze: 
                   self.validate_patient_id(subject_id) and 
                   self.validate_analysis_parameters(
                       window_hours or self.config['analysis_parameters']['window_hours'],
                       step_hours or self.config['analysis_parameters']['step_hours']
                   ))
    def analyze_physiological_stability(self, subject_id, window_hours=None, step_hours=None, force_reanalyze=False):
        """
        Optimized version of analyze_physiological_stability.
        
        Parameters:
        -----------
        subject_id : int
            Patient subject ID
        window_hours : int, optional
            Size of analysis window in hours
        step_hours : int, optional
            Step size for sliding window in hours
        force_reanalyze : bool, optional
            Whether to force reanalysis even if results exist
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with stability metrics for each time window
        """
        # Call parent method
        return super().analyze_physiological_stability(subject_id, window_hours, step_hours, force_reanalyze)
    
    # =========================================================================
    # Optimized Batch Processing
    # =========================================================================
    
    @measure_performance
    @robust_error_handling
    def process_patient_batch(self, subject_ids, analyses=None, parallel=None):
        """
        Optimized version of process_patient_batch.
        
        Parameters:
        -----------
        subject_ids : list
            List of patient subject IDs
        analyses : list, optional
            List of analyses to perform (default: all)
        parallel : bool, optional
            Whether to use parallel processing
            
        Returns:
        --------
        dict
            Batch processing results
        """
        # Validate subject_ids
        valid_subject_ids = []
        for subject_id in subject_ids:
            is_valid, _ = self.validate_patient_id(subject_id)
            if is_valid:
                valid_subject_ids.append(subject_id)
            else:
                self.logger.warning(f"Invalid subject ID: {subject_id}, skipping")
        
        if not valid_subject_ids:
            self.logger.error("No valid subject IDs provided")
            return None
        
        # Use process-based parallelism for CPU-bound tasks
        if parallel is None:
            parallel = self.config['parallel_processing']
        
        # Create batch ID
        batch_id = f"batch_{time.time()}"
        
        self.logger.info(f"Starting batch processing {batch_id} for {len(valid_subject_ids)} patients")
        self.logger.info(f"Analyses to perform: {', '.join(analyses) if analyses else 'all'}")
        
        # Initialize batch results
        self.batch_results[batch_id] = {
            'subject_ids': valid_subject_ids,
            'analyses': analyses,
            'start_time': time.time(),
            'end_time': None,
            'completed': 0,
            'failed': 0,
            'results': {}
        }
        
        # Process patients
        if parallel and len(valid_subject_ids) > 1:
            self._process_batch_parallel_optimized(batch_id, valid_subject_ids, analyses)
        else:
            self._process_batch_sequential(batch_id, valid_subject_ids, analyses)
        
        # Update batch results
        self.batch_results[batch_id]['end_time'] = time.time()
        duration = self.batch_results[batch_id]['end_time'] - self.batch_results[batch_id]['start_time']
        
        self.logger.info(f"Batch processing {batch_id} completed in {duration:.1f} seconds")
        self.logger.info(f"Processed {self.batch_results[batch_id]['completed']} patients successfully")
        self.logger.info(f"Failed to process {self.batch_results[batch_id]['failed']} patients")
        
        return self.batch_results[batch_id]
    
    def _process_batch_parallel_optimized(self, batch_id, subject_ids, analyses):
        """
        Process a batch of patients in parallel with optimizations.
        
        Parameters:
        -----------
        batch_id : str
            Batch ID
        subject_ids : list
            List of patient subject IDs
        analyses : list
            List of analyses to perform
        """
        max_workers = self.config['max_workers']
        
        # Use ProcessPoolExecutor for CPU-bound tasks
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit tasks
            future_to_subject = {
                executor.submit(self._perform_analyses_safe, subject_id, analyses): subject_id
                for subject_id in subject_ids
            }
            
            # Process results as they complete
            for future in concurrent.futures.as_completed(future_to_subject):
                subject_id = future_to_subject[future]
                
                try:
                    results = future.result()
                    
                    # Store results
                    self.batch_results[batch_id]['results'][subject_id] = results
                    self.batch_results[batch_id]['completed'] += 1
                    
                    self.logger.info(f"Completed processing patient {subject_id}")
                    
                except Exception as e:
                    self.logger.error(f"Error processing patient {subject_id}: {e}")
                    self.batch_results[batch_id]['failed'] += 1
    
    def _perform_analyses_safe(self, subject_id, analyses):
        """
        Perform requested analyses for a patient with error handling.
        
        Parameters:
        -----------
        subject_id : int
            Patient subject ID
        analyses : list
            List of analyses to perform
            
        Returns:
        --------
        dict
            Analysis results
        """
        results = {}
        
        try:
            # Create patient timeline
            if 'timeline' in analyses:
                timeline = self.create_patient_timeline(subject_id)
                results['timeline'] = timeline is not None
            
            # Process physiological signals
            if 'signals' in analyses:
                signals = self.process_patient_signals(subject_id)
                results['signals'] = signals is not None
            
            # Analyze physiological stability
            if 'stability' in analyses:
                stability = self.analyze_physiological_stability(subject_id)
                stability_report = self.create_stability_report(subject_id)
                results['stability'] = stability is not None and stability_report is not None
            
            # Analyze dynamical stability
            if 'dynamical' in analyses:
                dynamical = self.analyze_dynamical_stability(subject_id)
                results['dynamical'] = dynamical is not None
            
            # Detect critical transitions
            if 'transitions' in analyses:
                transitions = self.detect_critical_transitions(subject_id)
                results['transitions'] = transitions is not None
            
            # Analyze infection treatment
            if 'infection' in analyses:
                infection = self.analyze_infection_treatment(subject_id)
                results['infection'] = infection is not None
        
        except Exception as e:
            self.logger.error(f"Error performing analyses for patient {subject_id}: {e}")
            # Return partial results
        
        return results
    
    # =========================================================================
    # Performance Monitoring and Reporting
    # =========================================================================
    
    def get_performance_report(self):
        """
        Get a report of performance metrics.
        
        Returns:
        --------
        dict
            Performance report
        """
        report = {
            'metrics': self.performance_metrics,
            'memory_usage': self.memory_usage,
            'validation_errors': self.validation_errors,
            'summary': {
                'total_calls': sum(m['calls'] for m in self.performance_metrics.values()),
                'total_time': sum(m['total_time'] for m in self.performance_metrics.values()),
                'slowest_function': max(self.performance_metrics.items(), key=lambda x: x[1]['avg_time'])[0] if self.performance_metrics else None,
                'most_called_function': max(self.performance_metrics.items(), key=lambda x: x[1]['calls'])[0] if self.performance_metrics else None,
                'highest_memory_usage': max(self.memory_usage, key=lambda x: x['memory_mb'])['memory_mb'] if self.memory_usage else 0,
                'validation_error_count': sum(len(errors) for errors in self.validation_errors.values())
            }
        }
        
        return report
    
    def save_performance_report(self, file_path):
        """
        Save performance report to file.
        
        Parameters:
        -----------
        file_path : str
            Path to save the report
            
        Returns:
        --------
        bool
            Success status
        """
        try:
            # Get performance report
            report = self.get_performance_report()
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Save as JSON
            import json
            with open(file_path, 'w') as f:
                json.dump(report, f, indent=4, default=str)
            
            self.logger.info(f"Performance report saved to {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving performance report: {e}")
            return False
    
    def clear_performance_metrics(self):
        """
        Clear performance metrics.
        
        Returns:
        --------
        bool
            Success status
        """
        try:
            self.performance_metrics = {}
            self.memory_usage = []
            self.validation_errors = {}
            
            self.logger.info("Performance metrics cleared")
            return True
            
        except Exception as e:
            self.logger.error(f"Error clearing performance metrics: {e}")
            return False
    
    # =========================================================================
    # Data Validation and Cleaning
    # =========================================================================
    
    def validate_patient_data(self, subject_id):
        """
        Validate and clean patient data.
        
        Parameters:
        -----------
        subject_id : int
            Patient subject ID
            
        Returns:
        --------
        dict
            Validation results
        """
        results = {
            'subject_id': subject_id,
            'timeline_exists': False,
            'timeline_valid': False,
            'missing_values': {},
            'outliers': {},
            'cleaned': False
        }
        
        # Check if patient timeline exists
        if subject_id not in self.patient_timelines:
            self.logger.warning(f"Patient {subject_id} timeline not found")
            return results
        
        results['timeline_exists'] = True
        
        # Validate timeline
        is_valid, error_message = self.validate_timeline(self.patient_timelines[subject_id])
        results['timeline_valid'] = is_valid
        
        if not is_valid:
            self.logger.warning(f"Patient {subject_id} timeline invalid: {error_message}")
            return results
        
        # Get timeline DataFrame
        timeline_df = self.patient_timelines[subject_id]['timeline']
        
        # Check for missing values
        for column in timeline_df.columns:
            missing_count = timeline_df[column].isna().sum()
            if missing_count > 0:
                results['missing_values'][column] = {
                    'count': int(missing_count),
                    'percentage': float(missing_count / len(timeline_df) * 100)
                }
        
        # Check for outliers
        for column in timeline_df.select_dtypes(include=[np.number]).columns:
            # Calculate IQR
            Q1 = timeline_df[column].quantile(0.25)
            Q3 = timeline_df[column].quantile(0.75)
            IQR = Q3 - Q1
            
            # Define outlier bounds
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Count outliers
            outliers = timeline_df[(timeline_df[column] < lower_bound) | (timeline_df[column] > upper_bound)]
            outlier_count = len(outliers)
            
            if outlier_count > 0:
                results['outliers'][column] = {
                    'count': int(outlier_count),
                    'percentage': float(outlier_count / len(timeline_df) * 100),
                    'lower_bound': float(lower_bound),
                    'upper_bound': float(upper_bound)
                }
        
        # Clean data if requested
        try:
            # Create a copy of the timeline
            clean_timeline = self.patient_timelines[subject_id].copy()
            clean_df = clean_timeline['timeline'].copy()
            
            # Handle missing values
            for column in results['missing_values'].keys():
                if clean_df[column].dtype.kind in 'fc':  # Float or complex
                    # Interpolate numeric columns
                    clean_df[column] = clean_df[column].interpolate(method='linear')
                else:
                    # Forward fill non-numeric columns
                    clean_df[column] = clean_df[column].fillna(method='ffill')
                    # Backward fill any remaining NaNs
                    clean_df[column] = clean_df[column].fillna(method='bfill')
            
            # Handle outliers
            for column in results['outliers'].keys():
                lower_bound = results['outliers'][column]['lower_bound']
                upper_bound = results['outliers'][column]['upper_bound']
                
                # Cap outliers at bounds
                clean_df[column] = clean_df[column].clip(lower=lower_bound, upper=upper_bound)
            
            # Update timeline
            clean_timeline['timeline'] = clean_df
            
            # Store cleaned timeline
            self.patient_timelines[subject_id] = clean_timeline
            
            results['cleaned'] = True
            self.logger.info(f"Cleaned patient {subject_id} data")
            
        except Exception as e:
            self.logger.error(f"Error cleaning patient {subject_id} data: {e}")
            results['cleaned'] = False
        
        return results
    
    def validate_all_patients(self):
        """
        Validate and clean data for all patients.
        
        Returns:
        --------
        dict
            Validation results for all patients
        """
        results = {}
        
        for subject_id in list(self.patient_timelines.keys()):
            results[subject_id] = self.validate_patient_data(subject_id)
        
        return results
    
    # =========================================================================
    # Memory Management
    # =========================================================================
    
    def optimize_memory_usage(self):
        """
        Optimize memory usage by converting data types and clearing caches.
        
        Returns:
        --------
        dict
            Memory optimization results
        """
        results = {
            'memory_before': 0,
            'memory_after': 0,
            'memory_saved': 0,
            'optimized_tables': [],
            'cleared_caches': []
        }
        
        # Measure memory usage before
        memory_before = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)  # MB
        results['memory_before'] = memory_before
        
        # Optimize data types in tables
        for table_name, table_df in self.data_integrator.tables.items():
            try:
                # Get memory usage before
                table_memory_before = table_df.memory_usage(deep=True).sum() / (1024 * 1024)  # MB
                
                # Optimize numeric columns
                for column in table_df.select_dtypes(include=[np.number]).columns:
                    # Convert integers to smallest possible type
                    if table_df[column].dtype.kind == 'i':
                        min_val = table_df[column].min()
                        max_val = table_df[column].max()
                        
                        if min_val >= 0:
                            if max_val <= 255:
                                table_df[column] = table_df[column].astype(np.uint8)
                            elif max_val <= 65535:
                                table_df[column] = table_df[column].astype(np.uint16)
                            elif max_val <= 4294967295:
                                table_df[column] = table_df[column].astype(np.uint32)
                        else:
                            if min_val >= -128 and max_val <= 127:
                                table_df[column] = table_df[column].astype(np.int8)
                            elif min_val >= -32768 and max_val <= 32767:
                                table_df[column] = table_df[column].astype(np.int16)
                            elif min_val >= -2147483648 and max_val <= 2147483647:
                                table_df[column] = table_df[column].astype(np.int32)
                    
                    # Convert floats to smallest possible type
                    elif table_df[column].dtype.kind == 'f':
                        # Check if column can be represented as float32
                        if np.finfo(np.float32).min <= table_df[column].min() and table_df[column].max() <= np.finfo(np.float32).max:
                            table_df[column] = table_df[column].astype(np.float32)
                
                # Get memory usage after
                table_memory_after = table_df.memory_usage(deep=True).sum() / (1024 * 1024)  # MB
                
                # Calculate memory saved
                memory_saved = table_memory_before - table_memory_after
                
                if memory_saved > 0:
                    results['optimized_tables'].append({
                        'table': table_name,
                        'memory_before': table_memory_before,
                        'memory_after': table_memory_after,
                        'memory_saved': memory_saved
                    })
                    
                    self.logger.info(f"Optimized table {table_name}: saved {memory_saved:.2f} MB")
            
            except Exception as e:
                self.logger.error(f"Error optimizing table {table_name}: {e}")
        
        # Clear pandas cache
        import gc
        gc.collect()
        results['cleared_caches'].append('pandas')
        
        # Measure memory usage after
        memory_after = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)  # MB
        results['memory_after'] = memory_after
        results['memory_saved'] = memory_before - memory_after
        
        self.logger.info(f"Memory optimization complete: saved {results['memory_saved']:.2f} MB")
        
        return results

# Example usage
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create optimized system
    system = OptimizedBioDynamicsSystem()
    
    # Load data
    system.load_mimic_data(['PATIENTS', 'ADMISSIONS', 'ICUSTAYS'])
    
    # Get performance report
    report = system.get_performance_report()
    print(f"Performance report: {report['summary']}")
    
    # Optimize memory usage
    memory_results = system.optimize_memory_usage()
    print(f"Memory optimization: saved {memory_results['memory_saved']:.2f} MB")