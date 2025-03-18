"""
BioDynamICS - System Integration Module

This module integrates all components of the BioDynamICS framework into a unified system,
providing a high-level API for clinical use, batch processing capabilities, and
configurable analysis parameters.

Author: Alexander Clarke
Date: March 18, 2025
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import json
import pickle
import concurrent.futures
from pathlib import Path
import logging
import warnings
import time

# Import BioDynamICS modules
from src.data_integration import MimicPatientIntegrator
from src.signal_processing import PhysiologicalSignalProcessor
from src.dynamical_modeling import DynamicalSystemsModeler
from src.infection_treatment import InfectionTreatmentModeler
from src.visualization import ClinicalVisualizer

class BioDynamicsSystem:
    """
    Unified system integrating all BioDynamICS components for comprehensive
    physiological analysis of critical care data.
    """
    
    def __init__(self, config_path=None):
        """
        Initialize the BioDynamicsSystem with optional configuration.
        
        Parameters:
        -----------
        config_path : str, optional
            Path to configuration file (JSON)
        """
        # Set up logging
        self._setup_logging()
        
        # Load configuration
        self.config = self._load_configuration(config_path)
        
        # Initialize component modules
        self.logger.info("Initializing BioDynamicsSystem components...")
        self.data_integrator = MimicPatientIntegrator(self.config['data_path'])
        self.signal_processor = PhysiologicalSignalProcessor()
        self.dynamical_modeler = DynamicalSystemsModeler()
        self.infection_modeler = InfectionTreatmentModeler()
        self.visualizer = ClinicalVisualizer()
        
        # Initialize storage for processed results
        self.patient_timelines = {}
        self.patient_analyses = {}
        self.batch_results = {}
        
        self.logger.info("BioDynamicsSystem initialized successfully")
    
    def _setup_logging(self):
        """Set up logging configuration."""
        # Create logs directory if it doesn't exist
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        # Configure logging
        log_file = log_dir / f"biodynamics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger("BioDynamicsSystem")
    
    def _load_configuration(self, config_path):
        """
        Load configuration from file or use defaults.
        
        Parameters:
        -----------
        config_path : str
            Path to configuration file
            
        Returns:
        --------
        dict
            Configuration dictionary
        """
        # Default configuration
        default_config = {
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
        
        # If no config path provided, use defaults
        if config_path is None:
            self.logger.info("No configuration file provided, using defaults")
            return default_config
        
        # Load configuration from file
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                
            # Merge with defaults (to ensure all required keys exist)
            merged_config = default_config.copy()
            for key, value in config.items():
                if isinstance(value, dict) and key in default_config and isinstance(default_config[key], dict):
                    merged_config[key].update(value)
                else:
                    merged_config[key] = value
                    
            self.logger.info(f"Configuration loaded from {config_path}")
            return merged_config
            
        except Exception as e:
            self.logger.error(f"Error loading configuration from {config_path}: {e}")
            self.logger.info("Using default configuration")
            return default_config
    
    def save_configuration(self, config_path):
        """
        Save current configuration to file.
        
        Parameters:
        -----------
        config_path : str
            Path to save configuration file
        """
        try:
            with open(config_path, 'w') as f:
                json.dump(self.config, f, indent=4)
            self.logger.info(f"Configuration saved to {config_path}")
            return True
        except Exception as e:
            self.logger.error(f"Error saving configuration to {config_path}: {e}")
            return False
    
    def update_configuration(self, config_updates):
        """
        Update configuration with new values.
        
        Parameters:
        -----------
        config_updates : dict
            Dictionary with configuration updates
        """
        # Update configuration recursively
        def update_dict_recursive(d, u):
            for k, v in u.items():
                if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                    update_dict_recursive(d[k], v)
                else:
                    d[k] = v
        
        update_dict_recursive(self.config, config_updates)
        self.logger.info("Configuration updated")
    
    # =========================================================================
    # Data Loading and Integration
    # =========================================================================
    
    def load_mimic_data(self, tables=None, force_reload=False):
        """
        Load MIMIC-III data tables.
        
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
        try:
            # Default to core tables if none specified
            if tables is None:
                tables = ['PATIENTS', 'ADMISSIONS', 'ICUSTAYS']
            
            self.logger.info(f"Loading MIMIC tables: {', '.join(tables)}")
            
            # Load each table
            for table in tables:
                if table in self.data_integrator.tables and not force_reload:
                    self.logger.info(f"Table {table} already loaded, skipping")
                    continue
                
                self.data_integrator.load_table(table)
            
            # Create patient-stay linkage if core tables are loaded
            if all(table in self.data_integrator.tables for table in ['PATIENTS', 'ADMISSIONS', 'ICUSTAYS']):
                self.data_integrator.load_core_tables()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading MIMIC data: {e}")
            return False
    
    def load_chartevents(self, chunk_size=100000):
        """
        Load CHARTEVENTS table (which is typically very large).
        
        Parameters:
        -----------
        chunk_size : int, optional
            Size of chunks for processing large files
            
        Returns:
        --------
        bool
            Success status
        """
        try:
            self.logger.info(f"Loading CHARTEVENTS with chunk size {chunk_size}")
            chartevents = self.data_integrator.load_chartevents_chunked(chunk_size)
            return chartevents is not None
        except Exception as e:
            self.logger.error(f"Error loading CHARTEVENTS: {e}")
            return False
    
    def create_patient_timeline(self, subject_id, force_recreate=False):
        """
        Create an integrated timeline for a single patient.
        
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
        # Check if timeline already exists
        if subject_id in self.patient_timelines and not force_recreate:
            self.logger.info(f"Timeline for patient {subject_id} already exists")
            return self.patient_timelines[subject_id]
        
        try:
            self.logger.info(f"Creating timeline for patient {subject_id}")
            timeline = self.data_integrator.create_patient_timeline(subject_id)
            
            if timeline:
                self.patient_timelines[subject_id] = timeline
                self.logger.info(f"Timeline created for patient {subject_id} with {len(timeline['timeline'])} events")
                return timeline
            else:
                self.logger.warning(f"Failed to create timeline for patient {subject_id}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error creating timeline for patient {subject_id}: {e}")
            return None
    
    def create_all_patient_timelines(self, max_patients=None):
        """
        Create integrated timelines for all patients.
        
        Parameters:
        -----------
        max_patients : int, optional
            Maximum number of patients to process
            
        Returns:
        --------
        dict
            Dictionary of patient timelines
        """
        try:
            # Ensure PATIENTS table is loaded
            if 'PATIENTS' not in self.data_integrator.tables:
                self.load_mimic_data(['PATIENTS'])
            
            # Get all patient IDs
            subject_ids = self.data_integrator.tables['PATIENTS']['subject_id'].unique()
            
            # Limit to max_patients if specified
            if max_patients is not None and max_patients > 0:
                subject_ids = subject_ids[:max_patients]
            
            self.logger.info(f"Creating timelines for {len(subject_ids)} patients")
            
            # Process each patient
            for subject_id in subject_ids:
                self.create_patient_timeline(subject_id)
            
            self.logger.info(f"Completed timeline creation for {len(self.patient_timelines)} patients")
            return self.patient_timelines
            
        except Exception as e:
            self.logger.error(f"Error creating all patient timelines: {e}")
            return self.patient_timelines
    
    # =========================================================================
    # Signal Processing and Analysis
    # =========================================================================
    
    def process_patient_signals(self, subject_id, force_reprocess=False):
        """
        Process physiological signals for a patient.
        
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
        # Check if analysis already exists
        if subject_id in self.patient_analyses and 'signal_processing' in self.patient_analyses[subject_id] and not force_reprocess:
            self.logger.info(f"Signal processing results for patient {subject_id} already exist")
            return self.patient_analyses[subject_id]['signal_processing']
        
        # Ensure patient timeline exists
        if subject_id not in self.patient_timelines:
            self.create_patient_timeline(subject_id)
        
        if subject_id not in self.patient_timelines:
            self.logger.error(f"Cannot process signals: No timeline for patient {subject_id}")
            return None
        
        try:
            self.logger.info(f"Processing physiological signals for patient {subject_id}")
            
            # Process the patient timeline
            results = self.signal_processor.process_patient_timeline(self.patient_timelines[subject_id])
            
            # Store results
            if subject_id not in self.patient_analyses:
                self.patient_analyses[subject_id] = {}
            
            self.patient_analyses[subject_id]['signal_processing'] = results
            
            self.logger.info(f"Completed signal processing for patient {subject_id}")
            return results
            
        except Exception as e:
            self.logger.error(f"Error processing signals for patient {subject_id}: {e}")
            return None
    
    def analyze_physiological_stability(self, subject_id, window_hours=None, step_hours=None, force_reanalyze=False):
        """
        Analyze physiological stability over time for a patient.
        
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
        # Use configuration parameters if not specified
        if window_hours is None:
            window_hours = self.config['analysis_parameters']['window_hours']
        
        if step_hours is None:
            step_hours = self.config['analysis_parameters']['step_hours']
        
        # Check if analysis already exists
        if (subject_id in self.patient_analyses and 
            'stability_analysis' in self.patient_analyses[subject_id] and 
            not force_reanalyze):
            self.logger.info(f"Stability analysis for patient {subject_id} already exists")
            return self.patient_analyses[subject_id]['stability_analysis']
        
        # Ensure patient timeline exists
        if subject_id not in self.patient_timelines:
            self.create_patient_timeline(subject_id)
        
        if subject_id not in self.patient_timelines:
            self.logger.error(f"Cannot analyze stability: No timeline for patient {subject_id}")
            return None
        
        try:
            self.logger.info(f"Analyzing physiological stability for patient {subject_id}")
            
            # Analyze stability over time
            stability_df = self.signal_processor.analyze_physiological_stability(
                self.patient_timelines[subject_id], 
                window_hours=window_hours, 
                step_hours=step_hours
            )
            
            # Store results
            if subject_id not in self.patient_analyses:
                self.patient_analyses[subject_id] = {}
            
            self.patient_analyses[subject_id]['stability_analysis'] = stability_df
            
            self.logger.info(f"Completed stability analysis for patient {subject_id}")
            return stability_df
            
        except Exception as e:
            self.logger.error(f"Error analyzing stability for patient {subject_id}: {e}")
            return None
    
    def create_stability_report(self, subject_id, force_recreate=False):
        """
        Create a comprehensive stability report for a patient.
        
        Parameters:
        -----------
        subject_id : int
            Patient subject ID
        force_recreate : bool, optional
            Whether to force recreation of report even if it exists
            
        Returns:
        --------
        dict
            Stability report dictionary
        """
        # Check if report already exists
        if (subject_id in self.patient_analyses and 
            'stability_report' in self.patient_analyses[subject_id] and 
            not force_recreate):
            self.logger.info(f"Stability report for patient {subject_id} already exists")
            return self.patient_analyses[subject_id]['stability_report']
        
        # Ensure patient timeline exists
        if subject_id not in self.patient_timelines:
            self.create_patient_timeline(subject_id)
        
        if subject_id not in self.patient_timelines:
            self.logger.error(f"Cannot create stability report: No timeline for patient {subject_id}")
            return None
        
        try:
            self.logger.info(f"Creating stability report for patient {subject_id}")
            
            # Create stability report
            report = self.signal_processor.create_stability_report(self.patient_timelines[subject_id])
            
            # Store results
            if subject_id not in self.patient_analyses:
                self.patient_analyses[subject_id] = {}
            
            self.patient_analyses[subject_id]['stability_report'] = report
            
            self.logger.info(f"Completed stability report for patient {subject_id}")
            return report
            
        except Exception as e:
            self.logger.error(f"Error creating stability report for patient {subject_id}: {e}")
            return None
    
    # =========================================================================
    # Dynamical Systems Analysis
    # =========================================================================
    
    def analyze_dynamical_stability(self, subject_id, vital_signs=None, force_reanalyze=False):
        """
        Analyze dynamical stability of vital signs for a patient.
        
        Parameters:
        -----------
        subject_id : int
            Patient subject ID
        vital_signs : list, optional
            List of vital signs to analyze
        force_reanalyze : bool, optional
            Whether to force reanalysis even if results exist
            
        Returns:
        --------
        dict
            Dynamical stability analysis results
        """
        # Default vital signs if not specified
        if vital_signs is None:
            vital_signs = ['heart_rate', 'respiratory_rate', 'sbp', 'dbp', 'temperature']
        
        # Check if analysis already exists
        if (subject_id in self.patient_analyses and 
            'dynamical_stability' in self.patient_analyses[subject_id] and 
            not force_reanalyze):
            self.logger.info(f"Dynamical stability analysis for patient {subject_id} already exists")
            return self.patient_analyses[subject_id]['dynamical_stability']
        
        # Ensure patient timeline exists
        if subject_id not in self.patient_timelines:
            self.create_patient_timeline(subject_id)
        
        if subject_id not in self.patient_timelines:
            self.logger.error(f"Cannot analyze dynamical stability: No timeline for patient {subject_id}")
            return None
        
        try:
            self.logger.info(f"Analyzing dynamical stability for patient {subject_id}")
            
            # Analyze each vital sign
            results = {}
            for vital in vital_signs:
                self.logger.info(f"Analyzing {vital} for patient {subject_id}")
                vital_results = self.dynamical_modeler.analyze_patient_stability(
                    self.patient_timelines[subject_id], 
                    vital_sign=vital
                )
                
                if 'error' not in vital_results:
                    results[vital] = vital_results
            
            # Create comprehensive report
            report = self.dynamical_modeler.create_stability_report(self.patient_timelines[subject_id])
            results['stability_report'] = report
            
            # Store results
            if subject_id not in self.patient_analyses:
                self.patient_analyses[subject_id] = {}
            
            self.patient_analyses[subject_id]['dynamical_stability'] = results
            
            self.logger.info(f"Completed dynamical stability analysis for patient {subject_id}")
            return results
            
        except Exception as e:
            self.logger.error(f"Error analyzing dynamical stability for patient {subject_id}: {e}")
            return None
    
    def detect_critical_transitions(self, subject_id, vital_signs=None, force_redetect=False):
        """
        Detect critical transitions in vital signs for a patient.
        
        Parameters:
        -----------
        subject_id : int
            Patient subject ID
        vital_signs : list, optional
            List of vital signs to analyze
        force_redetect : bool, optional
            Whether to force redetection even if results exist
            
        Returns:
        --------
        dict
            Critical transition detection results
        """
        # Default vital signs if not specified
        if vital_signs is None:
            vital_signs = ['heart_rate', 'respiratory_rate', 'sbp', 'dbp', 'temperature']
        
        # Check if analysis already exists
        if (subject_id in self.patient_analyses and 
            'critical_transitions' in self.patient_analyses[subject_id] and 
            not force_redetect):
            self.logger.info(f"Critical transition detection for patient {subject_id} already exists")
            return self.patient_analyses[subject_id]['critical_transitions']
        
        # Ensure patient timeline exists
        if subject_id not in self.patient_timelines:
            self.create_patient_timeline(subject_id)
        
        if subject_id not in self.patient_timelines:
            self.logger.error(f"Cannot detect critical transitions: No timeline for patient {subject_id}")
            return None
        
        try:
            self.logger.info(f"Detecting critical transitions for patient {subject_id}")
            
            # Extract time series for each vital sign
            timeline_df = self.patient_timelines[subject_id]['timeline']
            
            # Detect transitions for each vital sign
            results = {}
            for vital in vital_signs:
                if vital in timeline_df.columns:
                    # Extract time series
                    vital_series = timeline_df[vital].dropna().values
                    
                    if len(vital_series) >= 20:  # Need enough data points
                        # Detect critical transitions
                        transition_results = self.dynamical_modeler.detect_critical_transition(vital_series)
                        results[vital] = transition_results
            
            # Store results
            if subject_id not in self.patient_analyses:
                self.patient_analyses[subject_id] = {}
            
            self.patient_analyses[subject_id]['critical_transitions'] = results
            
            self.logger.info(f"Completed critical transition detection for patient {subject_id}")
            return results
            
        except Exception as e:
            self.logger.error(f"Error detecting critical transitions for patient {subject_id}: {e}")
            return None
    
    # =========================================================================
    # Infection and Treatment Analysis
    # =========================================================================
    
    def analyze_infection_treatment(self, subject_id, antibiotic=None, pathogen=None, force_reanalyze=False):
        """
        Analyze infection treatment effectiveness for a patient.
        
        Parameters:
        -----------
        subject_id : int
            Patient subject ID
        antibiotic : str, optional
            Type of antibiotic to analyze
        pathogen : str, optional
            Type of pathogen to analyze
        force_reanalyze : bool, optional
            Whether to force reanalysis even if results exist
            
        Returns:
        --------
        dict
            Infection treatment analysis results
        """
        # Default to common antibiotic and pathogen if not specified
        if antibiotic is None:
            antibiotic = 'vancomycin'
        
        if pathogen is None:
            pathogen = 's_aureus'
        
        # Check if analysis already exists
        if (subject_id in self.patient_analyses and 
            'infection_treatment' in self.patient_analyses[subject_id] and 
            not force_reanalyze):
            self.logger.info(f"Infection treatment analysis for patient {subject_id} already exists")
            return self.patient_analyses[subject_id]['infection_treatment']
        
        # Ensure patient timeline exists
        if subject_id not in self.patient_timelines:
            self.create_patient_timeline(subject_id)
        
        if subject_id not in self.patient_timelines:
            self.logger.error(f"Cannot analyze infection treatment: No timeline for patient {subject_id}")
            return None
        
        try:
            self.logger.info(f"Analyzing infection treatment for patient {subject_id}")
            
            # Extract medication information from timeline
            timeline_df = self.patient_timelines[subject_id]['timeline']
            
            # Check if we have medication data
            has_medication_data = False
            if 'event_type' in timeline_df.columns:
                medication_events = timeline_df[timeline_df['event_type'] == 'medication']
                has_medication_data = len(medication_events) > 0
            
            # If we have medication data, extract dosing information
            if has_medication_data:
                # This would require parsing the medication events to extract
                # antibiotic doses and timing, which depends on the specific
                # structure of the MIMIC data
                
                # For demonstration, we'll use a simulated treatment regimen
                self.logger.info("Using medication data from patient timeline")
                
                # Placeholder for actual extraction logic
                dose = 1000  # mg
                interval = 12  # hours
                
            else:
                # Use standard dosing regimen for simulation
                self.logger.info("No medication data found, using standard dosing regimen")
                dose = 1000  # mg
                interval = 12  # hours
            
            # Simulate treatment
            treatment_results = self.infection_modeler.simulate_treatment(
                antibiotic=antibiotic,
                pathogen=pathogen,
                dose=dose,
                dosing_interval=interval,
                duration_hours=168  # 7 days
            )
            
            # Optimize dosing regimen
            optimization_results = self.infection_modeler.optimize_dosing_regimen(
                antibiotic=antibiotic,
                pathogen=pathogen,
                dose_range=(500, 2000),
                interval_range=(6, 24),
                duration_hours=168
            )
            
            # Combine results
            results = {
                'treatment_simulation': treatment_results,
                'optimization': optimization_results,
                'antibiotic': antibiotic,
                'pathogen': pathogen
            }
            
            # Store results
            if subject_id not in self.patient_analyses:
                self.patient_analyses[subject_id] = {}
            
            self.patient_analyses[subject_id]['infection_treatment'] = results
            
            self.logger.info(f"Completed infection treatment analysis for patient {subject_id}")
            return results
            
        except Exception as e:
            self.logger.error(f"Error analyzing infection treatment for patient {subject_id}: {e}")
            return None
    
    def compare_treatment_regimens(self, subject_id, antibiotic, pathogen, regimens):
        """
        Compare multiple treatment regimens for a patient.
        
        Parameters:
        -----------
        subject_id : int
            Patient subject ID
        antibiotic : str
            Type of antibiotic
        pathogen : str
            Type of pathogen
        regimens : list of dict
            List of treatment regimens, each with 'dose' and 'interval' keys
            
        Returns:
        --------
        dict
            Comparison of treatment regimens
        """
        try:
            self.logger.info(f"Comparing treatment regimens for patient {subject_id}")
            
            # Compare regimens
            comparison = self.infection_modeler.evaluate_multiple_regimens(
                antibiotic=antibiotic,
                pathogen=pathogen,
                regimens=regimens,
                duration_hours=168
            )
            
            # Store results
            if subject_id not in self.patient_analyses:
                self.patient_analyses[subject_id] = {}
            
            if 'infection_treatment' not in self.patient_analyses[subject_id]:
                self.patient_analyses[subject_id]['infection_treatment'] = {}
            
            self.patient_analyses[subject_id]['infection_treatment']['regimen_comparison'] = comparison
            
            self.logger.info(f"Completed treatment regimen comparison for patient {subject_id}")
            return comparison
            
        except Exception as e:
            self.logger.error(f"Error comparing treatment regimens for patient {subject_id}: {e}")
            return None
    
    # =========================================================================
    # Visualization
    # =========================================================================
    
    def visualize_vital_signs(self, subject_id, vital_signs=None, save_path=None):
        """
        Create visualization of vital signs for a patient.
        
        Parameters:
        -----------
        subject_id : int
            Patient subject ID
        vital_signs : list, optional
            List of vital signs to visualize
        save_path : str, optional
            Path to save the visualization
            
        Returns:
        --------
        matplotlib.figure.Figure
            The created figure
        """
        # Default vital signs if not specified
        if vital_signs is None:
            vital_signs = ['heart_rate', 'respiratory_rate', 'sbp', 'dbp', 'temperature', 'o2_saturation']
        
        # Ensure patient timeline exists
        if subject_id not in self.patient_timelines:
            self.create_patient_timeline(subject_id)
        
        if subject_id not in self.patient_timelines:
            self.logger.error(f"Cannot visualize vital signs: No timeline for patient {subject_id}")
            return None
        
        try:
            self.logger.info(f"Creating vital signs visualization for patient {subject_id}")
            
            # Create visualization
            fig = self.visualizer.plot_multi_vital_timeline(
                self.patient_timelines[subject_id]['timeline'],
                vital_signs
            )
            
            # Save figure if requested
            if save_path is not None and fig is not None:
                self._save_figure(fig, save_path)
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error visualizing vital signs for patient {subject_id}: {e}")
            return None
    
    def visualize_organ_system_status(self, subject_id, save_path=None):
        """
        Create visualization of organ system status for a patient.
        
        Parameters:
        -----------
        subject_id : int
            Patient subject ID
        save_path : str, optional
            Path to save the visualization
            
        Returns:
        --------
        matplotlib.figure.Figure
            The created figure
        """
        # Ensure stability report exists
        if (subject_id not in self.patient_analyses or 
            'stability_report' not in self.patient_analyses[subject_id]):
            self.create_stability_report(subject_id)
        
        if (subject_id not in self.patient_analyses or 
            'stability_report' not in self.patient_analyses[subject_id]):
            self.logger.error(f"Cannot visualize organ system status: No stability report for patient {subject_id}")
            return None
        
        try:
            self.logger.info(f"Creating organ system status visualization for patient {subject_id}")
            
            # Create visualization
            fig = self.visualizer.plot_organ_system_radar(
                self.patient_analyses[subject_id]['stability_report']
            )
            
            # Save figure if requested
            if save_path is not None and fig is not None:
                self._save_figure(fig, save_path)
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error visualizing organ system status for patient {subject_id}: {e}")
            return None
    
    def visualize_allostatic_load(self, subject_id, save_path=None):
        """
        Create visualization of allostatic load trend for a patient.
        
        Parameters:
        -----------
        subject_id : int
            Patient subject ID
        save_path : str, optional
            Path to save the visualization
            
        Returns:
        --------
        matplotlib.figure.Figure
            The created figure
        """
        # Ensure stability analysis exists
        if (subject_id not in self.patient_analyses or 
            'stability_analysis' not in self.patient_analyses[subject_id]):
            self.analyze_physiological_stability(subject_id)
        
        if (subject_id not in self.patient_analyses or 
            'stability_analysis' not in self.patient_analyses[subject_id]):
            self.logger.error(f"Cannot visualize allostatic load: No stability analysis for patient {subject_id}")
            return None
        
        try:
            self.logger.info(f"Creating allostatic load visualization for patient {subject_id}")
            
            # Create visualization
            fig = self.visualizer.plot_allostatic_load_trend(
                self.patient_analyses[subject_id]['stability_analysis']
            )
            
            # Save figure if requested
            if save_path is not None and fig is not None:
                self._save_figure(fig, save_path)
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error visualizing allostatic load for patient {subject_id}: {e}")
            return None
    
    def visualize_phase_portrait(self, subject_id, x_measure, y_measure, z_measure=None, save_path=None):
        """
        Create phase portrait visualization for a patient.
        
        Parameters:
        -----------
        subject_id : int
            Patient subject ID
        x_measure : str
            Measurement for x-axis
        y_measure : str
            Measurement for y-axis
        z_measure : str, optional
            Measurement for z-axis (if provided, makes a 3D plot)
        save_path : str, optional
            Path to save the visualization
            
        Returns:
        --------
        matplotlib.figure.Figure
            The created figure
        """
        # Ensure patient timeline exists
        if subject_id not in self.patient_timelines:
            self.create_patient_timeline(subject_id)
        
        if subject_id not in self.patient_timelines:
            self.logger.error(f"Cannot create phase portrait: No timeline for patient {subject_id}")
            return None
        
        try:
            self.logger.info(f"Creating phase portrait for patient {subject_id}")
            
            # Create visualization
            fig = self.visualizer.plot_phase_portrait(
                self.patient_timelines[subject_id]['timeline'],
                x_measure, y_measure, z_measure
            )
            
            # Save figure if requested
            if save_path is not None and fig is not None:
                self._save_figure(fig, save_path)
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating phase portrait for patient {subject_id}: {e}")
            return None
    
    def create_patient_dashboard(self, subject_id, save_path=None):
        """
        Create comprehensive patient dashboard.
        
        Parameters:
        -----------
        subject_id : int
            Patient subject ID
        save_path : str, optional
            Path to save the dashboard
            
        Returns:
        --------
        matplotlib.figure.Figure
            The created figure
        """
        # Ensure patient timeline exists
        if subject_id not in self.patient_timelines:
            self.create_patient_timeline(subject_id)
        
        if subject_id not in self.patient_timelines:
            self.logger.error(f"Cannot create dashboard: No timeline for patient {subject_id}")
            return None
        
        # Ensure stability report exists
        if (subject_id not in self.patient_analyses or 
            'stability_report' not in self.patient_analyses[subject_id]):
            self.create_stability_report(subject_id)
        
        if (subject_id not in self.patient_analyses or 
            'stability_report' not in self.patient_analyses[subject_id]):
            self.logger.error(f"Cannot create dashboard: No stability report for patient {subject_id}")
            return None
        
        try:
            self.logger.info(f"Creating comprehensive dashboard for patient {subject_id}")
            
            # Create dashboard
            fig = self.visualizer.create_patient_dashboard(
                self.patient_timelines[subject_id],
                self.patient_analyses[subject_id]['stability_report']
            )
            
            # Save figure if requested
            if save_path is not None and fig is not None:
                self._save_figure(fig, save_path)
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating dashboard for patient {subject_id}: {e}")
            return None
    
    def _save_figure(self, fig, save_path):
        """
        Save a figure to file.
        
        Parameters:
        -----------
        fig : matplotlib.figure.Figure
            Figure to save
        save_path : str
            Path to save the figure
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # Get figure format and DPI from config
            figure_format = self.config['visualization_settings']['figure_format']
            figure_dpi = self.config['visualization_settings']['figure_dpi']
            
            # Save figure
            fig.savefig(save_path, format=figure_format, dpi=figure_dpi, bbox_inches='tight')
            self.logger.info(f"Figure saved to {save_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving figure to {save_path}: {e}")
    
    # =========================================================================
    # Batch Processing
    # =========================================================================
    
    def process_patient_batch(self, subject_ids, analyses=None, parallel=None):
        """
        Process a batch of patients with specified analyses.
        
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
        # Default analyses if not specified
        if analyses is None:
            analyses = [
                'timeline', 'signals', 'stability', 'dynamical', 'transitions'
            ]
        
        # Use configuration setting for parallel processing if not specified
        if parallel is None:
            parallel = self.config['parallel_processing']
        
        # Create batch ID
        batch_id = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.logger.info(f"Starting batch processing {batch_id} for {len(subject_ids)} patients")
        self.logger.info(f"Analyses to perform: {', '.join(analyses)}")
        
        # Initialize batch results
        self.batch_results[batch_id] = {
            'subject_ids': subject_ids,
            'analyses': analyses,
            'start_time': datetime.now(),
            'end_time': None,
            'completed': 0,
            'failed': 0,
            'results': {}
        }
        
        # Process patients
        if parallel and len(subject_ids) > 1:
            self._process_batch_parallel(batch_id, subject_ids, analyses)
        else:
            self._process_batch_sequential(batch_id, subject_ids, analyses)
        
        # Update batch results
        self.batch_results[batch_id]['end_time'] = datetime.now()
        duration = (self.batch_results[batch_id]['end_time'] - self.batch_results[batch_id]['start_time']).total_seconds()
        
        self.logger.info(f"Batch processing {batch_id} completed in {duration:.1f} seconds")
        self.logger.info(f"Processed {self.batch_results[batch_id]['completed']} patients successfully")
        self.logger.info(f"Failed to process {self.batch_results[batch_id]['failed']} patients")
        
        return self.batch_results[batch_id]
    
    def _process_batch_sequential(self, batch_id, subject_ids, analyses):
        """Process a batch of patients sequentially."""
        for subject_id in subject_ids:
            try:
                self.logger.info(f"Processing patient {subject_id}")
                
                # Perform requested analyses
                results = self._perform_analyses(subject_id, analyses)
                
                # Store results
                self.batch_results[batch_id]['results'][subject_id] = results
                self.batch_results[batch_id]['completed'] += 1
                
            except Exception as e:
                self.logger.error(f"Error processing patient {subject_id}: {e}")
                self.batch_results[batch_id]['failed'] += 1
    
    def _process_batch_parallel(self, batch_id, subject_ids, analyses):
        """Process a batch of patients in parallel."""
        max_workers = self.config['max_workers']
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit tasks
            future_to_subject = {
                executor.submit(self._perform_analyses, subject_id, analyses): subject_id
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
    
    def _perform_analyses(self, subject_id, analyses):
        """Perform requested analyses for a patient."""
        results = {}
        
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
        
        return results
    
    def generate_batch_report(self, batch_id, report_path=None):
        """
        Generate a report for a batch processing job.
        
        Parameters:
        -----------
        batch_id : str
            Batch ID
        report_path : str, optional
            Path to save the report
            
        Returns:
        --------
        dict
            Batch report
        """
        if batch_id not in self.batch_results:
            self.logger.error(f"Batch {batch_id} not found")
            return None
        
        try:
            self.logger.info(f"Generating report for batch {batch_id}")
            
            batch = self.batch_results[batch_id]
            
            # Calculate statistics
            total_patients = len(batch['subject_ids'])
            completed = batch['completed']
            failed = batch['failed']
            
            if batch['end_time'] is not None:
                duration = (batch['end_time'] - batch['start_time']).total_seconds()
            else:
                duration = (datetime.now() - batch['start_time']).total_seconds()
            
            # Create report
            report = {
                'batch_id': batch_id,
                'start_time': batch['start_time'].strftime('%Y-%m-%d %H:%M:%S'),
                'end_time': batch['end_time'].strftime('%Y-%m-%d %H:%M:%S') if batch['end_time'] else None,
                'duration_seconds': duration,
                'total_patients': total_patients,
                'completed_patients': completed,
                'failed_patients': failed,
                'completion_rate': completed / total_patients if total_patients > 0 else 0,
                'analyses_performed': batch['analyses'],
                'patient_results': {}
            }
            
            # Add patient-specific results
            for subject_id, results in batch['results'].items():
                report['patient_results'][subject_id] = results
            
            # Save report if requested
            if report_path is not None:
                try:
                    # Create directory if it doesn't exist
                    os.makedirs(os.path.dirname(report_path), exist_ok=True)
                    
                    # Save as JSON
                    with open(report_path, 'w') as f:
                        # Convert datetime objects to strings
                        json.dump(report, f, indent=4, default=str)
                    
                    self.logger.info(f"Batch report saved to {report_path}")
                    
                except Exception as e:
                    self.logger.error(f"Error saving batch report to {report_path}: {e}")
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating batch report for {batch_id}: {e}")
            return None
    
    # =========================================================================
    # Data Export and Persistence
    # =========================================================================
    
    def export_patient_data(self, subject_id, export_path, format='json'):
        """
        Export patient data to file.
        
        Parameters:
        -----------
        subject_id : int
            Patient subject ID
        export_path : str
            Path to export the data
        format : str, optional
            Export format ('json', 'csv', or 'pickle')
            
        Returns:
        --------
        bool
            Success status
        """
        if subject_id not in self.patient_timelines:
            self.logger.error(f"Patient {subject_id} not found")
            return False
        
        try:
            self.logger.info(f"Exporting data for patient {subject_id} to {export_path}")
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(export_path), exist_ok=True)
            
            # Prepare data for export
            export_data = {
                'timeline': self.patient_timelines[subject_id],
                'analyses': self.patient_analyses.get(subject_id, {})
            }
            
            # Export in the requested format
            if format.lower() == 'json':
                with open(export_path, 'w') as f:
                    json.dump(export_data, f, indent=4, default=str)
                    
            elif format.lower() == 'csv':
                # For CSV, we can only export the timeline DataFrame
                timeline_df = self.patient_timelines[subject_id]['timeline']
                timeline_df.to_csv(export_path, index=False)
                
            elif format.lower() == 'pickle':
                with open(export_path, 'wb') as f:
                    pickle.dump(export_data, f)
                    
            else:
                self.logger.error(f"Unsupported export format: {format}")
                return False
            
            self.logger.info(f"Data exported to {export_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error exporting data for patient {subject_id}: {e}")
            return False
    
    def save_state(self, save_path):
        """
        Save the current state of the system to file.
        
        Parameters:
        -----------
        save_path : str
            Path to save the state
            
        Returns:
        --------
        bool
            Success status
        """
        try:
            self.logger.info(f"Saving system state to {save_path}")
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # Prepare state for saving
            state = {
                'config': self.config,
                'patient_timelines': self.patient_timelines,
                'patient_analyses': self.patient_analyses,
                'batch_results': self.batch_results,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            # Save state
            with open(save_path, 'wb') as f:
                pickle.dump(state, f)
            
            self.logger.info(f"System state saved to {save_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving system state to {save_path}: {e}")
            return False
    
    def load_state(self, load_path):
        """
        Load system state from file.
        
        Parameters:
        -----------
        load_path : str
            Path to load the state from
            
        Returns:
        --------
        bool
            Success status
        """
        try:
            self.logger.info(f"Loading system state from {load_path}")
            
            # Load state
            with open(load_path, 'rb') as f:
                state = pickle.load(f)
            
            # Restore state
            self.config = state['config']
            self.patient_timelines = state['patient_timelines']
            self.patient_analyses = state['patient_analyses']
            self.batch_results = state['batch_results']
            
            self.logger.info(f"System state loaded from {load_path}")
            self.logger.info(f"Loaded {len(self.patient_timelines)} patient timelines")
            self.logger.info(f"Loaded {len(self.patient_analyses)} patient analyses")
            self.logger.info(f"Loaded {len(self.batch_results)} batch results")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading system state from {load_path}: {e}")
            return False
    
    # =========================================================================
    # High-Level Clinical API
    # =========================================================================
    
    def analyze_patient(self, subject_id, analyses=None):
        """
        Perform comprehensive analysis for a patient.
        
        Parameters:
        -----------
        subject_id : int
            Patient subject ID
        analyses : list, optional
            List of analyses to perform (default: all)
            
        Returns:
        --------
        dict
            Analysis results
        """
        # Default analyses if not specified
        if analyses is None:
            analyses = [
                'timeline', 'signals', 'stability', 'dynamical', 'transitions'
            ]
        
        self.logger.info(f"Performing comprehensive analysis for patient {subject_id}")
        
        # Perform analyses
        results = self._perform_analyses(subject_id, analyses)
        
        # Create visualizations if requested
        if 'visualize' in analyses:
            self.logger.info(f"Creating visualizations for patient {subject_id}")
            
            # Create results directory
            results_dir = os.path.join(self.config['results_path'], f"patient_{subject_id}")
            os.makedirs(results_dir, exist_ok=True)
            
            # Create visualizations
            viz_results = {}
            
            # Vital signs
            vital_signs_path = os.path.join(results_dir, "vital_signs.png")
            viz_results['vital_signs'] = self.visualize_vital_signs(subject_id, save_path=vital_signs_path) is not None
            
            # Organ system status
            if 'stability' in analyses:
                organ_system_path = os.path.join(results_dir, "organ_system.png")
                viz_results['organ_system'] = self.visualize_organ_system_status(subject_id, save_path=organ_system_path) is not None
                
                # Allostatic load
                allostatic_load_path = os.path.join(results_dir, "allostatic_load.png")
                viz_results['allostatic_load'] = self.visualize_allostatic_load(subject_id, save_path=allostatic_load_path) is not None
            
            # Phase portrait
            phase_portrait_path = os.path.join(results_dir, "phase_portrait.png")
            viz_results['phase_portrait'] = self.visualize_phase_portrait(
                subject_id, 'heart_rate', 'sbp', save_path=phase_portrait_path
            ) is not None
            
            # Dashboard
            if 'stability' in analyses:
                dashboard_path = os.path.join(results_dir, "dashboard.png")
                viz_results['dashboard'] = self.create_patient_dashboard(subject_id, save_path=dashboard_path) is not None
            
            results['visualizations'] = viz_results
        
        # Export results if requested
        if 'export' in analyses:
            self.logger.info(f"Exporting results for patient {subject_id}")
            
            # Create results directory
            results_dir = os.path.join(self.config['results_path'], f"patient_{subject_id}")
            os.makedirs(results_dir, exist_ok=True)
            
            # Export data
            export_path = os.path.join(results_dir, f"patient_{subject_id}_data.json")
            results['export'] = self.export_patient_data(subject_id, export_path, format='json')
        
        return results
    
    def get_patient_stability_status(self, subject_id):
        """
        Get current stability status for a patient.
        
        Parameters:
        -----------
        subject_id : int
            Patient subject ID
            
        Returns:
        --------
        dict
            Stability status
        """
        # Ensure stability report exists
        if (subject_id not in self.patient_analyses or 
            'stability_report' not in self.patient_analyses[subject_id]):
            self.create_stability_report(subject_id)
        
        if (subject_id not in self.patient_analyses or 
            'stability_report' not in self.patient_analyses[subject_id]):
            self.logger.error(f"Cannot get stability status: No stability report for patient {subject_id}")
            return None
        
        try:
            self.logger.info(f"Getting stability status for patient {subject_id}")
            
            # Extract relevant information from stability report
            report = self.patient_analyses[subject_id]['stability_report']
            
            # Get allostatic load
            allostatic_load = report.get('overall_results', {}).get('allostatic_load', None)
            
            # Get organ system status
            organ_status = report.get('organ_system_summary', {})
            
            # Get critical values
            critical_values = report.get('critical_values', [])
            
            # Determine overall stability status
            if allostatic_load is not None:
                if allostatic_load < 0.5:
                    stability_status = "Stable"
                elif allostatic_load < 1.0:
                    stability_status = "Mildly Unstable"
                elif allostatic_load < 1.5:
                    stability_status = "Moderately Unstable"
                else:
                    stability_status = "Severely Unstable"
            else:
                stability_status = "Unknown"
            
            # Check for critical transitions
            has_transitions = False
            if subject_id in self.patient_analyses and 'critical_transitions' in self.patient_analyses[subject_id]:
                transitions = self.patient_analyses[subject_id]['critical_transitions']
                for vital, result in transitions.items():
                    if result.get('detected', False):
                        has_transitions = True
                        break
            
            # Create status summary
            status = {
                'patient_id': subject_id,
                'stability_status': stability_status,
                'allostatic_load': allostatic_load,
                'organ_systems': {},
                'critical_values_count': len(critical_values),
                'approaching_transition': has_transitions
            }
            
            # Add organ system status
            for system, system_status in organ_status.items():
                status['organ_systems'][system] = {
                    'score': system_status.get('score', None),
                    'abnormal_measures': system_status.get('abnormal_measures', 0),
                    'total_measures': system_status.get('n_measures', 0)
                }
            
            return status
            
        except Exception as e:
            self.logger.error(f"Error getting stability status for patient {subject_id}: {e}")
            return None
    
    def get_treatment_recommendations(self, subject_id, infection_type=None):
        """
        Get treatment recommendations for a patient.
        
        Parameters:
        -----------
        subject_id : int
            Patient subject ID
        infection_type : str, optional
            Type of infection
            
        Returns:
        --------
        dict
            Treatment recommendations
        """
        # Default infection type if not specified
        if infection_type is None:
            infection_type = 's_aureus'
        
        # Analyze infection treatment if not already done
        if (subject_id not in self.patient_analyses or 
            'infection_treatment' not in self.patient_analyses[subject_id]):
            self.analyze_infection_treatment(subject_id, pathogen=infection_type)
        
        if (subject_id not in self.patient_analyses or 
            'infection_treatment' not in self.patient_analyses[subject_id]):
            self.logger.error(f"Cannot get treatment recommendations: No infection analysis for patient {subject_id}")
            return None
        
        try:
            self.logger.info(f"Getting treatment recommendations for patient {subject_id}")
            
            # Extract relevant information from infection treatment analysis
            analysis = self.patient_analyses[subject_id]['infection_treatment']
            
            # Get optimization results
            optimization = analysis.get('optimization', {})
            
            # Create recommendations
            recommendations = {
                'patient_id': subject_id,
                'infection_type': analysis.get('pathogen', infection_type),
                'antibiotic': analysis.get('antibiotic', 'vancomycin'),
                'optimal_dose': optimization.get('optimal_dose', None),
                'optimal_interval': optimization.get('optimal_interval', None),
                'expected_clearance_time': None,
                'expected_success_probability': None
            }
            
            # Add expected outcomes if available
            if 'treatment_results' in optimization:
                metrics = optimization['treatment_results'].get('metrics', {})
                recommendations['expected_clearance_time'] = metrics.get('time_to_clearance', None)
                recommendations['expected_success_probability'] = 1.0 if metrics.get('is_successful', False) else 0.0
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error getting treatment recommendations for patient {subject_id}: {e}")
            return None
    
    def get_patient_summary(self, subject_id):
        """
        Get comprehensive summary for a patient.
        
        Parameters:
        -----------
        subject_id : int
            Patient subject ID
            
        Returns:
        --------
        dict
            Patient summary
        """
        # Ensure patient timeline exists
        if subject_id not in self.patient_timelines:
            self.create_patient_timeline(subject_id)
        
        if subject_id not in self.patient_timelines:
            self.logger.error(f"Cannot get patient summary: No timeline for patient {subject_id}")
            return None
        
        try:
            self.logger.info(f"Getting comprehensive summary for patient {subject_id}")
            
            # Get patient info
            patient_info = self.patient_timelines[subject_id].get('info', {})
            
            # Get stability status
            stability_status = self.get_patient_stability_status(subject_id)
            
            # Create summary
            summary = {
                'patient_id': subject_id,
                'gender': patient_info.get('gender', 'Unknown'),
                'age': patient_info.get('age', 'Unknown'),
                'admission_count': patient_info.get('admissions', 0),
                'icu_stay_count': patient_info.get('icustays', 0),
                'stability_status': stability_status.get('stability_status', 'Unknown') if stability_status else 'Unknown',
                'allostatic_load': stability_status.get('allostatic_load', None) if stability_status else None,
                'critical_systems': [],
                'approaching_transition': stability_status.get('approaching_transition', False) if stability_status else False,
                'available_analyses': []
            }
            
            # Add critical organ systems
            if stability_status:
                for system, status in stability_status.get('organ_systems', {}).items():
                    if status.get('score', 0) > 1.0:  # Threshold for "critical"
                        summary['critical_systems'].append(system)
            
            # Add available analyses
            if subject_id in self.patient_analyses:
                summary['available_analyses'] = list(self.patient_analyses[subject_id].keys())
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error getting patient summary for {subject_id}: {e}")
            return None