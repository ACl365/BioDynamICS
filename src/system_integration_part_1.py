"""
BioDynamICS - System Integration Module

This module integrates all BioDynamICS components into a unified framework for comprehensive
physiological data analysis, dynamical systems modeling, and treatment optimization.

Author: Alexander Clarke
Date: March 16, 2025
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import warnings

# Import all component modules
from src.data_integration import MimicPatientIntegrator
from src.signal_processing import PhysiologicalSignalProcessor
from src.dynamical_modeling import DynamicalSystemsModeler
from src.infection_treatment import InfectionTreatmentModeler
from src.visualization import ClinicalVisualizer

class BioDynamICSSystem:
    """
    Unified system that integrates all BioDynamICS components for comprehensive
    physiological analysis, dynamical systems modeling, and treatment optimization.
    
    This class serves as the main interface for the BioDynamICS framework, orchestrating
    the interactions between different modules and providing streamlined workflows
    for common analysis tasks.
    """
    
    def __init__(self, data_path=None):
        """
        Initialize the BioDynamICS system with all component modules.
        
        Parameters:
        -----------
        data_path : str, optional
            Path to MIMIC-III data (default: None)
        """
        # Initialize all component modules
        self.data_integrator = MimicPatientIntegrator(data_path) if data_path else None
        self.signal_processor = PhysiologicalSignalProcessor()
        self.dynamical_modeler = DynamicalSystemsModeler()
        self.infection_modeler = InfectionTreatmentModeler()
        self.visualizer = ClinicalVisualizer()
        
        # Store the data path
        self.data_path = data_path
        
        # Initialize cache for analysis results
        self.analysis_cache = {}
        
        # Log initialization
        self._log_operation("Initialized BioDynamICS System")
        if data_path:
            self._log_operation(f"Connected to data at: {data_path}")
        else:
            self._log_operation("No data path provided - some features will be limited")
    
    def _log_operation(self, message):
        """Log operations with timestamp."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] BioDynamICS: {message}")
    
    # =========================================================================
    # Data Integration Methods
    # =========================================================================
    
    def load_patient_data(self, subject_id=None):
        """
        Load patient data from MIMIC-III database.
        
        Parameters:
        -----------
        subject_id : int, optional
            Patient subject ID (if None, tries to find a suitable patient)
            
        Returns:
        --------
        dict
            Patient timeline dictionary with 'info' and 'timeline' keys
        """
        if self.data_integrator is None:
            self._log_operation("Error: Data integrator not initialized - no data path provided")
            return None
        
        # Load core tables if they haven't been loaded yet
        if 'PATIENTS' not in self.data_integrator.tables:
            self._log_operation("Loading core MIMIC-III tables...")
            self.data_integrator.load_core_tables()
        
        # Load additional tables for comprehensive analysis
        needed_tables = ['CHARTEVENTS', 'LABEVENTS', 'PRESCRIPTIONS', 'MICROBIOLOGYEVENTS']
        for table in needed_tables:
            if table not in self.data_integrator.tables:
                self._log_operation(f"Loading {table}...")
                if table == 'CHARTEVENTS':
                    self.data_integrator.load_chartevents_chunked()
                else:
                    self.data_integrator.load_table(table)
        
        # If no subject_id provided, find a suitable patient with good data
        if subject_id is None:
            subject_id = self._find_patient_with_good_data()
            if subject_id is None:
                self._log_operation("Error: Could not find a suitable patient with sufficient data")
                return None
        
        # Load patient timeline
        self._log_operation(f"Loading timeline for patient {subject_id}...")
        patient_timeline = self.data_integrator.create_patient_timeline(subject_id)
        
        if patient_timeline is None or 'timeline' not in patient_timeline:
            self._log_operation(f"Error: Failed to create timeline for patient {subject_id}")
            return None
        
        if len(patient_timeline['timeline']) == 0:
            self._log_operation(f"Warning: Empty timeline for patient {subject_id}")
        else:
            self._log_operation(f"Successfully loaded timeline with {len(patient_timeline['timeline'])} events")
        
        return patient_timeline
    
    def _find_patient_with_good_data(self, min_events=100):
        """
        Find a patient with sufficient data for comprehensive analysis.
        
        Parameters:
        -----------
        min_events : int, optional
            Minimum number of events required (default: 100)
            
        Returns:
        --------
        int or None
            Subject ID of suitable patient, or None if not found
        """
        if 'PATIENTS' not in self.data_integrator.tables:
            self._log_operation("Error: PATIENTS table not loaded")
            return None
        
        self._log_operation("Searching for a patient with sufficient data...")
        
        patient_data = {}
        patients = self.data_integrator.tables['PATIENTS']['subject_id'].unique()
        
        # Check first 10 patients
        for i, patient_id in enumerate(patients[:10]):
            self._log_operation(f"Checking patient {i+1}/10: {patient_id}")
            timeline = self.data_integrator.create_patient_timeline(patient_id)
            
            if 'timeline' in timeline and len(timeline['timeline']) > 0:
                event_count = len(timeline['timeline'])
                self._log_operation(f"  Found {event_count} events")
                
                patient_data[patient_id] = {
                    'events': event_count,
                    'timeline': timeline
                }
                
                if event_count >= min_events:
                    self._log_operation(f"  Sufficient data found!")
                    return patient_id
        
        # If no patient meets the threshold, return the one with the most events
        if patient_data:
            best_patient_id = max(patient_data, key=lambda p: patient_data[p]['events'])
            self._log_operation(f"Using patient {best_patient_id} with {patient_data[best_patient_id]['events']} events")
            return best_patient_id
        
        return None
    
    # =========================================================================
    # Signal Processing Methods
    # =========================================================================
    
    def process_physiological_signals(self, patient_timeline, reuse_cache=True):
        """
        Process physiological signals from patient timeline.
        
        Parameters:
        -----------
        patient_timeline : dict
            Patient timeline dictionary with 'info' and 'timeline' keys
        reuse_cache : bool, optional
            Whether to reuse cached results (default: True)
            
        Returns:
        --------
        dict
            Signal processing results with features and stability metrics
        """
        if patient_timeline is None or 'timeline' not in patient_timeline:
            self._log_operation("Error: Invalid patient timeline")
            return None
        
        # Check if we already have cached results for this patient
        subject_id = patient_timeline.get('info', {}).get('subject_id')
        cache_key = f"signal_processing_{subject_id}"
        
        if reuse_cache and cache_key in self.analysis_cache:
            self._log_operation(f"Using cached signal processing results for patient {subject_id}")
            return self.analysis_cache[cache_key]
        
        # Process the patient timeline
        self._log_operation(f"Processing physiological signals for patient {subject_id}...")
        processing_results = self.signal_processor.process_patient_timeline(patient_timeline)
        
        # Create stability report
        self._log_operation("Generating stability report...")
        stability_report = self.signal_processor.create_stability_report(patient_timeline)
        
        # Combine results
        results = {
            'processing_results': processing_results,
            'stability_report': stability_report
        }
        
        # Cache the results
        self.analysis_cache[cache_key] = results
        
        self._log_operation("Physiological signal processing complete")
        return results
    
    def analyze_vital_sign(self, patient_timeline, vital_sign='heart_rate'):
        """
        Extract and analyze a specific vital sign from patient timeline.
        
        Parameters:
        -----------
        patient_timeline : dict
            Patient timeline dictionary with 'info' and 'timeline' keys
        vital_sign : str, optional
            Name of the vital sign to analyze (default: 'heart_rate')
            
        Returns:
        --------
        tuple
            (time_series, analysis_results)
        """
        if patient_timeline is None or 'timeline' not in patient_timeline:
            self._log_operation("Error: Invalid patient timeline")
            return None, None
        
        timeline_df = patient_timeline['timeline']
        
        # Extract the vital sign time series
        self._log_operation(f"Extracting {vital_sign} time series...")
        
        time_series = None
        
        # Check if the vital sign is directly available as a column
        if vital_sign in timeline_df.columns:
            vital_data = timeline_df[[vital_sign, 'measurement_time']].dropna(subset=[vital_sign])
            vital_data = vital_data.sort_values('measurement_time')
            
            time_series = pd.Series(
                vital_data[vital_sign].values,
                index=pd.to_datetime(vital_data['measurement_time'])
            )
        
        # Otherwise, try to extract from MIMIC itemid mapping
        elif 'itemid' in timeline_df.columns and 'valuenum' in timeline_df.columns:
            # Use the mapping from the signal processor
            vital_itemids = []
            
            # Default mappings for common vital signs
            if vital_sign == 'heart_rate':
                vital_itemids = [211, 220045]
            elif vital_sign == 'respiratory_rate':
                vital_itemids = [618, 220210]
            elif vital_sign == 'sbp':  # Systolic blood pressure
                vital_itemids = [51, 220050]
            elif vital_sign == 'dbp':  # Diastolic blood pressure
                vital_itemids = [8368, 220051]
            elif vital_sign == 'temperature':
                vital_itemids = [223761, 678]
            
            # Extract values for these itemids
            if vital_itemids:
                vital_data = timeline_df[timeline_df['itemid'].isin(vital_itemids)].copy()
                
                if len(vital_data) > 0:
                    # Convert to time series
                    vital_data['measurement_time'] = pd.to_datetime(vital_data['charttime'])
                    vital_data = vital_data.sort_values('measurement_time')
                    
                    time_series = pd.Series(
                        vital_data['valuenum'].values,
                        index=vital_data['measurement_time']
                    )
        
        if time_series is None or len(time_series) < 5:
            self._log_operation(f"Error: Insufficient {vital_sign} data")
            return None, None
        
        self._log_operation(f"Extracted {vital_sign} time series with {len(time_series)} measurements")
        
        # Analyze the vital sign using the signal processor
        self._log_operation(f"Analyzing {vital_sign} time series...")
        
        # Preprocess the time series
        preprocessed_series = self.signal_processor.preprocess_timeseries(time_series, vital_sign)
        
        # Extract features
        features = self.signal_processor.extract_all_features_from_series(
            preprocessed_series, measurement_name=vital_sign
        )
        
        # Calculate stability metrics
        stability_metrics = self.signal_processor.calculate_stability_metrics(
            preprocessed_series, measurement_name=vital_sign
        )
        
        # Combine results
        analysis_results = {
            'features': features,
            'stability_metrics': stability_metrics
        }
        
        return time_series, analysis_results
    
    # =========================================================================
    # Dynamical Systems Analysis Methods
    # =========================================================================
    
    def perform_dynamical_analysis(self, time_series, vital_sign='heart_rate', reuse_cache=True):
        """
        Perform dynamical systems analysis on a time series.
        
        Parameters:
        -----------
        time_series : pandas.Series
            Time series data with datetime index
        vital_sign : str, optional
            Name of the vital sign (default: 'heart_rate')
        reuse_cache : bool, optional
            Whether to reuse cached results (default: True)
            
        Returns:
        --------
        dict
            Dynamical systems analysis results
        """
        if time_series is None or len(time_series) < 10:
            self._log_operation("Error: Insufficient data for dynamical systems analysis")
            return None
        
        # Generate a cache key based on the time series
        cache_key = f"dynamical_analysis_{vital_sign}_{hash(str(time_series.values))}"
        
        if reuse_cache and cache_key in self.analysis_cache:
            self._log_operation(f"Using cached dynamical systems analysis for {vital_sign}")
            return self.analysis_cache[cache_key]
        
        self._log_operation(f"Performing dynamical systems analysis on {vital_sign}...")
        
        # Find optimal embedding parameters
        self._log_operation("Estimating optimal embedding parameters...")
        optimal_delay = self.dynamical_modeler.estimate_optimal_time_delay(
            time_series.values, method='autocorr'
        )
        
        # Use a try-except block for embedding dimension estimation as it might fail
        try:
            optimal_dim, embed_info = self.dynamical_modeler.estimate_embedding_dimension(
                time_series.values, time_delay=optimal_delay
            )
        except Exception as e:
            self._log_operation(f"Warning: Embedding dimension estimation failed: {e}")
            optimal_dim = 3  # Use a default value
            embed_info = {'error': str(e)}
        
        self._log_operation(f"Optimal time delay: {optimal_delay}, dimension: {optimal_dim}")
        
        # Create state space embedding
        embedded_2d = self.dynamical_modeler.time_delay_embedding(
            time_series.values, embedding_dimension=2, time_delay=optimal_delay
        )
        
        embedded_3d = None
        if len(time_series) >= 3 * optimal_delay:
            embedded_3d = self.dynamical_modeler.time_delay_embedding(
                time_series.values, embedding_dimension=3, time_delay=optimal_delay
            )
        
        # Calculate stability metrics
        self._log_operation("Calculating dynamical stability metrics...")
        
        # Use try-except blocks for methods that might fail with limited data
        try:
            lyapunov, lyap_info = self.dynamical_modeler.calculate_lyapunov_exponent(
                time_series.values, embedding_dimension=optimal_dim, time_delay=optimal_delay
            )
        except Exception as e:
            self._log_operation(f"Warning: Lyapunov exponent calculation failed: {e}")
            lyapunov = np.nan
            lyap_info = {'error': str(e)}
        
        try:
            recurrence_matrix, recur_info = self.dynamical_modeler.calculate_recurrence_plot(
                time_series.values, embedding_dimension=optimal_dim, time_delay=optimal_delay
            )
            
            if len(recurrence_matrix) > 0:
                recurrence_metrics = self.dynamical_modeler.calculate_recurrence_quantification(
                    recurrence_matrix
                )
            else:
                recurrence_metrics = {'error': 'Recurrence matrix calculation failed'}
        except Exception as e:
            self._log_operation(f"Warning: Recurrence analysis failed: {e}")
            recurrence_matrix = np.array([])
            recur_info = {'error': str(e)}
            recurrence_metrics = {'error': str(e)}
        
        # Detect fixed points and limit cycles
        try:
            attractor_info = self.dynamical_modeler.detect_fixed_points(
                time_series.values, embedding_dimension=optimal_dim, time_delay=optimal_delay
            )
        except Exception as e:
            self._log_operation(f"Warning: Fixed point detection failed: {e}")
            attractor_info = {'fixed_points': [], 'limit_cycles': [], 'error': str(e)}
        
        # Combine all results
        results = {
            'embedding': {
                'optimal_time_delay': optimal_delay,
                'optimal_dimension': optimal_dim,
                'embedding_info': embed_info,
                'embedded_2d': embedded_2d,
                'embedded_3d': embedded_3d
            },
            'stability': {
                'lyapunov_exponent': lyapunov,
                'lyapunov_info': lyap_info,
                'recurrence_matrix': recurrence_matrix,
                'recurrence_info': recur_info,
                'recurrence_metrics': recurrence_metrics,
                'attractor_info': attractor_info
            }
        }
        
        # Interpret the results
        results['interpretation'] = self._interpret_dynamical_analysis(results)
        
        # Cache the results
        self.analysis_cache[cache_key] = results
        
        self._log_operation("Dynamical systems analysis complete")
        return results
    
    def _interpret_dynamical_analysis(self, analysis_results):
        """
        Interpret dynamical systems analysis results.
        
        Parameters:
        -----------
        analysis_results : dict
            Dynamical systems analysis results
            
        Returns:
        --------
        dict
            Interpretation of results in clinical terms
        """
        interpretation = {}
        
        # Interpret Lyapunov exponent
        lyapunov = analysis_results['stability']['lyapunov_exponent']
        if np.isnan(lyapunov):
            interpretation['stability'] = "Undetermined (insufficient data)"
        elif lyapunov < -0.05:
            interpretation['stability'] = "Highly stable (convergent dynamics)"
        elif lyapunov < 0:
            interpretation['stability'] = "Stable (mild convergence)"
        elif lyapunov < 0.05:
            interpretation['stability'] = "Marginally stable (potential limit cycle)"
        else:
            interpretation['stability'] = "Unstable (chaotic dynamics)"
        
        # Interpret recurrence metrics
        recurrence_metrics = analysis_results['stability']['recurrence_metrics']
        if 'error' not in recurrence_metrics:
            if recurrence_metrics.get('determinism', 0) > 0.7:
                interpretation['determinism'] = "High determinism (predictable patterns)"
            elif recurrence_metrics.get('determinism', 0) > 0.3:
                interpretation['determinism'] = "Moderate determinism (some predictability)"
            else:
                interpretation['determinism'] = "Low determinism (unpredictable dynamics)"
            
            if recurrence_metrics.get('laminarity', 0) > 0.7:
                interpretation['laminarity'] = "High laminarity (persistent states)"
            else:
                interpretation['laminarity'] = "Low laminarity (frequent state transitions)"
        
        # Interpret attractors
        attractor_info = analysis_results['stability']['attractor_info']
        if len(attractor_info.get('fixed_points', [])) > 0:
            interpretation['attractors'] = "Fixed points detected (stable equilibrium states)"
        elif len(attractor_info.get('limit_cycles', [])) > 0:
            interpretation['attractors'] = "Limit cycles detected (oscillatory behavior)"
        else:
            interpretation['attractors'] = "No clear attractors identified"
        
        # Overall assessment
        if interpretation['stability'].startswith("Stable") or interpretation['stability'].startswith("Highly stable"):
            interpretation['overall'] = "The system exhibits stable dynamics with good regulatory control"
        elif interpretation['stability'].startswith("Marginally stable"):
            interpretation['overall'] = "The system shows oscillatory behavior with adequate regulatory control"
        elif interpretation['stability'].startswith("Unstable"):
            interpretation['overall'] = "The system exhibits complex dynamics with potential regulatory issues"
        else:
            interpretation['overall'] = "Insufficient data for reliable assessment"
        
        return interpretation
    
    def detect_critical_transitions(self, time_series, vital_sign='heart_rate'):
        """
        Detect potential critical transitions in a time series.
        
        Parameters:
        -----------
        time_series : pandas.Series
            Time series data with datetime index
        vital_sign : str, optional
            Name of the vital sign (default: 'heart_rate')
            
        Returns:
        --------
        dict
            Critical transition detection results
        """
        if time_series is None or len(time_series) < 50:
            self._log_operation("Error: Insufficient data for critical transition detection")
            return {'detected': False, 'error': 'Insufficient data'}
        
        self._log_operation(f"Detecting critical transitions in {vital_sign}...")
        
        # Calculate early warning signals
        try:
            ews = self.dynamical_modeler.calculate_early_warning_signals(time_series.values)
            
            # Detect critical transitions
            transition = self.dynamical_modeler.detect_critical_transition(time_series.values)
            
            if transition['detected']:
                self._log_operation(f"Critical transition detected with probability {transition['probability']:.2f}")
            else:
                self._log_operation("No critical transitions detected")
            
            return {
                'early_warning_signals': ews,
                'transition_detection': transition
            }
        except Exception as e:
            self._log_operation(f"Warning: Critical transition detection failed: {e}")
            return {'detected': False, 'error': str(e)}
    
    # =========================================================================
    # Infection Treatment Methods
    # =========================================================================
    
    def analyze_infection_treatment(self, pathogen, antibiotic, treatments=None):
        """
        Analyze infection treatment dynamics for a specific pathogen-antibiotic combination.
        
        Parameters:
        -----------
        pathogen : str
            Type of pathogen (e.g., 's_aureus', 'e_coli')
        antibiotic : str
            Type of antibiotic (e.g., 'vancomycin', 'ceftriaxone')
        treatments : list of dict, optional
            List of treatment regimens to compare, each with 'dose' and 'interval' keys
            
        Returns:
        --------
        dict
            Infection treatment analysis results
        """
        self._log_operation(f"Analyzing {antibiotic} treatment for {pathogen}...")
        
        # If no treatments specified, use default options
        if treatments is None:
            treatments = [
                {'dose': 1000, 'interval': 12},  # Standard q12h
                {'dose': 1000, 'interval': 8},   # Standard q8h
                {'dose': 1500, 'interval': 12},  # Higher dose q12h
                {'dose': 750, 'interval': 6}     # Lower dose more frequently
            ]
        
        # Compare treatment regimens
        comparison = self.infection_modeler.evaluate_multiple_regimens(
            antibiotic=antibiotic,
            pathogen=pathogen,
            regimens=treatments,
            duration_hours=120
        )
        
        # Optimize treatment
        optimization = self.infection_modeler.optimize_dosing_regimen(
            antibiotic=antibiotic,
            pathogen=pathogen,
            dose_range=(500, 2000),
            interval_range=(6, 24),
            duration_hours=120,
            objective='clearance'
        )
        
        # Combine results
        results = {
            'pathogen': pathogen,
            'antibiotic': antibiotic,
            'treatment_comparison': comparison,
            'treatment_optimization': optimization
        }
        
        # Add interpretation
        results['interpretation'] = self._interpret_treatment_analysis(results)
        
        self._log_operation("Infection treatment analysis complete")
        return results
    
    def _interpret_treatment_analysis(self, treatment_results):
        """
        Interpret infection treatment analysis results.
        
        Parameters:
        -----------
        treatment_results : dict
            Infection treatment analysis results
            
        Returns:
        --------
        dict
            Interpretation of results in clinical terms
        """
        interpretation = {}
        
        pathogen = treatment_results['pathogen']
        antibiotic = treatment_results['antibiotic']
        
        # Get best regimen for clearance
        comparison = treatment_results['treatment_comparison']
        best_idx = comparison['best_regimens']['overall']
        best_regimen = comparison['all_regimens'][best_idx]
        
        # Optimal regimen from optimization
        optimization = treatment_results['treatment_optimization']
        
        # Basic effectiveness assessment
        if best_regimen['is_successful']:
            interpretation['effectiveness'] = f"{antibiotic.title()} is effective against {pathogen.replace('_', ' ').title()}"
        else:
            interpretation['effectiveness'] = (f"{antibiotic.title()} may not be fully effective against "
                                              f"{pathogen.replace('_', ' ').title()} without optimization")
        
        # PK/PD parameters importance
        if best_regimen['time_above_mic'] > 80:
            interpretation['key_parameter'] = "Time above MIC is the key driver of efficacy"
        elif best_regimen['peak_mic_ratio'] > 10:
            interpretation['key_parameter'] = "Peak/MIC ratio is the key driver of efficacy"
        elif best_regimen['auc_mic_ratio'] > 400:
            interpretation['key_parameter'] = "AUC/MIC ratio is the key driver of efficacy"
        else:
            interpretation['key_parameter'] = "Multiple PK/PD parameters are important for efficacy"
        
        # Dosing recommendation
        optimal_dose = optimization['optimal_dose']
        optimal_interval = optimization['optimal_interval']
        
        interpretation['recommendation'] = (f"Optimal regimen: {optimal_dose:.0f}mg every "
                                           f"{optimal_interval:.1f} hours")
        
        # Expected outcomes
        if optimization['treatment_results']['metrics']['is_successful']:
            time_to_clear = optimization['treatment_results']['metrics']['time_to_clearance']
            interpretation['expected_outcome'] = f"Expected clearance in {time_to_clear:.1f} hours with optimal regimen"
        else:
            interpretation['expected_outcome'] = "Even with optimization, complete clearance may be challenging"
        
        return interpretation
    
    def analyze_microbiology_data(self):
        """
        Analyze microbiology data from the MIMIC-III database.
        
        Returns:
        --------
        dict
            Analysis of microbiology data
        """
        if self.data_integrator is None:
            self._log_operation("Error: Data integrator not initialized")
            return None
        
        # Load microbiology data if not already loaded
        if 'MICROBIOLOGYEVENTS' not in self.data_integrator.tables:
            self._log_operation("Loading microbiology data...")
            self.data_integrator.load_table("MICROBIOLOGYEVENTS")
        
        # Load prescriptions data if not already loaded
        if 'PRESCRIPTIONS' not in self.data_integrator.tables:
            self._log_operation("Loading prescriptions data...")
            self.data_integrator.load_table("PRESCRIPTIONS")
        
        micro = self.data_integrator.tables.get('MICROBIOLOGYEVENTS')
        prescriptions = self.data_integrator.tables.get('PRESCRIPTIONS')
        
        if micro is None or prescriptions is None:
            self._log_operation("Error: Could not load required tables")
            return None
        
        self._log_operation("Analyzing microbiology and prescription data...")
        
        results = {
            'organism_counts': {},
            'antibiotic_counts': {},
            'organism_sensitivity': {},
            'patient_infections': {}
        }
        
        # Count organisms
        if 'ORGANISM' in micro.columns:
            results['organism_counts'] = micro['ORGANISM'].value_counts().to_dict()
        
        # Count antibiotics
        if 'DRUG' in prescriptions.columns:
            # Filter for common antibiotics
            common_antibiotics = [
                'VANCOMYCIN', 'CEFTRIAXONE', 'CIPROFLOXACIN', 'PIPERACILLIN',
                'MEROPENEM', 'LEVOFLOXACIN', 'AZITHROMYCIN'
            ]
            
            antibiotic_mask = prescriptions['DRUG'].str.contains('|'.join(common_antibiotics), 
                                                                case=False, 
                                                                na=False)
            antibiotic_prescriptions = prescriptions[antibiotic_mask]
            
            results['antibiotic_counts'] = antibiotic_prescriptions['DRUG'].value_counts().to_dict()
        
        # Analyze sensitivity patterns if available
        if 'ORGANISM' in micro.columns and 'INTERPRETATION' in micro.columns:
            # Group by organism and antibiotic
            sensitivity_data = {}
            
            for organism in micro['ORGANISM'].dropna().unique():
                org_data = micro[micro['ORGANISM'] == organism]
                
                if 'AB_NAME' in org_data.columns and 'INTERPRETATION' in org_data.columns:
                    sensitivities = {}
                    
                    for ab in org_data['AB_NAME'].dropna().unique():
                        ab_data = org_data[org_data['AB_NAME'] == ab]
                        if len(ab_data) > 0:
                            interp_counts = ab_data['INTERPRETATION'].value_counts()
                            sensitivities[ab] = {
                                'S': interp_counts.get('S', 0),  # Sensitive
                                'I': interp_counts.get('I', 0),  # Intermediate
                                'R': interp_counts.get('R', 0)   # Resistant
                            }
                    
                    sensitivity_data[organism] = sensitivities
            
            results['organism_sensitivity'] = sensitivity_data
        
        # Analyze patient infection patterns
        if 'subject_id' in micro.columns and 'ORGANISM' in micro.columns:
            patient_infections = {}
            
            for subject_id in micro['subject_id'].unique():
                patient_data = micro[micro['subject_id'] == subject_id]
                
                if len(patient_data) > 0:
                    organisms = patient_data['ORGANISM'].value_counts().to_dict()
                    
                    # Check for relevant prescriptions