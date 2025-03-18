"""
BioDynamICS - Testing and Validation Module

This module provides comprehensive testing and validation functionality
for the BioDynamICS system, including test cases, validation against
clinical literature, and testing with different patient cohorts.

Author: Alexander Clarke
Date: March 18, 2025
"""

import os
import pandas as pd
import numpy as np
import json
import logging
import matplotlib.pyplot as plt
from datetime import datetime
import unittest
from pathlib import Path
import csv
import re
from typing import Dict, List, Union, Optional, Callable, Any, Tuple

# Import BioDynamICS modules
from src.system_integration import BioDynamicsSystem
from src.code_optimization import OptimizedBioDynamicsSystem

class BioDynamicsTester:
    """
    Provides comprehensive testing and validation functionality
    for the BioDynamICS system.
    """
    
    def __init__(self, system=None, config_path=None, results_dir=None):
        """
        Initialize the BioDynamicsTester.
        
        Parameters:
        -----------
        system : BioDynamicsSystem, optional
            BioDynamicsSystem instance to test
        config_path : str, optional
            Path to configuration file
        results_dir : str, optional
            Directory to save test results
        """
        # Set up logging
        self.logger = logging.getLogger("BioDynamicsTester")
        
        # Initialize system
        if system is None:
            self.system = OptimizedBioDynamicsSystem(config_path)
        else:
            self.system = system
        
        # Set results directory
        self.results_dir = results_dir or "test_results"
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Initialize test results
        self.test_results = {}
        
        # Load clinical reference data
        self.clinical_references = self._load_clinical_references()
        
        # Define patient cohorts
        self.cohorts = self._define_patient_cohorts()
        
        self.logger.info("Initialized BioDynamicsTester")
    
    def _load_clinical_references(self):
        """
        Load clinical reference data for validation.
        
        Returns:
        --------
        dict
            Clinical reference data
        """
        # Define path to clinical reference data
        reference_path = Path("data/clinical_references.json")
        
        # Check if file exists
        if reference_path.exists():
            try:
                with open(reference_path, 'r') as f:
                    references = json.load(f)
                self.logger.info(f"Loaded clinical references from {reference_path}")
                return references
            except Exception as e:
                self.logger.error(f"Error loading clinical references: {e}")
        
        # If file doesn't exist or error occurred, create default references
        self.logger.info("Creating default clinical references")
        
        # Default clinical references based on literature
        references = {
            "vital_signs": {
                "heart_rate": {
                    "normal_range": [60, 100],
                    "critical_low": 40,
                    "critical_high": 130,
                    "source": "American Heart Association guidelines"
                },
                "respiratory_rate": {
                    "normal_range": [12, 20],
                    "critical_low": 8,
                    "critical_high": 30,
                    "source": "American Thoracic Society guidelines"
                },
                "sbp": {
                    "normal_range": [90, 140],
                    "critical_low": 70,
                    "critical_high": 180,
                    "source": "American College of Cardiology guidelines"
                },
                "dbp": {
                    "normal_range": [60, 90],
                    "critical_low": 40,
                    "critical_high": 120,
                    "source": "American College of Cardiology guidelines"
                },
                "temperature": {
                    "normal_range": [36.5, 37.5],
                    "critical_low": 35.0,
                    "critical_high": 39.0,
                    "source": "World Health Organization guidelines"
                },
                "o2_saturation": {
                    "normal_range": [94, 100],
                    "critical_low": 88,
                    "critical_high": 100,
                    "source": "American Thoracic Society guidelines"
                }
            },
            "stability_metrics": {
                "allostatic_load": {
                    "normal_range": [0, 0.5],
                    "mild_stress": [0.5, 1.0],
                    "moderate_stress": [1.0, 1.5],
                    "severe_stress": [1.5, 3.0],
                    "source": "BioDynamICS research"
                }
            },
            "treatment_outcomes": {
                "antibiotics": {
                    "vancomycin": {
                        "s_aureus": {
                            "expected_clearance_time": [48, 96],
                            "success_rate": 0.85,
                            "source": "Infectious Diseases Society of America guidelines"
                        }
                    },
                    "ceftriaxone": {
                        "e_coli": {
                            "expected_clearance_time": [24, 72],
                            "success_rate": 0.9,
                            "source": "Infectious Diseases Society of America guidelines"
                        }
                    }
                }
            }
        }
        
        # Save references to file
        try:
            os.makedirs(os.path.dirname(reference_path), exist_ok=True)
            with open(reference_path, 'w') as f:
                json.dump(references, f, indent=4)
            self.logger.info(f"Saved default clinical references to {reference_path}")
        except Exception as e:
            self.logger.error(f"Error saving clinical references: {e}")
        
        return references
    
    def _define_patient_cohorts(self):
        """
        Define patient cohorts for testing.
        
        Returns:
        --------
        dict
            Patient cohorts
        """
        # Default cohorts
        cohorts = {
            "all_patients": [],
            "icu_patients": [],
            "cardiac_patients": [],
            "respiratory_patients": [],
            "sepsis_patients": [],
            "renal_patients": []
        }
        
        # Check if PATIENTS table is loaded
        if 'PATIENTS' not in self.system.data_integrator.tables:
            self.logger.warning("PATIENTS table not loaded, cannot define cohorts")
            return cohorts
        
        # Get all patient IDs
        all_patients = self.system.data_integrator.tables['PATIENTS']['subject_id'].unique()
        cohorts["all_patients"] = all_patients.tolist()
        
        # Check if ICUSTAYS table is loaded
        if 'ICUSTAYS' in self.system.data_integrator.tables:
            # Get ICU patients
            icu_patients = self.system.data_integrator.tables['ICUSTAYS']['subject_id'].unique()
            cohorts["icu_patients"] = icu_patients.tolist()
        
        # Log cohort sizes
        for cohort, patients in cohorts.items():
            self.logger.info(f"Defined cohort '{cohort}' with {len(patients)} patients")
        
        return cohorts
    
    # =========================================================================
    # Unit Tests
    # =========================================================================
    
    def run_unit_tests(self):
        """
        Run unit tests for the BioDynamICS system.
        
        Returns:
        --------
        dict
            Unit test results
        """
        self.logger.info("Running unit tests")
        
        # Define test suite
        suite = unittest.TestSuite()
        
        # Add test cases
        suite.addTest(unittest.makeSuite(BioDynamicsSystemTests))
        suite.addTest(unittest.makeSuite(DataIntegrationTests))
        suite.addTest(unittest.makeSuite(SignalProcessingTests))
        
        # Run tests
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        
        # Store results
        test_results = {
            'total': result.testsRun,
            'failures': len(result.failures),
            'errors': len(result.errors),
            'skipped': len(result.skipped),
            'success_rate': (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun if result.testsRun > 0 else 0,
            'failures_detail': [str(failure) for failure in result.failures],
            'errors_detail': [str(error) for error in result.errors]
        }
        
        self.test_results['unit_tests'] = test_results
        
        self.logger.info(f"Unit tests completed: {test_results['success_rate']*100:.1f}% success rate")
        
        return test_results
    
    # =========================================================================
    # Integration Tests
    # =========================================================================
    
    def run_integration_tests(self):
        """
        Run integration tests for the BioDynamICS system.
        
        Returns:
        --------
        dict
            Integration test results
        """
        self.logger.info("Running integration tests")
        
        # Initialize results
        results = {
            'total': 0,
            'passed': 0,
            'failed': 0,
            'tests': []
        }
        
        # Test 1: Data loading and patient timeline creation
        test_result = self._test_data_loading_and_timeline()
        results['tests'].append(test_result)
        results['total'] += 1
        if test_result['passed']:
            results['passed'] += 1
        else:
            results['failed'] += 1
        
        # Test 2: Signal processing and stability analysis
        test_result = self._test_signal_processing_and_stability()
        results['tests'].append(test_result)
        results['total'] += 1
        if test_result['passed']:
            results['passed'] += 1
        else:
            results['failed'] += 1
        
        # Calculate success rate
        results['success_rate'] = results['passed'] / results['total'] if results['total'] > 0 else 0
        
        # Store results
        self.test_results['integration_tests'] = results
        
        self.logger.info(f"Integration tests completed: {results['success_rate']*100:.1f}% success rate")
        
        return results
    
    def _test_data_loading_and_timeline(self):
        """
        Test data loading and patient timeline creation.
        
        Returns:
        --------
        dict
            Test result
        """
        test_name = "Data Loading and Timeline Creation"
        self.logger.info(f"Running integration test: {test_name}")
        
        try:
            # Load core tables
            self.system.load_mimic_data(['PATIENTS', 'ADMISSIONS', 'ICUSTAYS'])
            
            # Check if tables were loaded
            if not all(table in self.system.data_integrator.tables for table in ['PATIENTS', 'ADMISSIONS', 'ICUSTAYS']):
                return {
                    'name': test_name,
                    'passed': False,
                    'message': "Failed to load core tables"
                }
            
            # Get a patient ID
            if len(self.cohorts['all_patients']) == 0:
                return {
                    'name': test_name,
                    'passed': False,
                    'message': "No patients found in dataset"
                }
            
            subject_id = self.cohorts['all_patients'][0]
            
            # Create patient timeline
            timeline = self.system.create_patient_timeline(subject_id)
            
            # Check if timeline was created
            if timeline is None:
                return {
                    'name': test_name,
                    'passed': False,
                    'message': f"Failed to create timeline for patient {subject_id}"
                }
            
            return {
                'name': test_name,
                'passed': True,
                'message': f"Successfully loaded data and created timeline for patient {subject_id}"
            }
            
        except Exception as e:
            return {
                'name': test_name,
                'passed': False,
                'message': f"Error: {str(e)}"
            }
    
    def _test_signal_processing_and_stability(self):
        """
        Test signal processing and stability analysis.
        
        Returns:
        --------
        dict
            Test result
        """
        test_name = "Signal Processing and Stability Analysis"
        self.logger.info(f"Running integration test: {test_name}")
        
        try:
            # Get a patient ID
            if len(self.cohorts['all_patients']) == 0:
                return {
                    'name': test_name,
                    'passed': False,
                    'message': "No patients found in dataset"
                }
            
            subject_id = self.cohorts['all_patients'][0]
            
            # Ensure patient timeline exists
            if subject_id not in self.system.patient_timelines:
                timeline = self.system.create_patient_timeline(subject_id)
                if timeline is None:
                    return {
                        'name': test_name,
                        'passed': False,
                        'message': f"Failed to create timeline for patient {subject_id}"
                    }
            
            # Process signals
            signals = self.system.process_patient_signals(subject_id)
            
            # Check if signals were processed
            if signals is None:
                return {
                    'name': test_name,
                    'passed': False,
                    'message': f"Failed to process signals for patient {subject_id}"
                }
            
            # Analyze stability
            stability = self.system.analyze_physiological_stability(subject_id)
            
            # Check if stability was analyzed
            if stability is None or stability.empty:
                return {
                    'name': test_name,
                    'passed': False,
                    'message': f"Failed to analyze stability for patient {subject_id}"
                }
            
            return {
                'name': test_name,
                'passed': True,
                'message': f"Successfully processed signals and analyzed stability for patient {subject_id}"
            }
            
        except Exception as e:
            return {
                'name': test_name,
                'passed': False,
                'message': f"Error: {str(e)}"
            }
    
    # =========================================================================
    # Clinical Validation
    # =========================================================================
    
    def validate_against_clinical_literature(self):
        """
        Validate system results against clinical literature.
        
        Returns:
        --------
        dict
            Validation results
        """
        self.logger.info("Validating against clinical literature")
        
        # Initialize results
        results = {
            'total': 0,
            'passed': 0,
            'failed': 0,
            'validations': []
        }
        
        # Get patient IDs
        if len(self.cohorts['all_patients']) == 0:
            self.logger.warning("No patients found in dataset, skipping validation")
            return results
        
        # Use first patient for validation
        subject_id = self.cohorts['all_patients'][0]
        
        # Ensure patient has been analyzed
        self._ensure_patient_analyzed(subject_id)
        
        # Validate vital sign ranges
        validation = self._validate_vital_sign_ranges(subject_id)
        results['validations'].append(validation)
        results['total'] += 1
        if validation['passed']:
            results['passed'] += 1
        else:
            results['failed'] += 1
        
        # Calculate success rate
        results['success_rate'] = results['passed'] / results['total'] if results['total'] > 0 else 0
        
        # Store results
        self.test_results['clinical_validation'] = results
        
        self.logger.info(f"Clinical validation completed: {results['success_rate']*100:.1f}% success rate")
        
        return results
    
    def _ensure_patient_analyzed(self, subject_id):
        """
        Ensure a patient has been fully analyzed.
        
        Parameters:
        -----------
        subject_id : int
            Patient subject ID
        """
        # Create patient timeline if needed
        if subject_id not in self.system.patient_timelines:
            self.system.create_patient_timeline(subject_id)
        
        # Process signals if needed
        if (subject_id not in self.system.patient_analyses or 
            'signal_processing' not in self.system.patient_analyses[subject_id]):
            self.system.process_patient_signals(subject_id)
        
        # Analyze stability if needed
        if (subject_id not in self.system.patient_analyses or 
            'stability_report' not in self.system.patient_analyses[subject_id]):
            self.system.create_stability_report(subject_id)
    
    def _validate_vital_sign_ranges(self, subject_id):
        """
        Validate vital sign ranges against clinical literature.
        
        Parameters:
        -----------
        subject_id : int
            Patient subject ID
            
        Returns:
        --------
        dict
            Validation result
        """
        validation_name = "Vital Sign Ranges"
        self.logger.info(f"Running clinical validation: {validation_name}")
        
        try:
            # Get patient timeline
            timeline = self.system.patient_timelines[subject_id]
            
            # Get vital sign data
            timeline_df = timeline['timeline']
            
            # Get clinical references
            vital_references = self.clinical_references.get('vital_signs', {})
            
            # Initialize results
            validation_results = {
                'name': validation_name,
                'passed': True,
                'details': {}
            }
            
            # Validate each vital sign
            for vital, reference in vital_references.items():
                if vital in timeline_df.columns:
                    # Get vital sign values
                    values = timeline_df[vital].dropna()
                    
                    if len(values) == 0:
                        continue
                    
                    # Get statistics
                    mean = values.mean()
                    
                    # Get reference range
                    normal_range = reference.get('normal_range', [0, 0])
                    
                    # Check if mean is within normal range
                    mean_in_range = normal_range[0] <= mean <= normal_range[1]
                    
                    # Store results
                    validation_results['details'][vital] = {
                        'mean': mean,
                        'normal_range': normal_range,
                        'mean_in_range': mean_in_range
                    }
                    
                    # Update overall result
                    if not mean_in_range:
                        validation_results['passed'] = False
            
            # Add message
            if validation_results['passed']:
                validation_results['message'] = "All vital signs within expected ranges"
            else:
                out_of_range = [vital for vital, detail in validation_results['details'].items() 
                               if not detail['mean_in_range']]
                validation_results['message'] = f"Vital signs out of expected ranges: {', '.join(out_of_range)}"
            
            return validation_results
            
        except Exception as e:
            return {
                'name': validation_name,
                'passed': False,
                'message': f"Error: {str(e)}"
            }
    
    # =========================================================================
    # Cohort Testing
    # =========================================================================
    
    def test_patient_cohorts(self, max_patients_per_cohort=5):
        """
        Test the system with different patient cohorts.
        
        Parameters:
        -----------
        max_patients_per_cohort : int, optional
            Maximum number of patients to test per cohort
            
        Returns:
        --------
        dict
            Cohort testing results
        """
        self.logger.info("Testing patient cohorts")
        
        # Initialize results
        results = {
            'cohorts': {},
            'summary': {}
        }
        
        # Test each cohort
        for cohort_name, patient_ids in self.cohorts.items():
            if not patient_ids:
                continue
            
            self.logger.info(f"Testing cohort: {cohort_name}")
            
            # Limit number of patients
            test_patients = patient_ids[:max_patients_per_cohort]
            
            # Initialize cohort results
            cohort_results = {
                'patients': len(test_patients),
                'success': 0,
                'failure': 0,
                'patient_results': {}
            }
            
            # Test each patient
            for subject_id in test_patients:
                # Analyze patient
                try:
                    # Create timeline
                    timeline = self.system.create_patient_timeline(subject_id)
                    if timeline is None:
                        cohort_results['patient_results'][subject_id] = {
                            'success': False,
                            'error': "Failed to create timeline"
                        }
                        cohort_results['failure'] += 1
                        continue
                    
                    # Process signals
                    signals = self.system.process_patient_signals(subject_id)
                    if signals is None:
                        cohort_results['patient_results'][subject_id] = {
                            'success': False,
                            'error': "Failed to process signals"
                        }
                        cohort_results['failure'] += 1
                        continue
                    
                    # Success
                    cohort_results['patient_results'][subject_id] = {
                        'success': True,
                        'timeline_events': len(timeline['timeline']),
                        'features': len(signals.get('features', {}))
                    }
                    cohort_results['success'] += 1
                    
                except Exception as e:
                    cohort_results['patient_results'][subject_id] = {
                        'success': False,
                        'error': str(e)
                    }
                    cohort_results['failure'] += 1
            
            # Calculate success rate
            cohort_results['success_rate'] = cohort_results['success'] / len(test_patients) if test_patients else 0
            
            # Store cohort results
            results['cohorts'][cohort_name] = cohort_results
        
        # Calculate overall summary
        total_patients = sum(cohort['patients'] for cohort in results['cohorts'].values())
        total_success = sum(cohort['success'] for cohort in results['cohorts'].values())
        
        results['summary'] = {
            'total_patients': total_patients,
            'total_success': total_success,
            'overall_success_rate': total_success / total_patients if total_patients > 0 else 0
        }
        
        # Store results
        self.test_results['cohort_testing'] = results
        
        self.logger.info(f"Cohort testing completed: {results['summary']['overall_success_rate']*100:.1f}% success rate")
        
        return results
    
    # =========================================================================
    # Test Report Generation
    # =========================================================================
    
    def generate_test_report(self, report_path=None):
        """
        Generate a comprehensive test report.
        
        Parameters:
        -----------
        report_path : str, optional
            Path to save the report
            
        Returns:
        --------
        dict
            Test report
        """
        self.logger.info("Generating test report")
        
        # Create report
        report = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'system_info': {
                'version': '1.0.0',
                'config': self.system.config
            },
            'test_results': self.test_results,
            'summary': {
                'unit_tests': self.test_results.get('unit_tests', {}).get('success_rate', 0),
                'integration_tests': self.test_results.get('integration_tests', {}).get('success_rate', 0),
                'clinical_validation': self.test_results.get('clinical_validation', {}).get('success_rate', 0),
                'cohort_testing': self.test_results.get('cohort_testing', {}).get('summary', {}).get('overall_success_rate', 0)
            }
        }
        
        # Calculate overall score
        scores = [
            report['summary']['unit_tests'],
            report['summary']['integration_tests'],
            report['summary']['clinical_validation'],
            report['summary']['cohort_testing']
        ]
        
        report['summary']['overall_score'] = sum(scores) / len(scores) if scores else 0
        
        # Save report if path provided
        if report_path:
            try:
                # Create directory if it doesn't exist
                os.makedirs(os.path.dirname(report_path), exist_ok=True)
                
                # Save as JSON
                with open(report_path, 'w') as f:
                    json.dump(report, f, indent=4, default=str)
                
                self.logger.info(f"Test report saved to {report_path}")
                
            except Exception as e:
                self.logger.error(f"Error saving test report: {e}")
        
        return report
    
    def run_all_tests(self, report_path=None):
        """
        Run all tests and generate a comprehensive report.
        
        Parameters:
        -----------
        report_path : str, optional
            Path to save the report
            
        Returns:
        --------
        dict
            Test report
        """
        self.logger.info("Running all tests")
        
        # Run unit tests
        self.run_unit_tests()
        
        # Run integration tests
        self.run_integration_tests()
        
        # Validate against clinical literature
        self.validate_against_clinical_literature()
        
        # Test patient cohorts
        self.test_patient_cohorts()
        
        # Generate report
        report = self.generate_test_report(report_path)
        
        self.logger.info(f"All tests completed with overall score: {report['summary']['overall_score']*100:.1f}%")
        
        return report

# Unit test classes for BioDynamicsSystem
class BioDynamicsSystemTests(unittest.TestCase):
    """Unit tests for BioDynamicsSystem."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.system = BioDynamicsSystem()
    
    def test_initialization(self):
        """Test initialization."""
        self.assertIsNotNone(self.system)
        self.assertIsNotNone(self.system.config)
    
    def test_configuration(self):
        """Test configuration."""
        self.assertIn('data_path', self.system.config)
        self.assertIn('results_path', self.system.config)

class DataIntegrationTests(unittest.TestCase):
    """Unit tests for data integration."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.system = BioDynamicsSystem()
    
    def test_data_integrator_initialization(self):
        """Test data integrator initialization."""
        self.assertIsNotNone(self.system.data_integrator)

class SignalProcessingTests(unittest.TestCase):
    """Unit tests for signal processing."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.system = BioDynamicsSystem()
    
    def test_signal_processor_initialization(self):
        """Test signal processor initialization."""
        self.assertIsNotNone(self.system.signal_processor)
    
    def test_reference_ranges(self):
        """Test reference ranges."""
        self.assertIn('heart_rate', self.system.signal_processor.reference_ranges)
        self.assertEqual(len(self.system.signal_processor.reference_ranges['heart_rate']), 2)

# Example usage
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create tester
    tester = BioDynamicsTester()
    
    # Run all tests
    report = tester.run_all_tests("test_results/test_report.json")
    
    print(f"Overall test score: {report['summary']['overall_score']*100:.1f}%")
