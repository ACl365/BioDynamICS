"""
Unit tests for the BioDynamICS System Integration Module.

This file contains tests for the BioDynamicsSystem class and its methods.
It verifies the functionality of the system integration module and provides
examples of how to use the module programmatically.

Author: Alexander Clarke
Date: March 18, 2025
"""

import unittest
import os
import json
import tempfile
import shutil
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Import the BioDynamicsSystem
from src.system_integration import BioDynamicsSystem

class TestBioDynamicsSystem(unittest.TestCase):
    """Test cases for the BioDynamicsSystem class."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures that are used for all tests."""
        # Create a temporary directory for test files
        cls.temp_dir = tempfile.mkdtemp()
        
        # Create a test configuration
        cls.config = {
            "data_path": "data/mimic-iii-clinical-database-demo-1.4",
            "results_path": cls.temp_dir,
            "cache_enabled": True,
            "cache_path": os.path.join(cls.temp_dir, "cache"),
            "parallel_processing": False,  # Disable for testing
            "max_workers": 1,
            "analysis_parameters": {
                "window_hours": 12,  # Smaller for faster tests
                "step_hours": 6,
                "embedding_dimension": 2,
                "stability_threshold": 0.5
            },
            "visualization_settings": {
                "save_figures": True,
                "figure_format": "png",
                "figure_dpi": 100  # Lower for faster tests
            }
        }
        
        # Save configuration to file
        cls.config_path = os.path.join(cls.temp_dir, "test_config.json")
        with open(cls.config_path, 'w') as f:
            json.dump(cls.config, f, indent=4)
        
        # Initialize the system
        cls.system = BioDynamicsSystem(config_path=cls.config_path)
        
        # Create mock data for testing
        cls._create_mock_data()
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test fixtures after all tests."""
        # Remove temporary directory
        shutil.rmtree(cls.temp_dir)
    
    @classmethod
    def _create_mock_data(cls):
        """Create mock data for testing."""
        # Create mock patient data
        cls.mock_patient_id = 12345
        
        # Create mock timeline
        timeline_data = {
            'measurement_time': pd.date_range(start='2025-01-01', periods=100, freq='H'),
            'heart_rate': np.random.normal(80, 10, 100),
            'sbp': np.random.normal(120, 15, 100),
            'dbp': np.random.normal(80, 10, 100),
            'respiratory_rate': np.random.normal(16, 3, 100),
            'temperature': np.random.normal(37, 0.5, 100),
            'o2_saturation': np.random.normal(98, 2, 100),
            'event_type': ['chart'] * 100
        }
        
        # Add some trends and patterns
        # Heart rate increasing trend
        timeline_data['heart_rate'] += np.linspace(0, 20, 100)
        
        # Blood pressure with a dip in the middle
        timeline_data['sbp'] -= 20 * np.sin(np.linspace(0, np.pi, 100))
        timeline_data['dbp'] -= 15 * np.sin(np.linspace(0, np.pi, 100))
        
        # Create DataFrame
        timeline_df = pd.DataFrame(timeline_data)
        
        # Create patient info
        patient_info = {
            'subject_id': cls.mock_patient_id,
            'gender': 'M',
            'dob': '1980-01-01',
            'dod': None,
            'expire_flag': 0,
            'admissions': 1,
            'icustays': 1
        }
        
        # Create patient timeline
        cls.mock_timeline = {
            'info': patient_info,
            'timeline': timeline_df
        }
        
        # Add to system
        cls.system.patient_timelines[cls.mock_patient_id] = cls.mock_timeline
    
    def test_initialization(self):
        """Test initialization and configuration."""
        # Test initialization with config file
        system = BioDynamicsSystem(config_path=self.config_path)
        self.assertIsNotNone(system)
        self.assertEqual(system.config['data_path'], self.config['data_path'])
        
        # Test initialization without config file
        system = BioDynamicsSystem()
        self.assertIsNotNone(system)
        self.assertIn('data_path', system.config)
        
        # Test updating configuration
        system.update_configuration({'max_workers': 8})
        self.assertEqual(system.config['max_workers'], 8)
        
        # Test saving configuration
        save_path = os.path.join(self.temp_dir, "saved_config.json")
        system.save_configuration(save_path)
        self.assertTrue(os.path.exists(save_path))
        
        # Verify saved configuration
        with open(save_path, 'r') as f:
            saved_config = json.load(f)
        self.assertEqual(saved_config['max_workers'], 8)
    
    def test_patient_timeline(self):
        """Test patient timeline creation and access."""
        # Test accessing existing timeline
        timeline = self.system.create_patient_timeline(self.mock_patient_id)
        self.assertIsNotNone(timeline)
        self.assertEqual(timeline['info']['subject_id'], self.mock_patient_id)
        
        # Test timeline content
        self.assertIn('timeline', timeline)
        self.assertIsInstance(timeline['timeline'], pd.DataFrame)
        self.assertIn('heart_rate', timeline['timeline'].columns)
    
    def test_signal_processing(self):
        """Test signal processing functionality."""
        # Process signals
        signals = self.system.process_patient_signals(self.mock_patient_id)
        self.assertIsNotNone(signals)
        
        # Check signal processing results
        self.assertIn('features', signals)
        self.assertIn('organ_status', signals)
        
        # Test stability analysis
        stability = self.system.analyze_physiological_stability(self.mock_patient_id)
        self.assertIsNotNone(stability)
        self.assertIsInstance(stability, pd.DataFrame)
        
        # Test stability report
        report = self.system.create_stability_report(self.mock_patient_id)
        self.assertIsNotNone(report)
        self.assertIn('organ_system_summary', report)
    
    def test_dynamical_analysis(self):
        """Test dynamical systems analysis."""
        # Analyze dynamical stability
        dynamical = self.system.analyze_dynamical_stability(self.mock_patient_id)
        self.assertIsNotNone(dynamical)
        
        # Check for vital sign analyses
        self.assertIn('heart_rate', dynamical)
        
        # Test critical transition detection
        transitions = self.system.detect_critical_transitions(self.mock_patient_id)
        self.assertIsNotNone(transitions)
    
    def test_infection_analysis(self):
        """Test infection treatment analysis."""
        # Analyze infection treatment
        infection = self.system.analyze_infection_treatment(self.mock_patient_id)
        self.assertIsNotNone(infection)
        
        # Check for treatment simulation results
        self.assertIn('treatment_simulation', infection)
        
        # Test treatment regimen comparison
        regimens = [
            {'dose': 1000, 'interval': 12},
            {'dose': 500, 'interval': 6},
            {'dose': 2000, 'interval': 24}
        ]
        comparison = self.system.compare_treatment_regimens(
            self.mock_patient_id, 'vancomycin', 's_aureus', regimens
        )
        self.assertIsNotNone(comparison)
        self.assertIn('all_regimens', comparison)
        self.assertIn('best_regimens', comparison)
    
    def test_visualization(self):
        """Test visualization functionality."""
        # Test vital signs visualization
        vital_signs_fig = self.system.visualize_vital_signs(self.mock_patient_id)
        self.assertIsNotNone(vital_signs_fig)
        plt.close(vital_signs_fig)
        
        # Test organ system visualization
        # First create stability report if not exists
        if (self.mock_patient_id not in self.system.patient_analyses or 
            'stability_report' not in self.system.patient_analyses[self.mock_patient_id]):
            self.system.create_stability_report(self.mock_patient_id)
        
        organ_system_fig = self.system.visualize_organ_system_status(self.mock_patient_id)
        self.assertIsNotNone(organ_system_fig)
        plt.close(organ_system_fig)
        
        # Test phase portrait
        phase_portrait_fig = self.system.visualize_phase_portrait(
            self.mock_patient_id, 'heart_rate', 'sbp'
        )
        self.assertIsNotNone(phase_portrait_fig)
        plt.close(phase_portrait_fig)
        
        # Test patient dashboard
        dashboard_fig = self.system.create_patient_dashboard(self.mock_patient_id)
        self.assertIsNotNone(dashboard_fig)
        plt.close(dashboard_fig)
    
    def test_batch_processing(self):
        """Test batch processing functionality."""
        # Create a batch of patients
        patient_ids = [self.mock_patient_id]
        
        # Process batch
        batch_results = self.system.process_patient_batch(
            patient_ids,
            analyses=['timeline', 'signals']
        )
        self.assertIsNotNone(batch_results)
        
        # Check batch results
        self.assertIn('subject_ids', batch_results)
        self.assertIn('results', batch_results)
        self.assertIn(self.mock_patient_id, batch_results['results'])
        
        # Generate batch report
        batch_id = list(self.system.batch_results.keys())[0]
        report_path = os.path.join(self.temp_dir, "batch_report.json")
        report = self.system.generate_batch_report(batch_id, report_path)
        self.assertIsNotNone(report)
        self.assertTrue(os.path.exists(report_path))
    
    def test_data_export(self):
        """Test data export and persistence."""
        # Export patient data
        export_path = os.path.join(self.temp_dir, "patient_data.json")
        success = self.system.export_patient_data(self.mock_patient_id, export_path)
        self.assertTrue(success)
        self.assertTrue(os.path.exists(export_path))
        
        # Save system state
        state_path = os.path.join(self.temp_dir, "system_state.pkl")
        success = self.system.save_state(state_path)
        self.assertTrue(success)
        self.assertTrue(os.path.exists(state_path))
        
        # Load system state
        new_system = BioDynamicsSystem()
        success = new_system.load_state(state_path)
        self.assertTrue(success)
        self.assertIn(self.mock_patient_id, new_system.patient_timelines)
    
    def test_clinical_api(self):
        """Test high-level clinical API."""
        # Test comprehensive patient analysis
        results = self.system.analyze_patient(
            self.mock_patient_id,
            analyses=['timeline', 'signals', 'stability']
        )
        self.assertIsNotNone(results)
        
        # Test patient stability status
        status = self.system.get_patient_stability_status(self.mock_patient_id)
        self.assertIsNotNone(status)
        self.assertIn('stability_status', status)
        
        # Test treatment recommendations
        recommendations = self.system.get_treatment_recommendations(self.mock_patient_id)
        self.assertIsNotNone(recommendations)
        self.assertIn('antibiotic', recommendations)
        
        # Test patient summary
        summary = self.system.get_patient_summary(self.mock_patient_id)
        self.assertIsNotNone(summary)
        self.assertEqual(summary['patient_id'], self.mock_patient_id)

if __name__ == '__main__':
    unittest.main()