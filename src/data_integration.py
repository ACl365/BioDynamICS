"""
BioDynamICS - Data Integration Module

This module handles loading and integrating MIMIC-III data tables
to create comprehensive patient timelines.

Author: [Your Name]
Date: [Current Date]
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class MimicPatientIntegrator:
    """
    Integrates MIMIC-III data tables to create comprehensive patient timelines
    with physiological measurements, interventions, and outcomes.
    """
    
    def __init__(self, data_path):
        """Initialize with path to MIMIC-III CSV files"""
        self.data_path = data_path
        self.tables = {}
        self.patient_timelines = {}
        
        # Log initialization
        self._log_operation("Initialized MimicPatientIntegrator")
        
    def _log_operation(self, message):
        """Log operations with timestamp"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {message}")
        
        # Future: Could write to a log file instead
    
    def load_table(self, table_name):
        """Load a single table from CSV"""
        file_path = os.path.join(self.data_path, f"{table_name}.csv")
        if os.path.exists(file_path):
            self.tables[table_name] = pd.read_csv(file_path)
            self._log_operation(f"Loaded {table_name}: {len(self.tables[table_name])} rows")
            return self.tables[table_name]
        else:
            self._log_operation(f"Error: File not found: {file_path}")
            return None
    
    def load_core_tables(self):
        """Load essential MIMIC tables"""
        core_tables = ['PATIENTS', 'ADMISSIONS', 'ICUSTAYS']
        
        for table in core_tables:
            self.load_table(table)
        
        # Create patient-admission-ICU stay linkage
        if all(table in self.tables for table in core_tables):
            self.patient_stays = self.tables['PATIENTS'].merge(
                self.tables['ADMISSIONS'], on='subject_id'
            ).merge(
                self.tables['ICUSTAYS'], on=['subject_id', 'hadm_id'], how='left'
            )
            
            self._log_operation(f"Created patient stays table: {len(self.patient_stays)} rows")
            return self.patient_stays
        else:
            self._log_operation("Error: Could not create patient stays - missing core tables")
            return None
    
    def load_chartevents_chunked(self, chunk_size=100000):
        """Load CHARTEVENTS in chunks due to file size"""
        file_path = os.path.join(self.data_path, "CHARTEVENTS.csv")
        
        if not os.path.exists(file_path):
            self._log_operation(f"Error: File not found: {file_path}")
            return None
        
        # Get file size to provide progress updates
        file_size = os.path.getsize(file_path) / (1024 * 1024)  # in MB
        self._log_operation(f"Processing CHARTEVENTS.csv ({file_size:.1f} MB)")
        
        # Process in chunks
        chunks = []
        total_rows = 0
        
        for i, chunk in enumerate(pd.read_csv(file_path, chunksize=chunk_size)):
            total_rows += len(chunk)
            
            # Add to our collection
            chunks.append(chunk)
            
            # Log progress
            self._log_operation(f"Processed chunk {i+1} with {len(chunk)} rows. Total: {total_rows} rows")
        
        # Combine chunks
        chartevents = pd.concat(chunks)
        self.tables['CHARTEVENTS'] = chartevents
        self._log_operation(f"Completed loading CHARTEVENTS with {len(chartevents)} rows")
        return chartevents
    
    def create_patient_timeline(self, subject_id):
        """
        Create integrated timeline for a single patient
        
        Parameters:
        -----------
        subject_id : int
            Patient subject ID
            
        Returns:
        --------
        dict with patient information and timeline DataFrame
        """
        # Check if we have the necessary tables
        required_tables = ['PATIENTS', 'ADMISSIONS', 'ICUSTAYS']
        if not all(table in self.tables for table in required_tables):
            self._log_operation(f"Error: Missing required tables for patient timeline")
            return None
        
        # Get patient data
        patient_row = self.tables['PATIENTS'][self.tables['PATIENTS']['subject_id'] == subject_id]
        if len(patient_row) == 0:
            self._log_operation(f"Error: Patient {subject_id} not found")
            return None
        
        # Get admissions
        admissions = self.tables['ADMISSIONS'][self.tables['ADMISSIONS']['subject_id'] == subject_id]
        if len(admissions) == 0:
            self._log_operation(f"Error: No admissions found for patient {subject_id}")
            return None
        
        # Get ICU stays
        icustays = self.tables['ICUSTAYS'][self.tables['ICUSTAYS']['subject_id'] == subject_id]
        
        # Create patient info dictionary
        patient_info = {
            'subject_id': subject_id,
            'gender': patient_row['gender'].iloc[0],
            'dob': patient_row['dob'].iloc[0],
            'dod': patient_row['dod'].iloc[0] if not pd.isna(patient_row['dod'].iloc[0]) else None,
            'expire_flag': patient_row['expire_flag'].iloc[0],
            'admissions': len(admissions),
            'icustays': len(icustays)
        }
        
        # Initialize timeline components
        timeline_components = []
        
        # Add measurements if available
        if 'CHARTEVENTS' in self.tables:
            # Get chart events for this patient
            chartevents = self.tables['CHARTEVENTS'][
                self.tables['CHARTEVENTS']['subject_id'] == subject_id
            ].copy()
            
            if len(chartevents) > 0:
                chartevents['measurement_time'] = pd.to_datetime(chartevents['charttime'])
                chartevents['event_type'] = 'chart'
                timeline_components.append(chartevents)
        
        # Add lab events if available
        if 'LABEVENTS' in self.tables:
            # Get lab events for this patient
            labevents = self.tables['LABEVENTS'][
                self.tables['LABEVENTS']['subject_id'] == subject_id
            ].copy()
            
            if len(labevents) > 0:
                labevents['measurement_time'] = pd.to_datetime(labevents['charttime'])
                labevents['event_type'] = 'lab'
                timeline_components.append(labevents)
        
        # Add medication events if available
        if 'PRESCRIPTIONS' in self.tables:
            # Get prescriptions for this patient
            prescriptions = self.tables['PRESCRIPTIONS'][
                self.tables['PRESCRIPTIONS']['subject_id'] == subject_id
            ].copy()
            
            if len(prescriptions) > 0:
                prescriptions['measurement_time'] = pd.to_datetime(prescriptions['startdate'])
                prescriptions['event_type'] = 'medication'
                timeline_components.append(prescriptions)
        
        # Combine all events into a timeline
        if timeline_components:
            timeline = pd.concat(timeline_components, sort=False)
            
            # Sort by time
            timeline = timeline.sort_values('measurement_time')
            
            # Store in dictionary
            self.patient_timelines[subject_id] = {
                'info': patient_info,
                'timeline': timeline
            }
            
            self._log_operation(f"Created timeline for patient {subject_id} with {len(timeline)} events")
            return self.patient_timelines[subject_id]
        else:
            self._log_operation(f"Warning: No events found for patient {subject_id}")
            return {'info': patient_info, 'timeline': pd.DataFrame()}
    
    def create_all_patient_timelines(self):
        """Create integrated timelines for all patients"""
        if 'PATIENTS' not in self.tables:
            self._log_operation("Error: PATIENTS table not loaded")
            return None
        
        # Get all patient IDs
        subject_ids = self.tables['PATIENTS']['subject_id'].unique()
        
        self._log_operation(f"Creating timelines for {len(subject_ids)} patients")
        
        # Process each patient
        for subject_id in subject_ids:
            self.create_patient_timeline(subject_id)
        
        self._log_operation(f"Completed timeline creation for {len(self.patient_timelines)} patients")
        return self.patient_timelines