import numpy as np
import pandas as pd
from scipy import signal, stats
import pywt
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose
import warnings

class PhysiologicalSignalProcessor:
    """
    Processes physiological time series data to extract meaningful features
    and patterns relevant to clinical state assessment.
    
    This class implements several advanced signal processing techniques:
    1. Time domain analysis (statistical features, trends)
    2. Frequency domain analysis (spectral features)
    3. Wavelet decomposition (multi-scale analysis)
    4. Stability metrics (sample entropy, Lyapunov exponents)
    5. Temporal pattern extraction (transitions, state changes)
    """
    
    def __init__(self):
        """Initialize the PhysiologicalSignalProcessor with reference ranges."""
        
        # Define normal reference ranges for common vital signs and lab values
        self.reference_ranges = {
            # Vital signs
            'heart_rate': (60, 100),              # beats per minute
            'respiratory_rate': (12, 20),         # breaths per minute
            'sbp': (90, 140),                     # Systolic BP (mmHg)
            'dbp': (60, 90),                      # Diastolic BP (mmHg)
            'map': (70, 100),                     # Mean arterial pressure (mmHg)
            'temperature': (36.5, 37.5),          # Celsius
            'o2_saturation': (94, 100),           # percent
            
            # Common labs
            'wbc': (4.5, 11.0),                   # White blood cells (K/uL)
            'hemoglobin': (12.0, 17.5),           # g/dL
            'platelet': (150, 450),               # K/uL
            'sodium': (135, 145),                 # mmol/L
            'potassium': (3.5, 5.0),              # mmol/L
            'chloride': (98, 107),                # mmol/L
            'bicarbonate': (22, 29),              # mmol/L
            'bun': (7, 20),                       # Blood urea nitrogen (mg/dL)
            'creatinine': (0.6, 1.2),             # mg/dL
            'glucose': (70, 110),                 # mg/dL
            'calcium': (8.5, 10.5),               # mg/dL
            'magnesium': (1.5, 2.5),              # mg/dL
            'phosphate': (2.5, 4.5),              # mg/dL
            'ast': (10, 40),                      # Aspartate aminotransferase (U/L)
            'alt': (7, 56),                       # Alanine aminotransferase (U/L)
            'bilirubin_total': (0.1, 1.2),        # mg/dL
            'albumin': (3.4, 5.4),                # g/dL
            'lactate': (0.5, 2.2),                # mmol/L
            'ph': (7.35, 7.45),                   # pH units
            'pao2': (75, 100),                    # mmHg
            'paco2': (35, 45),                    # mmHg
            'base_excess': (-2, 2),               # mEq/L
        }
        
        # Define categories of measurements for organ systems
        self.organ_system_measures = {
            'cardiovascular': [
                'heart_rate', 'sbp', 'dbp', 'map', 'troponin', 'bnp', 'lactate'
            ],
            'respiratory': [
                'respiratory_rate', 'o2_saturation', 'pao2', 'paco2', 'fio2',
                'peep', 'tidal_volume', 'plateau_pressure'
            ],
            'renal': [
                'creatinine', 'bun', 'urine_output', 'urine_sodium'
            ],
            'hepatic': [
                'ast', 'alt', 'bilirubin_total', 'bilirubin_direct', 'albumin',
                'inr', 'ammonia'
            ],
            'hematologic': [
                'wbc', 'hemoglobin', 'platelet', 'inr', 'ptt', 'fibrinogen', 'd_dimer'
            ],
            'metabolic': [
                'sodium', 'potassium', 'chloride', 'bicarbonate', 'glucose',
                'calcium', 'magnesium', 'phosphate'
            ],
            'neurologic': [
                'gcs', 'gcs_motor', 'gcs_verbal', 'gcs_eyes', 'pupils'
            ],
            'infectious': [
                'temperature', 'wbc', 'bands', 'procalcitonin', 'crp'
            ]
        }
        
        # Mapping MIMIC-III itemids to standard names (partial mapping)
        # This would be extended with a comprehensive mapping
        self.itemid_mapping = {
            # Heart rate
            211: 'heart_rate',
            220045: 'heart_rate',
            
            # Blood pressure
            51: 'sbp',
            442: 'sbp',
            455: 'sbp',
            6701: 'sbp',
            220050: 'sbp',
            
            8368: 'dbp',
            8440: 'dbp',
            8441: 'dbp',
            8555: 'dbp',
            220051: 'dbp',
            
            456: 'map',
            52: 'map',
            220052: 'map',
            220181: 'map',
            225312: 'map',
            
            # Respiratory
            615: 'respiratory_rate',
            618: 'respiratory_rate',
            220210: 'respiratory_rate',
            
            646: 'o2_saturation',
            220277: 'o2_saturation',
            
            3420: 'fio2',
            3422: 'fio2',
            223835: 'fio2',
            
            505: 'peep',
            506: 'peep',
            220339: 'peep',
            
            # Temperature
            223761: 'temperature',
            676: 'temperature', 
            678: 'temperature',
            
            # Labs (partial list)
            50800: 'wbc',
            51300: 'wbc',
            
            50912: 'creatinine',
            51082: 'creatinine',
            
            50971: 'potassium',
            51237: 'potassium',
            
            50983: 'sodium',
            51265: 'sodium',
            
            50802: 'glucose',
            51478: 'glucose',
        }
        
        # Initialize logging
        self._log_initialization()
    
    def _log_initialization(self):
        """Log initialization information."""
        print(f"Initialized PhysiologicalSignalProcessor")
        print(f"- {len(self.reference_ranges)} clinical measures with reference ranges")
        print(f"- {len(self.organ_system_measures)} organ systems defined")
        print(f"- {len(self.itemid_mapping)} MIMIC itemIDs mapped to standard names")
    
    def preprocess_timeseries(self, series, measurement_name=None):
        """
        Preprocess a time series for analysis by:
        1. Removing outliers
        2. Handling missing values
        3. Resampling to regular intervals if needed
        
        Parameters:
        -----------
        series : pandas.Series
            The time series to preprocess with datetime index
        measurement_name : str, optional
            Name of the measurement for reference range checking
            
        Returns:
        --------
        pandas.Series
            Preprocessed series
        """
        if series is None or len(series) == 0:
            return None
        
        # Make a copy to avoid modifying the original
        processed = series.copy()
        
        # Sort by timestamp
        if isinstance(processed.index, pd.DatetimeIndex):
            processed = processed.sort_index()
        
        # Remove extreme outliers using IQR method
        Q1 = processed.quantile(0.25)
        Q3 = processed.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 3 * IQR
        upper_bound = Q3 + 3 * IQR
        
        # If we have reference ranges, use them to refine bounds
        if measurement_name and measurement_name in self.reference_ranges:
            ref_lower, ref_upper = self.reference_ranges[measurement_name]
            # Expand the reference range a bit to account for abnormal but plausible values
            range_width = ref_upper - ref_lower
            allowed_lower = max(lower_bound, ref_lower - range_width)
            allowed_upper = min(upper_bound, ref_upper + range_width)
        else:
            allowed_lower = lower_bound
            allowed_upper = upper_bound
        
        # Apply the bounds
        processed = processed[(processed >= allowed_lower) & (processed <= allowed_upper)]
        
        # Handle regular sampling if needed
        if isinstance(processed.index, pd.DatetimeIndex):
            # Check if timeseries has regular sampling
            timestamps = processed.index
            if len(timestamps) > 1:
                intervals = np.diff(timestamps.astype(np.int64)) / 10**9  # in seconds
                if np.std(intervals) / np.mean(intervals) > 0.1:  # If high variability in sampling
                    # Irregular sampling detected, but we'll leave it as is for now
                    # More sophisticated resampling would be applied based on specific needs
                    pass
        
        return processed
    
    def extract_time_domain_features(self, series, measurement_name=None):
        """
        Extract statistical features from the time domain.
        
        Parameters:
        -----------
        series : pandas.Series
            Time series data
        measurement_name : str, optional
            Name of the measurement
            
        Returns:
        --------
        dict
            Dictionary of time domain features
        """
        # Preprocess the series
        clean_series = self.preprocess_timeseries(series, measurement_name)
        
        if clean_series is None or len(clean_series) < 3:
            return {}  # Not enough data for meaningful analysis
        
        features = {}
        
        # Basic statistics
        features['mean'] = clean_series.mean()
        features['median'] = clean_series.median()
        features['std'] = clean_series.std()
        features['min'] = clean_series.min()
        features['max'] = clean_series.max()
        features['range'] = features['max'] - features['min']
        features['iqr'] = clean_series.quantile(0.75) - clean_series.quantile(0.25)
        
        # Variability metrics
        if features['mean'] != 0:
            features['cv'] = features['std'] / features['mean']  # Coefficient of variation
        else:
            features['cv'] = np.nan
            
        # Calculate trend using linear regression
        if isinstance(clean_series.index, pd.DatetimeIndex):
            # Convert timestamps to seconds since start
            x = (clean_series.index - clean_series.index[0]).total_seconds().values
        else:
            x = np.arange(len(clean_series))
            
        y = clean_series.values
        
        if len(x) > 1:  # Need at least 2 points for regression
            try:
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
                features['trend_slope'] = slope
                features['trend_r_squared'] = r_value**2
                features['trend_p_value'] = p_value
            except:
                # Handle potential errors in regression
                features['trend_slope'] = np.nan
                features['trend_r_squared'] = np.nan
                features['trend_p_value'] = np.nan
        
        # Check for reference range violations if available
        if measurement_name and measurement_name in self.reference_ranges:
            ref_lower, ref_upper = self.reference_ranges[measurement_name]
            features['pct_below_ref'] = (clean_series < ref_lower).mean() * 100
            features['pct_above_ref'] = (clean_series > ref_upper).mean() * 100
            features['pct_in_ref'] = ((clean_series >= ref_lower) & 
                                      (clean_series <= ref_upper)).mean() * 100
                
        # Advanced statistics
        if len(clean_series) >= 4:  # Need more data for these
            features['skew'] = stats.skew(clean_series)
            features['kurtosis'] = stats.kurtosis(clean_series)
        
        return features
    
    def extract_frequency_domain_features(self, series, measurement_name=None, fs=None):
        """
        Extract features from the frequency domain using FFT.
        
        Parameters:
        -----------
        series : pandas.Series
            Time series data
        measurement_name : str, optional
            Name of the measurement
        fs : float, optional
            Sampling frequency in Hz. If None, will be estimated from timestamps
            
        Returns:
        --------
        dict
            Dictionary of frequency domain features
        """
        # Preprocess the series
        clean_series = self.preprocess_timeseries(series, measurement_name)
        
        if clean_series is None or len(clean_series) < 10:  # Need enough points for spectral analysis
            return {}
        
        features = {}
        
        # Determine sampling frequency if not provided
        if fs is None and isinstance(clean_series.index, pd.DatetimeIndex):
            # Calculate median time difference in seconds
            time_diffs = np.diff(clean_series.index.astype(np.int64)) / 10**9
            if len(time_diffs) > 0:
                median_diff = np.median(time_diffs)
                if median_diff > 0:
                    fs = 1.0 / median_diff
                else:
                    fs = 1.0  # Default if can't determine
            else:
                fs = 1.0  # Default if can't determine
        elif fs is None:
            fs = 1.0  # Default to unit frequency if not timestamp index
        
        # Get values (detrend to improve spectral estimation)
        values = clean_series.values
        detrended = signal.detrend(values)
        
        # Compute PSD (Power Spectral Density)
        try:
            freqs, psd = signal.welch(detrended, fs=fs, nperseg=min(256, len(detrended)))
            
            if len(psd) > 0:
                # Extract frequency features
                features['power_total'] = np.sum(psd)
                features['freq_peak'] = freqs[np.argmax(psd)]
                features['power_peak'] = np.max(psd)
                
                # Power in different frequency bands (these would be adjusted based on the signal)
                # Here we use general bands, but specific physiological signals need custom bands
                if len(freqs) > 3:
                    # Very low frequency (first 1/4 of spectrum)
                    vlf_idx = int(len(freqs) / 4)
                    features['power_vlf'] = np.sum(psd[:vlf_idx]) / features['power_total']
                    
                    # Low frequency (second 1/4 of spectrum)
                    lf_idx = int(len(freqs) / 2)
                    features['power_lf'] = np.sum(psd[vlf_idx:lf_idx]) / features['power_total']
                    
                    # High frequency (second half of spectrum)
                    features['power_hf'] = np.sum(psd[lf_idx:]) / features['power_total']
                    
                    # LF/HF ratio (important for HRV analysis)
                    if features['power_hf'] > 0:
                        features['lf_hf_ratio'] = features['power_lf'] / features['power_hf']
        except Exception as e:
            # Handle potential errors in spectral analysis
            print(f"Error in frequency analysis: {e}")
            pass
        
        return features
    
    def extract_wavelet_features(self, series, measurement_name=None):
        """
        Extract multi-scale features using wavelet decomposition.
        
        Parameters:
        -----------
        series : pandas.Series
            Time series data
        measurement_name : str, optional
            Name of the measurement
            
        Returns:
        --------
        dict
            Dictionary of wavelet features
        """
        # Preprocess the series
        clean_series = self.preprocess_timeseries(series, measurement_name)
        
        if clean_series is None or len(clean_series) < 8:  # Need enough points for wavelet analysis
            return {}
        
        features = {}
        
        # Get values
        values = clean_series.values
        
        try:
            # Determine maximum decomposition level based on signal length
            max_level = pywt.dwt_max_level(len(values), 'db4')
            level = min(4, max_level)  # Use up to 4 levels or max possible
            
            # Perform wavelet decomposition
            coeffs = pywt.wavedec(values, 'db4', level=level)
            
            # Extract energy features from each level
            features['wavelet_approx_energy'] = np.sum(coeffs[0]**2)
            total_energy = features['wavelet_approx_energy']
            
            for i in range(1, len(coeffs)):
                detail_energy = np.sum(coeffs[i]**2)
                features[f'wavelet_detail{i}_energy'] = detail_energy
                total_energy += detail_energy
            
            # Normalize energy by total
            if total_energy > 0:
                for key in list(features.keys()):
                    if key.startswith('wavelet_'):
                        features[key + '_norm'] = features[key] / total_energy
        except Exception as e:
            # Handle potential errors in wavelet analysis
            print(f"Error in wavelet analysis: {e}")
            pass
        
        return features
    
    def calculate_sample_entropy(self, series, m=2, r=0.2):
        """
        Calculate sample entropy, a measure of system complexity.
        
        Parameters:
        -----------
        series : array-like
            Time series data
        m : int
            Embedding dimension
        r : float
            Tolerance (as a fraction of standard deviation)
            
        Returns:
        --------
        float
            Sample entropy value
        """
        if len(series) < 100:  # Sample entropy needs sufficient data
            return np.nan
            
        # Normalize the series
        series = np.array(series)
        series = (series - np.mean(series)) / np.std(series)
        
        # Calculate tolerance distance
        r = r * np.std(series)
        
        # Create embedding vectors
        def create_vectors(m):
            vectors = []
            for i in range(len(series) - m + 1):
                vectors.append(series[i:i+m])
            return np.array(vectors)
        
        # Calculate the number of matches
        def count_matches(vectors, tolerance):
            N = len(vectors)
            B = 0
            for i in range(N - 1):
                # Calculate distances
                distances = np.max(np.abs(vectors[i] - vectors[i+1:]), axis=1)
                # Count matches
                B += np.sum(distances < tolerance)
            return B / (N * (N - 1) / 2)  # Normalize
        
        # Calculate B(m) and B(m+1)
        try:
            vectors_m = create_vectors(m)
            vectors_m1 = create_vectors(m + 1)
            
            B_m = count_matches(vectors_m, r)
            B_m1 = count_matches(vectors_m1, r)
            
            # Calculate sample entropy
            if B_m > 0 and B_m1 > 0:
                return -np.log(B_m1 / B_m)
            else:
                return np.nan
        except:
            return np.nan
    
    def calculate_stability_metrics(self, series, measurement_name=None):
        """
        Calculate metrics that quantify system stability and complexity.
        
        Parameters:
        -----------
        series : pandas.Series
            Time series data
        measurement_name : str, optional
            Name of the measurement
            
        Returns:
        --------
        dict
            Dictionary of stability metrics
        """
        # Preprocess the series
        clean_series = self.preprocess_timeseries(series, measurement_name)
        
        if clean_series is None or len(clean_series) < 10:
            return {}
        
        features = {}
        values = clean_series.values
        
        # Calculate autocorrelation metrics
        try:
            if len(values) >= 10:
                # Autocorrelation at lag 1
                acf_values = acf(values, nlags=min(10, len(values) - 1), fft=True)
                features['autocorr_lag1'] = acf_values[1] if len(acf_values) > 1 else np.nan
                
                # Autocorrelation decay (from lag 1 to 5)
                if len(acf_values) > 5:
                    features['autocorr_decay'] = acf_values[1] - acf_values[5]
        except:
            # Handle potential errors in autocorrelation
            pass
        
        # Calculate complexity metrics
        try:
            if len(values) >= 100:
                # Sample entropy (higher values indicate more complexity/randomness)
                features['sample_entropy'] = self.calculate_sample_entropy(values)
                
                # Approximate entropy (similar to sample entropy but different algorithm)
                # This would be implemented separately if needed
        except:
            # Handle potential errors in entropy calculations
            pass
        
        # Calculate variability metrics
        if len(values) >= 3:
            # Root mean square of successive differences (RMSSD)
            # Important for heart rate variability analysis
            successive_diffs = np.diff(values)
            features['rmssd'] = np.sqrt(np.mean(successive_diffs**2))
            
            # Percentage of successive differences exceeding 50ms (pNN50)
            # Also used in HRV analysis
            if measurement_name == 'heart_rate':
                # Convert heart rate differences to RR interval differences (approx)
                # HR of 60 bpm = RR interval of 1000ms
                # Change in RR ~= -1000²/HR² * change in HR
                avg_hr = np.mean(values)
                if avg_hr > 0:
                    rr_diffs = -1000**2 / (avg_hr**2) * successive_diffs
                    features['pnn50'] = np.mean(np.abs(rr_diffs) > 50) * 100
        
        return features
    
    def extract_all_features_from_series(self, series, measurement_name=None):
        """
        Extract all features from a single time series.
        
        Parameters:
        -----------
        series : pandas.Series
            Time series data with datetime index
        measurement_name : str, optional
            Name of the measurement
            
        Returns:
        --------
        dict
            Dictionary of all extracted features
        """
        if series is None or len(series) < 3:
            return {}
        
        # Get features from different domains
        time_features = self.extract_time_domain_features(series, measurement_name)
        freq_features = self.extract_frequency_domain_features(series, measurement_name)
        wavelet_features = self.extract_wavelet_features(series, measurement_name)
        stability_features = self.calculate_stability_metrics(series, measurement_name)
        
        # Combine all features
        features = {}
        for feature_dict in [time_features, freq_features, wavelet_features, stability_features]:
            for key, value in feature_dict.items():
                features[f"{measurement_name}_{key}" if measurement_name else key] = value
        
        return features
    
    def process_vital_signs(self, patient_data, vital_sign_map=None):
        """
        Process all vital signs for a patient and extract features.
        
        Parameters:
        -----------
        patient_data : pandas.DataFrame
            DataFrame containing patient vital signs
        vital_sign_map : dict, optional
            Mapping of column names to standard vital sign names
            
        Returns:
        --------
        dict
            Dictionary of vital sign features
        """
        if patient_data is None or len(patient_data) == 0:
            return {}
        
        # Use the provided mapping or the default one
        if vital_sign_map is None:
            vital_sign_map = self.itemid_mapping
            
        features = {}
        vital_series = {}
            
        # Check if we have itemid and value columns (MIMIC format)
        if 'itemid' in patient_data.columns and 'valuenum' in patient_data.columns:
            # Group by measurement type
            for itemid, group in patient_data.groupby('itemid'):
                if itemid in vital_sign_map:
                    measurement = vital_sign_map[itemid]
                    
                    # Create time series for this vital sign
                    if 'charttime' in group.columns:
                        ts = pd.Series(
                            group['valuenum'].values,
                            index=pd.to_datetime(group['charttime'])
                        )
                    else:
                        ts = pd.Series(group['valuenum'].values)
                        
                    if measurement in vital_series:
                        # Append to existing series
                        vital_series[measurement] = pd.concat([vital_series[measurement], ts])
                    else:
                        vital_series[measurement] = ts
        else:
            # Assume the columns are already named by measurement
            for col in patient_data.columns:
                if col in self.reference_ranges:
                    if isinstance(patient_data.index, pd.DatetimeIndex):
                        vital_series[col] = patient_data[col]
                    else:
                        # Try to find a timestamp column
                        if 'charttime' in patient_data.columns:
                            ts = pd.Series(
                                patient_data[col].values,
                                index=pd.to_datetime(patient_data['charttime'])
                            )
                            vital_series[col] = ts
                        else:
                            vital_series[col] = patient_data[col]
        
        # Sort each time series and extract features
        for measurement, series in vital_series.items():
            if isinstance(series.index, pd.DatetimeIndex):
                series = series.sort_index()
            
            # Extract features from this vital sign
            measurement_features = self.extract_all_features_from_series(
                series, measurement_name=measurement
            )
            
            # Add to overall features
            features.update(measurement_features)
        
        return features
    
    def process_patient_timeline(self, patient_timeline):
        """
        Process a complete patient timeline to extract physiological features.
        
        Parameters:
        -----------
        patient_timeline : dict
            Patient timeline dictionary with 'info' and 'timeline' keys
            
        Returns:
        --------
        dict
            Dictionary with extracted features and organ system status
        """
        if not patient_timeline or 'timeline' not in patient_timeline:
            print("Error: Invalid patient timeline format")
            return {}
        
        timeline = patient_timeline['timeline']
        if timeline is None or len(timeline) == 0:
            print("Error: Empty patient timeline")
            return {}
        
        # Initialize results
        results = {
            'patient_id': patient_timeline.get('info', {}).get('subject_id', None),
            'features': {},
            'organ_status': {},
            'allostatic_load': np.nan,
            'timestamps': []
        }
        
        # Filter for chart events (measurements)
        if 'event_type' in timeline.columns:
            chart_events = timeline[timeline['event_type'] == 'chart'].copy()
            lab_events = timeline[timeline['event_type'] == 'lab'].copy()
        else:
            # If event_type not available, try to use other columns to identify events
            chart_events = pd.DataFrame()
            lab_events = pd.DataFrame()
            
            # Try to identify chart events if itemid and other MIMIC-specific columns exist
            if all(col in timeline.columns for col in ['itemid', 'valuenum', 'charttime']):
                chart_events = timeline.copy()
        
        # Process vital signs if available
        if len(chart_events) > 0:
            vital_features = self.process_vital_signs(chart_events)
            results['features'].update(vital_features)
        
        # Process lab values if available
        if len(lab_events) > 0:
            lab_features = self.process_vital_signs(lab_events)
            results['features'].update(lab_features)
        
        # Calculate organ system status
        if results['features']:
            results['organ_status'] = self.calculate_organ_system_status(results['features'])
            results['allostatic_load'] = self.calculate_allostatic_load(results['organ_status'])
        
        return results
    
    def process_patient_timeline(self, patient_timeline):
        """
        Process a complete patient timeline to extract physiological features.
        
        Parameters:
        -----------
        patient_timeline : dict
            Patient timeline dictionary with 'info' and 'timeline' keys
            
        Returns:
        --------
        dict
            Dictionary with extracted features and organ system status
        """
        if not patient_timeline or 'timeline' not in patient_timeline:
            print("Error: Invalid patient timeline format")
            return {}
        
        timeline = patient_timeline['timeline']
        if timeline is None or len(timeline) == 0:
            print("Error: Empty patient timeline")
            return {}
        
        # Initialize results
        results = {
            'patient_id': patient_timeline.get('info', {}).get('subject_id', None),
            'features': {},
            'organ_status': {},
            'allostatic_load': np.nan,
            'timestamps': []
        }
        
        # Filter for chart events (measurements)
        if 'event_type' in timeline.columns:
            chart_events = timeline[timeline['event_type'] == 'chart'].copy()
            lab_events = timeline[timeline['event_type'] == 'lab'].copy()
        else:
            # If event_type not available, try to use other columns to identify events
            chart_events = pd.DataFrame()
            lab_events = pd.DataFrame()
            
            # Try to identify chart events if itemid and other MIMIC-specific columns exist
            if all(col in timeline.columns for col in ['itemid', 'valuenum', 'charttime']):
                chart_events = timeline.copy()
        
        # Process vital signs if available
        if len(chart_events) > 0:
            vital_features = self.process_vital_signs(chart_events)
            results['features'].update(vital_features)
        
        # Process lab values if available
        if len(lab_events) > 0:
            lab_features = self.process_vital_signs(lab_events)
            results['features'].update(lab_features)
        
        # Calculate organ system status
        if results['features']:
            results['organ_status'] = self.calculate_organ_system_status(results['features'])
            results['allostatic_load'] = self.calculate_allostatic_load(results['organ_status'])
        
        return results
        
    def analyze_physiological_stability(self, patient_timeline, window_hours=24, step_hours=6):
        """
        Analyze physiological stability over time using sliding windows.
        
        Parameters:
        -----------
        patient_timeline : dict
            Patient timeline dictionary with 'info' and 'timeline' keys
        window_hours : int, optional
            Size of analysis window in hours
        step_hours : int, optional
            Step size for sliding window in hours
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with stability metrics for each time window
        """
        if not patient_timeline or 'timeline' not in patient_timeline:
            print("Error: Invalid patient timeline format")
            return pd.DataFrame()
        
        timeline = patient_timeline['timeline']
        if timeline is None or len(timeline) == 0:
            print("Error: Empty patient timeline")
            return pd.DataFrame()
        
        # Ensure we have measurement_time column
        if 'measurement_time' not in timeline.columns:
            print("Error: Timeline missing measurement_time column")
            return pd.DataFrame()
        
        # Convert to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(timeline['measurement_time']):
            timeline['measurement_time'] = pd.to_datetime(timeline['measurement_time'])
        
        # Sort by time
        timeline = timeline.sort_values('measurement_time')
        
        # Get time range
        start_time = timeline['measurement_time'].min()
        end_time = timeline['measurement_time'].max()
        
        # Create sliding windows
        window_td = pd.Timedelta(hours=window_hours)
        step_td = pd.Timedelta(hours=step_hours)
        
        windows = []
        current_start = start_time
        
        while current_start + window_td <= end_time:
            current_end = current_start + window_td
            
            # Get data for this window
            window_data = timeline[
                (timeline['measurement_time'] >= current_start) &
                (timeline['measurement_time'] < current_end)
            ]
            
            # Only analyze if we have enough data
            if len(window_data) >= 10:
                # Create a sub-timeline with just this window
                window_timeline = {
                    'info': patient_timeline.get('info', {}),
                    'timeline': window_data
                }
                
                # Process this window
                window_results = self.process_patient_timeline(window_timeline)
                
                # Add window metadata
                window_results['window_start'] = current_start
                window_results['window_end'] = current_end
                window_results['window_hours'] = window_hours
                window_results['data_points'] = len(window_data)
                
                windows.append(window_results)
            
            # Move to next window
            current_start += step_td
        
        # Convert to DataFrame
        if not windows:
            return pd.DataFrame()
            
        # Extract key metrics for each window
        rows = []
        for window in windows:
            row = {
                'patient_id': window['patient_id'],
                'window_start': window['window_start'],
                'window_end': window['window_end'],
                'data_points': window['data_points'],
                'allostatic_load': window['allostatic_load']
            }
            
            # Add organ system scores
            for system, status in window['organ_status'].items():
                row[f"{system}_score"] = status['system_score']
            
            # Add select features (can be customized)
            for key, value in window['features'].items():
                if key.endswith('_mean') or key.endswith('_trend_slope'):
                    row[key] = value
            
            rows.append(row)
        
        return pd.DataFrame(rows)  

    def create_stability_report(self, patient_timeline):
        """
        Create a comprehensive stability report for a patient.
        
        Parameters:
        -----------
        patient_timeline : dict
            Patient timeline dictionary with 'info' and 'timeline' keys
            
        Returns:
        --------
        dict
            Dictionary with stability analysis results
        """
        # Process the full timeline
        full_results = self.process_patient_timeline(patient_timeline)
        
        # Analyze stability over time
        stability_over_time = self.analyze_physiological_stability(
            patient_timeline, window_hours=24, step_hours=8
        )
        
        # Create comprehensive report
        report = {
            'patient_info': patient_timeline.get('info', {}),
            'overall_results': full_results,
            'stability_over_time': stability_over_time,
            'organ_system_summary': {},
            'critical_values': [],
            'stability_trends': {}
        }
        
        # Add organ system summary
        for system, status in full_results['organ_status'].items():
            report['organ_system_summary'][system] = {
                'score': status['system_score'],
                'abnormal_measures': status['abnormal_measures'],
                'total_measures': status['n_measures'],
                'max_deviation': status['max_deviation']
            }
        
        # Find critical values (significant deviations from normal)
        if full_results['features']:
            for key, value in full_results['features'].items():
                if '_mean' in key and isinstance(value, (int, float)):
                    # Extract the measurement name
                    measurement = key.split('_mean')[0]
                    
                    if measurement in self.reference_ranges:
                        lower, upper = self.reference_ranges[measurement]
                        
                        # Check if significantly outside reference range
                        if value < lower * 0.8 or value > upper * 1.2:
                            report['critical_values'].append({
                                'measurement': measurement,
                                'value': value,
                                'reference_range': self.reference_ranges[measurement],
                                'percent_deviation': max(
                                    (lower - value) / lower * 100 if value < lower else 0,
                                    (value - upper) / upper * 100 if value > upper else 0
                                )
                            })
        
        # Add stability trends if we have time windows
        if not stability_over_time.empty and len(stability_over_time) > 1:
            # Calculate trends in allostatic load
            x = np.arange(len(stability_over_time))
            y = stability_over_time['allostatic_load'].values
            
            if len(y) > 1 and not np.all(np.isnan(y)):
                # Remove NaN values for regression
                mask = ~np.isnan(y)
                if np.sum(mask) > 1:
                    try:
                        slope, _, r_value, p_value, _ = stats.linregress(x[mask], y[mask])
                        report['stability_trends']['allostatic_load_trend'] = slope
                        report['stability_trends']['allostatic_load_trend_r2'] = r_value**2
                        report['stability_trends']['allostatic_load_trend_p'] = p_value
                    except:
                        pass
            
            # Also calculate trends for each organ system
            for system in report['organ_system_summary'].keys():
                col = f"{system}_score"
                if col in stability_over_time.columns:
                    y = stability_over_time[col].values
                    if len(y) > 1 and not np.all(np.isnan(y)):
                        mask = ~np.isnan(y)
                        if np.sum(mask) > 1:
                            try:
                                slope, _, r_value, p_value, _ = stats.linregress(x[mask], y[mask])
                                report['stability_trends'][f"{system}_trend"] = slope
                                report['stability_trends'][f"{system}_trend_r2"] = r_value**2
                                report['stability_trends'][f"{system}_trend_p"] = p_value
                            except:
                                pass
        
        return report    
    
    def calculate_organ_system_status(self, features_dict):
        """
        Calculate health status for each organ system based on extracted features.
        
        Parameters:
        -----------
        features_dict : dict
            Dictionary of extracted features
            
        Returns:
        --------
        dict
            Dictionary with organ system status scores
        """
        if not features_dict:
            return {}
        
        organ_status = {}
        
        # For each organ system, calculate a status score
        for system, measures in self.organ_system_measures.items():
            # Count how many measures we have for this system
            available_measures = [m for m in measures if any(f.startswith(f"{m}_") for f in features_dict.keys())]
            
            if not available_measures:
                continue  # Skip if no measures available
                
            # Initialize status metrics
            status = {
                'n_measures': len(available_measures),
                'abnormal_measures': 0,
                'avg_deviation': 0,
                'max_deviation': 0,
                'system_score': 0  # Higher is worse
            }
            
            # Calculate deviations from normal
            deviations = []
            
            for measure in available_measures:
                # Check if we have mean and reference range
                mean_key = f"{measure}_mean"
                if mean_key in features_dict and measure in self.reference_ranges:
                    value = features_dict[mean_key]
                    lower, upper = self.reference_ranges[measure]
                    
                    # Calculate how far outside reference range (normalized)
                    range_width = upper - lower
                    if range_width == 0:  # Avoid division by zero
                        continue
                        
                    if value < lower:
                        deviation = (lower - value) / range_width
                    elif value > upper:
                        deviation = (value - upper) / range_width
                    else:
                        deviation = 0
                        
                    deviations.append(deviation)
                    
                    if deviation > 0:
                        status['abnormal_measures'] += 1
            
            # Calculate summary statistics
            if deviations:
                status['avg_deviation'] = np.mean(deviations)
                status['max_deviation'] = np.max(deviations)
                
                # Calculate overall score (blend of breadth and severity of abnormalities)
                abnormal_percent = status['abnormal_measures'] / status['n_measures']
                status['system_score'] = (0.7 * status['max_deviation'] + 
                                         0.3 * status['avg_deviation']) * (0.5 + 0.5 * abnormal_percent)
            
            # Add to results
            organ_status[system] = status
        
        return organ_status
    
    def calculate_allostatic_load(self, organ_status):
        """
        Calculate allostatic load - a measure of cumulative physiological strain.
        
        Parameters:
        -----------
        organ_status : dict
            Dictionary with organ system status scores
            
        Returns:
        --------
        float
            Allostatic load score (higher = worse)
        """
        if not organ_status:
            return np.nan
        
        # Initialize components
        breadth = 0  # How many systems are affected
        severity = 0  # How severely they're affected
        
        # System weights (some systems have more impact than others)
        system_weights = {
            'cardiovascular': 1.2,
            'respiratory': 1.2,
            'renal': 1.0,
            'hepatic': 0.8,
            'hematologic': 0.7,
            'metabolic': 0.9,
            'neurologic': 1.1,
            'infectious': 1.0
        }
        
        # Calculate components
        for system, status in organ_status.items():
            weight = system_weights.get(system, 1.0)
            
            # Add to breadth if system is affected
            if status['system_score'] > 0:
                breadth += weight
                
            # Add to severity based on weighted system score
            severity += status['system_score'] * weight
                
        # Total number of systems we're evaluating
        total_systems = sum(system_weights.values())
        
        # Normalize breadth by total possible
        if total_systems > 0:
            breadth_normalized = breadth / total_systems
        else:
            breadth_normalized = 0
            
        # Calculate overall allostatic load
        # Combine breadth (how many systems affected) with severity (how badly)
        allostatic_load = 0.4 * breadth_normalized + 0.6 * severity
        
        return allostatic_load