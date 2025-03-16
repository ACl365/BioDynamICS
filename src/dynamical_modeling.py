import numpy as np
import pandas as pd
from scipy import signal, stats, integrate
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from datetime import datetime
import warnings

class DynamicalSystemsModeler:
    
    def __init__(self):
        # Configuration parameters for state space reconstruction
        self.default_embedding_dimension = 3
        self.default_time_delay = 1
        self.max_embedding_dimension = 10
        
        # Configuration for stability analysis
        self.lyapunov_k = 5  # Nearest neighbors for Lyapunov calculation
        self.lyapunov_max_steps = 20  # Maximum prediction steps for Lyapunov
        
        # System control parameters (for specific physiological models)
        self.system_parameters = {
            'cardiovascular': {
                'heart_rate_baseline': 70,      # bpm
                'contractility_factor': 0.5,    # dimensionless
                'vascular_resistance': 1.0,     # dimensionless
                'baroreceptor_gain': 1.0,       # dimensionless
                'baroreceptor_delay': 2.0       # seconds
            },
            'respiratory': {
                'respiratory_rate_baseline': 12,  # breaths/min
                'tidal_volume_baseline': 500,     # mL
                'chemoreceptor_gain_co2': 2.0,    # dimensionless
                'chemoreceptor_gain_o2': 1.0,     # dimensionless
                'respiratory_drive': 1.0          # dimensionless
            }
        }
        
        # Early warning signal parameters
        self.ews_window_size = 20  # Window size for early warning signals
        self.ews_step_size = 5     # Step size for sliding window
        
        # Initialize logging
        self._log_initialization()
    
    def _log_initialization(self):
        # Log initialization information
        print("Initialized DynamicalSystemsModeler")
        print(f"- Default embedding parameters: dimension={self.default_embedding_dimension}, delay={self.default_time_delay}")
        for system, params in self.system_parameters.items():
            print(f"- {system.title()} system: {len(params)} control parameters")
    
    # =========================================================================
    # State Space Reconstruction Methods
    # =========================================================================
    
    def time_delay_embedding(self, time_series, embedding_dimension=None, time_delay=None):
        # Perform time-delay embedding to reconstruct state space from a time series
        if embedding_dimension is None:
            embedding_dimension = self.default_embedding_dimension
        
        if time_delay is None:
            time_delay = self.default_time_delay
        
        # Convert to numpy array if needed
        if not isinstance(time_series, np.ndarray):
            time_series = np.array(time_series)
        
        # Remove NaN values
        if np.isnan(time_series).any():
            time_series = time_series[~np.isnan(time_series)]
        
        if len(time_series) < embedding_dimension * time_delay:
            warnings.warn(f"Time series too short for embedding with dimension {embedding_dimension} and delay {time_delay}")
            return np.array([])
        
        # Calculate the number of points in the embedded space
        n_points = len(time_series) - (embedding_dimension - 1) * time_delay
        
        # Create the embedded matrix
        embedded = np.zeros((n_points, embedding_dimension))
        
        for i in range(embedding_dimension):
            embedded[:, i] = time_series[i * time_delay:i * time_delay + n_points]
        
        return embedded
    
    def estimate_optimal_time_delay(self, time_series, max_delay=20, method='mutual_info'):
        # Estimate the optimal time delay for state space reconstruction
        if not isinstance(time_series, np.ndarray):
            time_series = np.array(time_series)
        
        # Remove NaN values
        if np.isnan(time_series).any():
            time_series = time_series[~np.isnan(time_series)]
        
        if len(time_series) < 10:
            return 1  # Default for very short time series
        
        if method == 'autocorr':
            # Use autocorrelation method - first minimum or 1/e threshold
            acf_vals = []
            for delay in range(1, min(max_delay + 1, len(time_series) // 2)):
                acf = np.corrcoef(time_series[:-delay], time_series[delay:])[0, 1]
                acf_vals.append(acf)
            
            # Find first minimum
            for i in range(1, len(acf_vals) - 1):
                if acf_vals[i-1] > acf_vals[i] < acf_vals[i+1]:
                    return i + 1
            
            # If no minimum found, find where ACF drops below 1/e
            e_threshold = 1/np.e
            for i, acf in enumerate(acf_vals):
                if acf < e_threshold:
                    return i + 1
            
            return 1  # Default if no clear minimum
            
        elif method == 'mutual_info':
            try:
                from sklearn.feature_selection import mutual_info_regression
                
                mi_vals = []
                for delay in range(1, min(max_delay + 1, len(time_series) // 2)):
                    x = time_series[:-delay].reshape(-1, 1)
                    y = time_series[delay:]
                    mi = mutual_info_regression(x, y)[0]
                    mi_vals.append(mi)
                
                # Find first minimum of mutual information
                for i in range(1, len(mi_vals) - 1):
                    if mi_vals[i-1] > mi_vals[i] < mi_vals[i+1]:
                        return i + 1
                
                # If no clear minimum, use 1/5 of the length (heuristic)
                return max(1, len(time_series) // 5)
                
            except ImportError:
                warnings.warn("sklearn not available; using autocorrelation method instead")
                return self.estimate_optimal_time_delay(time_series, max_delay, 'autocorr')
        else:
            raise ValueError(f"Unknown method: {method}. Use 'mutual_info' or 'autocorr'")
    
    def estimate_embedding_dimension(self, time_series, time_delay=None, max_dim=None, threshold=0.05):
        # Estimate the optimal embedding dimension using the false nearest neighbors method
        if not isinstance(time_series, np.ndarray):
            time_series = np.array(time_series)
        
        # Remove NaN values
        if np.isnan(time_series).any():
            time_series = time_series[~np.isnan(time_series)]
        
        if len(time_series) < 20:
            return 2, {'false_nn_rates': [], 'message': 'Time series too short'}
        
        # Set defaults
        if time_delay is None:
            time_delay = self.estimate_optimal_time_delay(time_series)
        
        if max_dim is None:
            max_dim = min(self.max_embedding_dimension, len(time_series) // (time_delay * 2))
        
        # Ensure minimum length for reliable calculation
        max_dim = min(max_dim, len(time_series) // (time_delay * 2))
        max_dim = max(2, max_dim)  # Always consider at least 2 dimensions
        
        # Calculate false nearest neighbor rate for each dimension
        fnn_rates = []
        
        for d in range(1, max_dim):
            # Embed in d dimensions
            embedded_d = self.time_delay_embedding(time_series, d, time_delay)
            
            # Embed in d+1 dimensions
            embedded_d_plus_1 = self.time_delay_embedding(time_series, d + 1, time_delay)
            
            # Skip if embedding failed due to insufficient data
            if len(embedded_d) == 0 or len(embedded_d_plus_1) == 0:
                fnn_rates.append(1.0)  # Default to high rate if insufficient data
                continue
            
            # Find nearest neighbors in d dimensions
            n_points = len(embedded_d)
            nn = NearestNeighbors(n_neighbors=2)  # 2 includes the point itself
            nn.fit(embedded_d)
            distances, indices = nn.kneighbors(embedded_d)
            
            # Count false nearest neighbors
            false_nn_count = 0
            valid_points = 0
            
            # Ensure we don't exceed bounds of either embedding
            max_idx = min(len(embedded_d), len(embedded_d_plus_1))
            
            for i in range(max_idx):
                # Skip if nearest neighbor is the point itself
                if distances[i, 1] == 0:
                    continue
                
                # Get the nearest neighbor index
                nn_idx = indices[i, 1]
                
                # Skip if the neighbor index is out of bounds for d+1 embedding
                if nn_idx >= len(embedded_d_plus_1):
                    continue
                
                # Calculate distance in d dimension
                distance_d = distances[i, 1]
                
                # Calculate distance in d+1 dimension
                try:
                    # We no longer need to slice the vectors since their dimensions already match
                    distance_d_plus_1 = np.linalg.norm(
                        embedded_d_plus_1[i] - embedded_d_plus_1[nn_idx]
                    )
                    
                    # Check if it's a false nearest neighbor
                    if distance_d > 0:  # Avoid division by zero
                        relative_increase = (distance_d_plus_1 - distance_d) / distance_d
                        valid_points += 1
                        if relative_increase > 2.0:  # Typical threshold
                            false_nn_count += 1
                except:
                    # Skip any points that cause errors
                    continue
            
            # Calculate rate if we have valid points
            if valid_points > 0:
                false_nn_rate = false_nn_count / valid_points
            else:
                false_nn_rate = 1.0  # Default to high rate if no valid comparisons
                
            fnn_rates.append(false_nn_rate)
        
        # Find the first dimension where FNN rate drops below threshold
        for i, rate in enumerate(fnn_rates):
            if rate < threshold:
                return i + 1, {'false_nn_rates': fnn_rates}
        
        # If no dimension falls below threshold, return the dimension with minimum rate
        if fnn_rates:
            best_dim = np.argmin(fnn_rates) + 1
            return best_dim, {'false_nn_rates': fnn_rates, 'message': 'No dimension below threshold'}
        else:
            # If we couldn't calculate any rates, return default dimension
            return 3, {'false_nn_rates': [], 'message': 'Could not calculate FNN rates'}
    
    # =========================================================================
    # Stability Analysis Methods
    # =========================================================================
    
    def calculate_lyapunov_exponent(self, time_series, embedding_dimension=None, time_delay=None, k=None):
        # Calculate the maximal Lyapunov exponent to quantify the chaos/stability of a system
        if not isinstance(time_series, np.ndarray):
            time_series = np.array(time_series)
        
        # Remove NaN values
        if np.isnan(time_series).any():
            time_series = time_series[~np.isnan(time_series)]
        
        # Set defaults
        if embedding_dimension is None:
            embedding_dimension = self.default_embedding_dimension
        
        if time_delay is None:
            time_delay = self.estimate_optimal_time_delay(time_series)
        
        if k is None:
            k = self.lyapunov_k
        
        # Ensure minimum length for reliable calculation
        if len(time_series) < embedding_dimension * time_delay * 2:
            return np.nan, {'error': 'Time series too short for Lyapunov calculation'}
        
        # Perform time-delay embedding
        embedded = self.time_delay_embedding(time_series, embedding_dimension, time_delay)
        
        # Ensure sufficient data
        if len(embedded) < k + 1:
            return np.nan, {'error': 'Insufficient points after embedding'}
        
        # Find nearest neighbors for each point in state space
        nn = NearestNeighbors(n_neighbors=k+1)  # +1 because the point itself is included
        nn.fit(embedded)
        distances, indices = nn.kneighbors(embedded)
        
        # Exclude the first few and last few points for reliable prediction
        valid_range = range(5, len(embedded) - self.lyapunov_max_steps)
        
        # Calculate divergence rates
        divergence_rates = []
        
        for i in valid_range:
            # Get k nearest neighbors (excluding the point itself)
            neighbors = indices[i, 1:k+1]
            
            # Calculate mean initial distance
            initial_distance = distances[i, 1:k+1].mean()
            
            if initial_distance == 0:
                continue  # Skip if initial distance is zero
            
            # Calculate mean distances after j steps
            future_distances = []
            
            for j in range(1, self.lyapunov_max_steps + 1):
                if i + j >= len(embedded) or any(n + j >= len(embedded) for n in neighbors):
                    break
                
                # Distance between the evolved point and evolved neighbors
                distances_j = [
                    np.linalg.norm(embedded[i + j] - embedded[n + j])
                    for n in neighbors if n + j < len(embedded)
                ]
                
                if not distances_j:
                    break
                
                mean_distance_j = np.mean(distances_j)
                future_distances.append(mean_distance_j)
            
            if not future_distances:
                continue
            
            # Calculate logarithms of distance ratios
            log_ratios = [np.log(d / initial_distance) for d in future_distances]
            
            # Linear fit to estimate divergence rate (Lyapunov exponent)
            steps = np.arange(1, len(log_ratios) + 1)
            if len(steps) > 1:
                slope, _, _, _, _ = stats.linregress(steps, log_ratios)
                divergence_rates.append(slope)
        
        # Calculate the maximal Lyapunov exponent as the mean of individual rates
        if not divergence_rates:
            return np.nan, {'error': 'Could not calculate divergence rates'}
        
        lyapunov_exponent = np.mean(divergence_rates)
        
        return lyapunov_exponent, {'n_rates': len(divergence_rates), 'std_dev': np.std(divergence_rates)}
    
    def calculate_recurrence_plot(self, time_series, embedding_dimension=None, time_delay=None, 
                               threshold=None, norm='euclidean'):
        # Calculate a recurrence plot for the time series
        if not isinstance(time_series, np.ndarray):
            time_series = np.array(time_series)
        
        # Remove NaN values
        if np.isnan(time_series).any():
            time_series = time_series[~np.isnan(time_series)]
        
        # Set defaults
        if embedding_dimension is None:
            embedding_dimension = self.default_embedding_dimension
        
        if time_delay is None:
            time_delay = self.estimate_optimal_time_delay(time_series)
        
        # Perform time-delay embedding
        embedded = self.time_delay_embedding(time_series, embedding_dimension, time_delay)
        
        # Check if embedding was successful
        if len(embedded) == 0:
            return np.array([]), {'error': 'Embedding failed, insufficient data'}
        
        # Calculate distance matrix
        n_points = len(embedded)
        dist_matrix = np.zeros((n_points, n_points))
        
        # Calculate distances using the specified norm
        for i in range(n_points):
            for j in range(i, n_points):
                if norm == 'euclidean':
                    dist = np.linalg.norm(embedded[i] - embedded[j])
                elif norm == 'manhattan':
                    dist = np.sum(np.abs(embedded[i] - embedded[j]))
                elif norm == 'chebyshev':
                    dist = np.max(np.abs(embedded[i] - embedded[j]))
                else:
                    raise ValueError(f"Unknown norm: {norm}")
                
                dist_matrix[i, j] = dist_matrix[j, i] = dist
        
        # Set threshold if not provided
        if threshold is None:
            threshold = 0.1 * np.max(dist_matrix)
        
        # Create recurrence matrix
        recurrence_matrix = (dist_matrix < threshold).astype(int)
        
        return recurrence_matrix, {
            'threshold': threshold,
            'embedding_dimension': embedding_dimension,
            'time_delay': time_delay,
            'norm': norm
        }
    
    def calculate_recurrence_quantification(self, recurrence_matrix):
        # Calculate recurrence quantification measures from a recurrence matrix
        if len(recurrence_matrix) == 0:
            return {
                'recurrence_rate': np.nan,
                'determinism': np.nan,
                'laminarity': np.nan,
                'average_diagonal_line': np.nan,
                'entropy_diagonal_lines': np.nan
            }
        
        n_points = len(recurrence_matrix)
        
        # Recurrence Rate (RR) - ratio of recurrence points to total points
        recurrence_rate = np.sum(recurrence_matrix) / (n_points**2)
        
        # Calculate diagonal line lengths
        # A diagonal line is a sequence of recurrent points parallel to the main diagonal
        diagonal_lines = []
        for i in range(-(n_points-1), n_points):
            diag = np.diagonal(recurrence_matrix, offset=i)
            # Extract lengths of consecutive 1's
            line_lengths = []
            current_length = 0
            for point in diag:
                if point == 1:
                    current_length += 1
                else:
                    if current_length > 0:
                        line_lengths.append(current_length)
                        current_length = 0
            if current_length > 0:
                line_lengths.append(current_length)
            diagonal_lines.extend([l for l in line_lengths if l >= 2])  # Only lines of length >= 2
        
        # Calculate vertical line lengths (for laminarity)
        vertical_lines = []
        for i in range(n_points):
            col = recurrence_matrix[:, i]
            line_lengths = []
            current_length = 0
            for point in col:
                if point == 1:
                    current_length += 1
                else:
                    if current_length > 0:
                        line_lengths.append(current_length)
                        current_length = 0
            if current_length > 0:
                line_lengths.append(current_length)
            vertical_lines.extend([l for l in line_lengths if l >= 2])  # Only lines of length >= 2
        
        # Determinism (DET) - ratio of recurrence points forming diagonal lines to all recurrence points
        determinism = np.sum(diagonal_lines) / max(1, np.sum(recurrence_matrix))
        
        # Laminarity (LAM) - ratio of recurrence points forming vertical lines to all recurrence points
        laminarity = np.sum(vertical_lines) / max(1, np.sum(recurrence_matrix))
        
        # Average diagonal line length
        average_diagonal_line = np.mean(diagonal_lines) if diagonal_lines else np.nan
        
        # Entropy of diagonal line lengths
        if diagonal_lines:
            # Calculate histogram of line lengths
            unique_lengths, counts = np.unique(diagonal_lines, return_counts=True)
            probs = counts / np.sum(counts)
            entropy_diagonal_lines = -np.sum(probs * np.log(probs))
        else:
            entropy_diagonal_lines = np.nan
        
        return {
            'recurrence_rate': recurrence_rate,
            'determinism': determinism,
            'laminarity': laminarity,
            'average_diagonal_line': average_diagonal_line,
            'entropy_diagonal_lines': entropy_diagonal_lines,
            'diagonal_line_lengths': diagonal_lines,
            'vertical_line_lengths': vertical_lines
        }
    
    def detect_fixed_points(self, time_series, embedding_dimension=None, time_delay=None,
                        threshold=0.1, min_duration=5):
        # Detect potential fixed points and limit cycles in the time series
        if not isinstance(time_series, np.ndarray):
            time_series = np.array(time_series)
        
        # Remove NaN values
        if np.isnan(time_series).any():
            time_series = time_series[~np.isnan(time_series)]
        
        # Set defaults
        if embedding_dimension is None:
            embedding_dimension = self.default_embedding_dimension
        
        if time_delay is None:
            time_delay = self.estimate_optimal_time_delay(time_series)
        
        # Perform time-delay embedding
        embedded = self.time_delay_embedding(time_series, embedding_dimension, time_delay)
        
        # Check if embedding was successful
        if len(embedded) == 0:
            return {'fixed_points': [], 'limit_cycles': [], 'error': 'Embedding failed'}
        
        # Calculate the first difference of the embedded time series
        velocity = np.zeros_like(embedded)
        velocity[:-1] = embedded[1:] - embedded[:-1]
        velocity[-1] = velocity[-2]  # use the second-to-last velocity for the last point
        
        # Calculate the speed (magnitude of velocity)
        speed = np.sqrt(np.sum(velocity**2, axis=1))
        
        # Normalize the speed
        if np.max(speed) > 0:
            normalized_speed = speed / np.max(speed)
        else:
            normalized_speed = speed
        
        # Identify potential fixed points where the normalized speed is below threshold
        potential_fixed_points = np.where(normalized_speed < threshold)[0]
        
        # Group consecutive points
        fixed_point_segments = []
        current_segment = []
        
        for i in range(len(potential_fixed_points)):
            idx = potential_fixed_points[i]
            
            if i == 0 or idx == potential_fixed_points[i-1] + 1:
                # Consecutive point, add to current segment
                current_segment.append(idx)
            else:
                # Start of new segment
                if len(current_segment) >= min_duration:
                    fixed_point_segments.append(current_segment)
                current_segment = [idx]
        
        # Add the last segment if it's long enough
        if len(current_segment) >= min_duration:
            fixed_point_segments.append(current_segment)
        
        # Extract fixed points (take the mean of each segment)
        fixed_points = []
        for segment in fixed_point_segments:
            segment_points = embedded[segment]
            fixed_point = {
                'value': np.mean(segment_points, axis=0),
                'start_idx': segment[0],
                'end_idx': segment[-1],
                'duration': len(segment),
                'point_values': segment_points
            }
            fixed_points.append(fixed_point)
        
        # Detect potential limit cycles - look for repeating patterns in the time series
        # Simplify by using autocorrelation to detect periodicity
        limit_cycles = []
        if len(time_series) > 20:
            # Calculate autocorrelation
            nlags = min(len(time_series) // 2, 100)
            acf = np.array([1.0] + [np.corrcoef(time_series[:-lag], time_series[lag:])[0, 1] 
                                   for lag in range(1, nlags)])
            
            # Find peaks in the autocorrelation, which indicate potential cycles
            peaks, _ = signal.find_peaks(acf, height=0.3, distance=3)
            
            if len(peaks) > 0:
                # Estimate cycle period from the first prominent peak
                cycle_period = peaks[0] if peaks.size > 0 else None
                
                # Look for segments where the speed oscillates with this period
                if cycle_period is not None and cycle_period < len(time_series) // 2:
                    # Find segments where speed oscillates with low variance around the trend
                    cycle_segments = []
                    for i in range(len(time_series) - cycle_period * min_duration):
                        segment = time_series[i:i + cycle_period * min_duration]
                        # Detrend the segment
                        detrended = signal.detrend(segment)
                        # Check variance
                        if np.var(detrended) < threshold * np.var(time_series):
                            cycle_segments.append((i, i + cycle_period * min_duration))
                    
                    # Merge overlapping segments
                    if cycle_segments:
                        merged_segments = [cycle_segments[0]]
                        for current in cycle_segments[1:]:
                            prev = merged_segments[-1]
                            if current[0] <= prev[1]:
                                # Merge overlapping segments
                                merged_segments[-1] = (prev[0], max(prev[1], current[1]))
                            else:
                                merged_segments.append(current)
                        
                        # Create limit cycle objects
                        for start, end in merged_segments:
                            limit_cycle = {
                                'period': cycle_period,
                                'start_idx': start,
                                'end_idx': end,
                                'duration': end - start,
                                'values': time_series[start:end]
                            }
                            limit_cycles.append(limit_cycle)
        
        return {
            'fixed_points': fixed_points,
            'limit_cycles': limit_cycles,
            'normalized_speed': normalized_speed,
            'embedding_dimension': embedding_dimension,
            'time_delay': time_delay
        }
    
    # =========================================================================
    # Physiological Models
    # =========================================================================
    
    def cardiovascular_model(self, time_points, initial_state, parameters=None):
        # Simple cardiovascular system model based on feedback control
        if parameters is None:
            parameters = self.system_parameters['cardiovascular']
        
        # Extract parameters
        heart_rate_baseline = parameters['heart_rate_baseline']
        contractility_factor = parameters['contractility_factor']
        vascular_resistance = parameters['vascular_resistance']
        baroreceptor_gain = parameters['baroreceptor_gain']
        baroreceptor_delay = parameters['baroreceptor_delay']
        
        # Define the system of differential equations
        def system(t, y):
            heart_rate, mean_arterial_pressure = y
            
            # Calculate the feedback from baroreceptors (with delay if possible)
            # In a real system this would use past values, but we simplify
            pressure_error = mean_arterial_pressure - 93  # 93 mmHg is "normal"
            
            # Heart rate change based on baroreceptor feedback
            # Higher pressure -> lower heart rate (negative feedback)
            heart_rate_change = (heart_rate_baseline - heart_rate - 
                               baroreceptor_gain * pressure_error) / 2.0
            
            # Blood pressure change based on heart rate and other factors
            pressure_change = (contractility_factor * heart_rate - 
                              vascular_resistance * mean_arterial_pressure) / 10.0
            
            return [heart_rate_change, pressure_change]
        
        # Solve the system
        solution = integrate.solve_ivp(
            system, 
            [time_points[0], time_points[-1]], 
            initial_state, 
            t_eval=time_points,
            method='RK45'
        )
        
        return solution.y.T  # Return as [time_points, state_variables]
    
    def respiratory_model(self, time_points, initial_state, parameters=None):
        # Simple respiratory system model based on feedback control
        if parameters is None:
            parameters = self.system_parameters['respiratory']
        
        # Extract parameters
        respiratory_rate_baseline = parameters['respiratory_rate_baseline']
        tidal_volume_baseline = parameters['tidal_volume_baseline']
        chemoreceptor_gain_co2 = parameters['chemoreceptor_gain_co2']
        chemoreceptor_gain_o2 = parameters['chemoreceptor_gain_o2']
        respiratory_drive = parameters['respiratory_drive']
        
        # Define the system of differential equations
        def system(t, y):
            respiratory_rate, tidal_volume, pco2 = y
            
            # Calculate the total minute ventilation
            minute_ventilation = respiratory_rate * tidal_volume / 1000  # L/min
            
            # CO2 changes based on ventilation (CO2 removal) and production
            co2_production = 0.2  # L/min (roughly constant)
            co2_removal = 0.8 * minute_ventilation  # Depends on ventilation
            pco2_change = 5 * (co2_production - co2_removal)  # Scale factor for reasonable changes
            
            # Respiratory drive based on CO2 (simplified model)
            # Higher CO2 -> higher respiratory drive (positive feedback)
            co2_error = pco2 - 40  # 40 mmHg is "normal"
            drive = respiratory_drive + chemoreceptor_gain_co2 * co2_error
            
            # Respiratory rate and tidal volume change based on drive
            rate_change = (respiratory_rate_baseline * (1 + 0.2 * drive) - respiratory_rate) / 2.0
            volume_change = (tidal_volume_baseline * (1 + 0.1 * drive) - tidal_volume) / 5.0
            
            return [rate_change, volume_change, pco2_change]
        
        # Solve the system
        solution = integrate.solve_ivp(
            system, 
            [time_points[0], time_points[-1]], 
            initial_state, 
            t_eval=time_points,
            method='RK45'
        )
        
        return solution.y.T  # Return as [time_points, state_variables]
    
    def simulate_physiological_system(self, system_name, duration_seconds=300, initial_state=None):
        # Simulate a physiological system using the appropriate model
        # Generate time points
        time_points = np.linspace(0, duration_seconds, num=int(duration_seconds))
        
        if system_name == 'cardiovascular':
            # Default initial state: [heart_rate, mean_arterial_pressure]
            if initial_state is None:
                initial_state = [70, 90]  # 70 bpm, 90 mmHg
            
            # Simulate the cardiovascular system
            states = self.cardiovascular_model(time_points, initial_state)
            return time_points, states
            
        elif system_name == 'respiratory':
            # Default initial state: [respiratory_rate, tidal_volume, pCO2]
            if initial_state is None:
                initial_state = [12, 500, 40]  # 12 breaths/min, 500 mL, 40 mmHg
            
            # Simulate the respiratory system
            states = self.respiratory_model(time_points, initial_state)
            return time_points, states
            
        else:
            raise ValueError(f"Unknown system: {system_name}")
    
    # =========================================================================
    # Critical Transition Detection
    # =========================================================================
    
    def calculate_early_warning_signals(self, time_series, window_size=None, step_size=None):
        # Calculate early warning signals for critical transitions
        if not isinstance(time_series, np.ndarray):
            time_series = np.array(time_series)
        
        # Remove NaN values
        if np.isnan(time_series).any():
            time_series = time_series[~np.isnan(time_series)]
        
        # Set defaults
        if window_size is None:
            window_size = self.ews_window_size
        
        if step_size is None:
            step_size = self.ews_step_size
        
        # Ensure the time series is long enough
        if len(time_series) < window_size * 2:
            return {
                'variance': np.array([]),
                'autocorrelation': np.array([]),
                'skewness': np.array([]),
                'kurtosis': np.array([]),
                'window_indices': np.array([]),
                'error': 'Time series too short for EWS calculation'
            }
        
        # Detrend the data to remove long-term trends
        try:
            detrended = signal.detrend(time_series)
        except:
            detrended = time_series
        
        # Calculate sliding window indices
        n_windows = (len(detrended) - window_size) // step_size + 1
        window_indices = np.array([i * step_size + window_size // 2 for i in range(n_windows)])
        
        # Initialize arrays for EWS metrics
        variance = np.zeros(n_windows)
        autocorrelation = np.zeros(n_windows)
        skewness = np.zeros(n_windows)
        kurtosis = np.zeros(n_windows)
        
        # Calculate metrics for each window
        for i in range(n_windows):
            start_idx = i * step_size
            end_idx = start_idx + window_size
            window_data = detrended[start_idx:end_idx]
            
            # Variance (increases near tipping points)
            variance[i] = np.var(window_data)
            
            # Lag-1 autocorrelation (increases near tipping points)
            if len(window_data) > 1:
                autocorrelation[i] = np.corrcoef(window_data[:-1], window_data[1:])[0, 1]
            else:
                autocorrelation[i] = np.nan
            
            # Skewness (changes near tipping points)
            skewness[i] = stats.skew(window_data)
            
            # Kurtosis (changes near tipping points)
            kurtosis[i] = stats.kurtosis(window_data)
        
        return {
            'variance': variance,
            'autocorrelation': autocorrelation,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'window_indices': window_indices,
            'window_size': window_size,
            'step_size': step_size
        }
    
    def detect_critical_transition(self, time_series, window_size=None, step_size=None, 
                                  threshold=2.0):
        # Detect potential critical transitions in a time series
        # Calculate early warning signals
        ews = self.calculate_early_warning_signals(time_series, window_size, step_size)
        
        # Check if EWS calculation was successful
        if 'error' in ews:
            return {'detected': False, 'probability': 0.0, 'error': ews['error']}
        
        # Calculate z-scores for each metric
        variance_z = self._calculate_trend_strength(ews['variance'])
        autocorr_z = self._calculate_trend_strength(ews['autocorrelation'])
        skewness_z = self._calculate_trend_strength(np.abs(ews['skewness']))
        
        # Create a combined indicator (weighted average of z-scores)
        combined_z = (0.4 * autocorr_z + 0.4 * variance_z + 0.2 * skewness_z)
        
        # Determine whether a critical transition is approaching
        transition_detected = combined_z > threshold
        
        # Calculate a probability score (sigmoid function of combined_z)
        transition_probability = 1.0 / (1.0 + np.exp(-combined_z + threshold))
        
        # Identify the window where the combined indicator peaks
        if len(ews['window_indices']) > 0:
            max_idx = np.nanargmax(combined_z) if not np.all(np.isnan(combined_z)) else 0
            transition_point = ews['window_indices'][max_idx] if max_idx < len(ews['window_indices']) else None
        else:
            transition_point = None
        
        return {
            'detected': transition_detected,
            'probability': float(transition_probability),
            'transition_point': transition_point,
            'combined_indicator': combined_z,
            'variance_indicator': variance_z,
            'autocorrelation_indicator': autocorr_z,
            'skewness_indicator': skewness_z,
            'window_indices': ews['window_indices']
        }
    
    def _calculate_trend_strength(self, values):
        # Calculate the strength of an upward trend in a time series
        if len(values) < 3 or np.all(np.isnan(values)):
            return 0.0
        
        # Remove NaNs
        values = values[~np.isnan(values)]
        
        if len(values) < 3:
            return 0.0
        
        # Detrend to remove first-order trend
        try:
            detrended = signal.detrend(values)
            trend = values - detrended
            
            # Calculate the slope of the trend line
            x = np.arange(len(values))
            slope, _, r_value, p_value, _ = stats.linregress(x, values)
            
            # Normalize into a z-score-like value
            # Positive values indicate upward trend, negative for downward
            trend_z = slope * np.sqrt(len(values)) / np.std(values) if np.std(values) > 0 else 0
            
            # Weight by r² and significance
            significance = max(0, 1 - p_value)
            weighted_z = trend_z * r_value**2 * significance
            
            return weighted_z
        except:
            return 0.0
    
    # =========================================================================
    # Integration with Signal Processing
    # =========================================================================
    
    def analyze_patient_stability(self, patient_timeline, vital_sign='heart_rate'):
        # Analyze the stability of a patient's vital sign using dynamical systems approaches
        if not patient_timeline or 'timeline' not in patient_timeline:
            return {'error': 'Invalid patient timeline format'}
        
        timeline = patient_timeline['timeline']
        if timeline is None or len(timeline) == 0:
            return {'error': 'Empty patient timeline'}
        
        # Extract the vital sign time series
        vital_series = None
        
        # Check if the vital sign is directly available as a column
        if vital_sign in timeline.columns:
            vital_series = timeline[vital_sign].dropna().values
            
        # Otherwise, try to extract from MIMIC itemid mapping
        elif 'itemid' in timeline.columns and 'valuenum' in timeline.columns:
            # Look for the vital sign in the signal processor's itemid mapping
            # This part would need adaptation based on your actual implementation
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
                vital_data = timeline[timeline['itemid'].isin(vital_itemids)]
                if len(vital_data) > 0:
                    vital_series = vital_data['valuenum'].values
        
        if vital_series is None or len(vital_series) < 10:
            return {'error': f'Insufficient data for {vital_sign}'}
        
        # Perform analysis
        results = {
            'vital_sign': vital_sign,
            'n_measurements': len(vital_series),
            'embedding': {},
            'stability': {},
            'critical_transitions': {},
        }
        
        # 1. Determine optimal embedding parameters
        time_delay = self.estimate_optimal_time_delay(vital_series)
        embedding_dim, embed_info = self.estimate_embedding_dimension(vital_series, time_delay)
        
        results['embedding'] = {
            'optimal_time_delay': time_delay,
            'optimal_dimension': embedding_dim,
            'embedding_info': embed_info
        }
        
        # 2. Calculate stability metrics
        lyapunov, lyap_info = self.calculate_lyapunov_exponent(vital_series, embedding_dim, time_delay)
        
        # Calculate recurrence plot and quantification
        recurrence_matrix, recur_info = self.calculate_recurrence_plot(
            vital_series, embedding_dim, time_delay
        )
        
        if len(recurrence_matrix) > 0:
            recurrence_metrics = self.calculate_recurrence_quantification(recurrence_matrix)
        else:
            recurrence_metrics = {'error': 'Failed to calculate recurrence metrics'}
        
        # Find fixed points and limit cycles
        attractor_info = self.detect_fixed_points(vital_series, embedding_dim, time_delay)
        
        results['stability'] = {
            'maximal_lyapunov_exponent': lyapunov,
            'lyapunov_info': lyap_info,
            'recurrence_metrics': recurrence_metrics,
            'fixed_points': attractor_info['fixed_points'],
            'limit_cycles': attractor_info['limit_cycles']
        }
        
        # 3. Detect critical transitions
        transition_detection = self.detect_critical_transition(vital_series)
        
        results['critical_transitions'] = transition_detection
        
        # 4. Interpret the results
        # Classify the system state based on the analysis
        
        # Stable fixed point: negative Lyapunov exponent, fixed points detected
        # Limit cycle: Lyapunov exponent near zero, limit cycles detected
        # Chaotic: positive Lyapunov exponent, no clear fixed points or limit cycles
        # Critical transition: early warning signals detected
        
        if np.isnan(lyapunov):
            system_state = "Undetermined (insufficient data)"
        elif lyapunov < -0.05 and len(attractor_info['fixed_points']) > 0:
            system_state = "Stable Fixed Point"
        elif abs(lyapunov) < 0.05 and len(attractor_info['limit_cycles']) > 0:
            system_state = "Limit Cycle (periodic behavior)"
        elif lyapunov > 0.05:
            system_state = "Chaotic Dynamics"
        else:
            system_state = "Complex Dynamics"
        
        # Add transition warning if detected
        if transition_detection['detected']:
            system_state += " - Approaching Critical Transition"
        
        results['interpretation'] = {
            'system_state': system_state,
            'stability_assessment': "Stable" if lyapunov < 0 else "Unstable",
            'transition_warning': transition_detection['detected'],
            'transition_probability': transition_detection['probability']
        }
        
        return results
    
    def create_stability_report(self, patient_timeline):
        # Create a comprehensive stability report for a patient
        if not patient_timeline or 'timeline' not in patient_timeline:
            return {'error': 'Invalid patient timeline format'}
        
        timeline = patient_timeline['timeline']
        if timeline is None or len(timeline) == 0:
            return {'error': 'Empty patient timeline'}
        
        # List of vital signs to analyze
        vital_signs = [
            'heart_rate', 'respiratory_rate', 'sbp', 'dbp', 'temperature'
        ]
        
        # Initialize report
        report = {
            'patient_id': patient_timeline.get('info', {}).get('subject_id'),
            'vital_sign_analysis': {},
            'system_stability': {},
            'critical_transitions': {},
            'overall_assessment': {}
        }
        
        # Analyze each vital sign
        for vital in vital_signs:
            analysis = self.analyze_patient_stability(patient_timeline, vital)
            if 'error' not in analysis:
                report['vital_sign_analysis'][vital] = analysis
        
        # Aggregate stability measures across vital signs
        if report['vital_sign_analysis']:
            # Calculate average Lyapunov exponent
            lyapunov_values = [
                analysis['stability']['maximal_lyapunov_exponent']
                for analysis in report['vital_sign_analysis'].values()
                if not np.isnan(analysis['stability']['maximal_lyapunov_exponent'])
            ]
            
            avg_lyapunov = np.mean(lyapunov_values) if lyapunov_values else np.nan
            
            # Detect systems with critical transitions
            systems_with_transitions = [
                vital for vital, analysis in report['vital_sign_analysis'].items()
                if analysis['critical_transitions']['detected']
            ]
            
            # Calculate maximum transition probability
            max_transition_prob = max(
                [analysis['critical_transitions']['probability'] 
                 for analysis in report['vital_sign_analysis'].values()],
                default=0.0
            )
            
            # Map vital signs to physiological systems
            system_mapping = {
                'heart_rate': 'cardiovascular',
                'sbp': 'cardiovascular',
                'dbp': 'cardiovascular',
                'respiratory_rate': 'respiratory',
                'temperature': 'thermoregulatory'
            }
            
            # Aggregate by system
            system_stability = {}
            for vital, analysis in report['vital_sign_analysis'].items():
                system = system_mapping.get(vital, 'other')
                if system not in system_stability:
                    system_stability[system] = {
                        'vital_signs': [],
                        'lyapunov_exponents': [],
                        'has_critical_transition': False,
                        'stability_score': 0.0
                    }
                
                system_stability[system]['vital_signs'].append(vital)
                system_stability[system]['lyapunov_exponents'].append(
                    analysis['stability']['maximal_lyapunov_exponent']
                )
                
                if analysis['critical_transitions']['detected']:
                    system_stability[system]['has_critical_transition'] = True
            
            # Calculate stability score for each system
            for system in system_stability:
                lyapunov_values = [x for x in system_stability[system]['lyapunov_exponents'] 
                                 if not np.isnan(x)]
                
                if lyapunov_values:
                    # Transform Lyapunov exponents to stability score
                    # Negative Lyapunov -> high stability, Positive -> low stability
                    # Map from typical range [-0.5, 0.5] to [1, 0]
                    avg_lyap = np.mean(lyapunov_values)
                    stability_score = max(0.0, min(1.0, 0.5 - avg_lyap))
                    system_stability[system]['stability_score'] = stability_score
                    
                    # Classify system stability
                    if stability_score > 0.8:
                        system_stability[system]['classification'] = "Highly Stable"
                    elif stability_score > 0.6:
                        system_stability[system]['classification'] = "Stable"
                    elif stability_score > 0.4:
                        system_stability[system]['classification'] = "Moderately Stable"
                    elif stability_score > 0.2:
                        system_stability[system]['classification'] = "Somewhat Unstable"
                    else:
                        system_stability[system]['classification'] = "Unstable"
                    
                    # Adjust classification if critical transition detected
                    if system_stability[system]['has_critical_transition']:
                        system_stability[system]['classification'] += " - Approaching Transition"
            
            report['system_stability'] = system_stability
            
            # Overall assessment
            if avg_lyapunov is not np.nan:
                if avg_lyapunov < -0.1:
                    stability_class = "Highly Stable Physiological State"
                elif avg_lyapunov < 0:
                    stability_class = "Stable Physiological State"
                elif avg_lyapunov < 0.1:
                    stability_class = "Marginally Stable Physiological State"
                else:
                    stability_class = "Unstable Physiological State"
                
                if systems_with_transitions:
                    stability_class += " with Critical Transitions"
                
                report['overall_assessment'] = {
                    'average_lyapunov_exponent': avg_lyapunov,
                    'stability_classification': stability_class,
                    'systems_with_transitions': systems_with_transitions,
                    'maximum_transition_probability': max_transition_prob
                }
            else:
                report['overall_assessment'] = {
                    'error': 'Insufficient data for overall stability assessment'
                }
        
        return report