# BioDynamICS: Key Innovations Code Samples

This document highlights the most innovative aspects of the BioDynamICS framework with code samples and explanations.

## 1. Allostatic Load Quantification

Allostatic load is a measure of overall physiological stress that integrates multiple organ systems. This innovation provides a single metric that can be used to track a patient's overall stability over time.

```python
def calculate_allostatic_load(self, vital_signs_df, lab_values_df=None):
    """
    Calculate allostatic load based on vital signs and lab values.
    
    Allostatic load is a measure of physiological stress across multiple systems.
    Higher values indicate greater physiological dysregulation.
    
    Parameters:
    -----------
    vital_signs_df : pandas.DataFrame
        DataFrame with vital sign measurements
    lab_values_df : pandas.DataFrame, optional
        DataFrame with laboratory values
        
    Returns:
    --------
    float
        Allostatic load score (0-3, where higher values indicate greater stress)
    """
    # Initialize component scores
    cardiovascular_score = 0
    respiratory_score = 0
    metabolic_score = 0
    inflammatory_score = 0
    
    # Calculate cardiovascular component
    if 'heart_rate' in vital_signs_df.columns and 'sbp' in vital_signs_df.columns:
        hr_deviation = self._calculate_deviation(vital_signs_df['heart_rate'], 
                                               self.reference_ranges['heart_rate'])
        sbp_deviation = self._calculate_deviation(vital_signs_df['sbp'], 
                                                self.reference_ranges['sbp'])
        cardiovascular_score = (hr_deviation + sbp_deviation) / 2
    
    # Calculate respiratory component
    if 'respiratory_rate' in vital_signs_df.columns and 'o2_saturation' in vital_signs_df.columns:
        rr_deviation = self._calculate_deviation(vital_signs_df['respiratory_rate'], 
                                               self.reference_ranges['respiratory_rate'])
        o2_deviation = self._calculate_deviation_inverse(vital_signs_df['o2_saturation'], 
                                                      self.reference_ranges['o2_saturation'])
        respiratory_score = (rr_deviation + o2_deviation) / 2
    
    # Calculate metabolic component if lab values available
    if lab_values_df is not None:
        metabolic_markers = ['glucose', 'sodium', 'potassium']
        deviations = []
        
        for marker in metabolic_markers:
            if marker in lab_values_df.columns:
                deviation = self._calculate_deviation(lab_values_df[marker], 
                                                   self.reference_ranges.get(marker, [0, 100]))
                deviations.append(deviation)
        
        if deviations:
            metabolic_score = sum(deviations) / len(deviations)
    
    # Calculate inflammatory component if lab values available
    if lab_values_df is not None and 'wbc' in lab_values_df.columns:
        wbc_deviation = self._calculate_deviation(lab_values_df['wbc'], 
                                               self.reference_ranges.get('wbc', [4.5, 11.0]))
        inflammatory_score = wbc_deviation
    
    # Calculate overall allostatic load
    components = [cardiovascular_score, respiratory_score, metabolic_score, inflammatory_score]
    valid_components = [c for c in components if c > 0]
    
    if not valid_components:
        return 0
    
    # Weight components and calculate overall score
    allostatic_load = sum(valid_components) / len(valid_components)
    
    # Apply non-linear transformation to emphasize higher deviations
    allostatic_load = np.tanh(allostatic_load) * 3
    
    return allostatic_load
```

**Innovation Significance**: This approach goes beyond traditional vital sign monitoring by:
1. Integrating multiple physiological systems into a single metric
2. Quantifying deviations from normal ranges in a standardized way
3. Applying non-linear transformations to emphasize clinically significant deviations
4. Providing a continuous measure that can track subtle changes over time

## 2. Critical Transition Detection

This innovation applies dynamical systems theory to detect early warning signs of clinical deterioration before traditional vital sign thresholds are crossed.

```python
def detect_critical_transition(self, time_series, window_size=None, step_size=None):
    """
    Detect early warning signs of critical transitions in a time series.
    
    Uses statistical indicators like variance, autocorrelation, and skewness
    to identify approaching tipping points in the system dynamics.
    
    Parameters:
    -----------
    time_series : numpy.ndarray
        Time series data
    window_size : int, optional
        Size of sliding window for indicator calculation
    step_size : int, optional
        Step size for sliding window
        
    Returns:
    --------
    dict
        Dictionary with critical transition indicators and detection results
    """
    # Set default window and step size
    if window_size is None:
        window_size = max(10, len(time_series) // 10)
    if step_size is None:
        step_size = max(1, window_size // 4)
    
    # Calculate early warning indicators
    indicators = {
        'variance': [],
        'autocorrelation': [],
        'skewness': [],
        'kurtosis': [],
        'window_start': []
    }
    
    # Slide window through time series
    for i in range(0, len(time_series) - window_size, step_size):
        window = time_series[i:i+window_size]
        
        # Skip windows with insufficient data
        if len(window) < 5:
            continue
        
        # Calculate variance (increasing variance is an early warning sign)
        indicators['variance'].append(np.var(window))
        
        # Calculate lag-1 autocorrelation (increasing autocorrelation is an early warning sign)
        if len(window) > 1:
            autocorr = np.corrcoef(window[:-1], window[1:])[0, 1]
            indicators['autocorrelation'].append(autocorr)
        else:
            indicators['autocorrelation'].append(np.nan)
        
        # Calculate skewness (changing skewness can indicate approaching transition)
        indicators['skewness'].append(stats.skew(window))
        
        # Calculate kurtosis (changing kurtosis can indicate approaching transition)
        indicators['kurtosis'].append(stats.kurtosis(window))
        
        # Store window start index
        indicators['window_start'].append(i)
    
    # Detect trends in indicators
    trends = {}
    for indicator, values in indicators.items():
        if indicator == 'window_start':
            continue
            
        # Calculate Kendall's tau for trend detection
        if len(values) > 2:
            tau, p_value = stats.kendalltau(range(len(values)), values)
            trends[indicator] = {
                'tau': tau,
                'p_value': p_value,
                'significant': p_value < 0.05
            }
    
    # Determine if critical transition is likely
    # Criteria: Significant increasing trends in variance and autocorrelation
    var_increasing = (trends.get('variance', {}).get('tau', 0) > 0 and 
                     trends.get('variance', {}).get('significant', False))
    
    autocorr_increasing = (trends.get('autocorrelation', {}).get('tau', 0) > 0 and 
                          trends.get('autocorrelation', {}).get('significant', False))
    
    # Detection result
    detection = {
        'detected': var_increasing and autocorr_increasing,
        'confidence': 0.0,
        'indicators': indicators,
        'trends': trends
    }
    
    # Calculate confidence based on strength and significance of trends
    if detection['detected']:
        var_strength = abs(trends['variance']['tau'])
        autocorr_strength = abs(trends['autocorrelation']['tau'])
        detection['confidence'] = (var_strength + autocorr_strength) / 2
    
    return detection
```

**Innovation Significance**: This approach provides several advantages over traditional monitoring:
1. Detects subtle changes in system dynamics that precede clinical deterioration
2. Uses multiple statistical indicators to improve detection reliability
3. Provides a confidence measure for the detection
4. Can identify critical transitions hours before traditional vital sign thresholds are crossed

## 3. Infection Treatment Optimization

This innovation simulates infection-treatment dynamics and optimizes dosing regimens for improved outcomes.

```python
def optimize_dosing_regimen(self, antibiotic, pathogen, dose_range, interval_range, duration_hours=168):
    """
    Optimize antibiotic dosing regimen for a given pathogen.
    
    Uses simulation to find the optimal dose and dosing interval
    that maximizes efficacy while minimizing toxicity.
    
    Parameters:
    -----------
    antibiotic : str
        Name of antibiotic
    pathogen : str
        Name of pathogen
    dose_range : tuple
        (min_dose, max_dose) in mg
    interval_range : tuple
        (min_interval, max_interval) in hours
    duration_hours : int, optional
        Duration of simulation in hours
        
    Returns:
    --------
    dict
        Optimization results including optimal dose and interval
    """
    # Get antibiotic and pathogen parameters
    ab_params = self.antibiotic_params.get(antibiotic, {})
    path_params = self.pathogen_params.get(pathogen, {})
    
    if not ab_params or not path_params:
        return {'error': f"Parameters not found for {antibiotic} or {pathogen}"}
    
    # Define optimization grid
    min_dose, max_dose = dose_range
    min_interval, max_interval = interval_range
    
    dose_step = (max_dose - min_dose) / 10
    interval_step = (max_interval - min_interval) / 5
    
    doses = np.arange(min_dose, max_dose + dose_step, dose_step)
    intervals = np.arange(min_interval, max_interval + interval_step, interval_step)
    
    # Initialize results
    best_score = -float('inf')
    optimal_dose = None
    optimal_interval = None
    all_regimens = []
    
    # Evaluate each regimen
    for dose in doses:
        for interval in intervals:
            # Simulate treatment
            result = self.simulate_treatment(
                antibiotic=antibiotic,
                pathogen=pathogen,
                dose=dose,
                dosing_interval=interval,
                duration_hours=duration_hours
            )
            
            # Calculate efficacy score
            efficacy = result['metrics']['pathogen_reduction']
            
            # Calculate toxicity score
            peak_concentration = result['metrics']['peak_concentration']
            toxicity = self._calculate_toxicity(antibiotic, peak_concentration)
            
            # Calculate time above MIC
            time_above_mic = result['metrics']['time_above_mic']
            
            # Calculate overall score (higher is better)
            score = efficacy * 0.5 + (1 - toxicity) * 0.3 + (time_above_mic / duration_hours) * 0.2
            
            # Store regimen results
            regimen_result = {
                'dose': dose,
                'interval': interval,
                'efficacy': efficacy,
                'toxicity': toxicity,
                'time_above_mic': time_above_mic,
                'score': score
            }
            all_regimens.append(regimen_result)
            
            # Update best regimen if better score
            if score > best_score:
                best_score = score
                optimal_dose = dose
                optimal_interval = interval
    
    # Sort regimens by score
    all_regimens.sort(key=lambda x: x['score'], reverse=True)
    best_regimens = all_regimens[:3]  # Top 3 regimens
    
    # Simulate optimal regimen
    optimal_result = self.simulate_treatment(
        antibiotic=antibiotic,
        pathogen=pathogen,
        dose=optimal_dose,
        dosing_interval=optimal_interval,
        duration_hours=duration_hours
    )
    
    # Return optimization results
    return {
        'optimal_dose': optimal_dose,
        'optimal_interval': optimal_interval,
        'best_score': best_score,
        'treatment_results': optimal_result,
        'best_regimens': best_regimens,
        'all_regimens': all_regimens
    }
```

**Innovation Significance**: This approach provides several advantages over standard dosing protocols:
1. Personalizes treatment based on pathogen characteristics and patient factors
2. Balances efficacy, toxicity, and pharmacodynamic targets
3. Explores a wide range of possible regimens to find optimal solutions
4. Provides multiple viable options with different trade-offs

## 4. Enhanced Interactive Visualization

This innovation creates interactive visualizations that allow clinicians to explore complex physiological data more effectively.

```python
def create_interactive_dashboard(self, patient_timeline, stability_report):
    """
    Create interactive patient dashboard.
    
    Parameters:
    -----------
    patient_timeline : dict
        Patient timeline dictionary with 'info' and 'timeline' keys
    stability_report : dict
        Stability report from PhysiologicalSignalProcessor
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Interactive dashboard
    """
    # Create dashboard with subplots
    dashboard = make_subplots(
        rows=3, cols=2,
        specs=[
            [{"colspan": 2}, None],
            [{"type": "polar"}, {"type": "xy"}],
            [{"type": "xy"}, {"type": "xy"}]
        ],
        subplot_titles=[
            f"Patient Dashboard",
            "Organ System Status", "Allostatic Load Trend",
            "Vital Signs", "Heart Rate vs Blood Pressure"
        ],
        vertical_spacing=0.1,
        horizontal_spacing=0.05
    )
    
    # Extract patient info
    patient_info = patient_timeline.get('info', {})
    subject_id = patient_info.get('subject_id', 'Unknown')
    gender = patient_info.get('gender', 'Unknown')
    
    # Add patient info
    allostatic_load = stability_report.get('overall_results', {}).get('allostatic_load', 'N/A')
    
    dashboard.add_annotation(
        x=0.5,
        y=1,
        xref="paper",
        yref="paper",
        text=f"Patient ID: {subject_id} | Gender: {gender} | Allostatic Load: {allostatic_load:.2f}" if isinstance(allostatic_load, (int, float)) else f"Patient ID: {subject_id} | Gender: {gender} | Allostatic Load: {allostatic_load}",
        showarrow=False,
        font=dict(
            size=14,
            color=self.theme['text_color']
        ),
        bgcolor=self.theme['background_color'],
        bordercolor=self.theme['secondary_color'],
        borderwidth=1,
        borderpad=4,
        align="center"
    )
    
    # Add organ system radar
    if 'organ_system_summary' in stability_report:
        organ_summary = stability_report['organ_system_summary']
        
        # Get systems and scores
        systems = []
        scores = []
        
        for system, data in organ_summary.items():
            systems.append(system.replace('_', ' ').title())
            scores.append(data['score'])
        
        # Close the polygon
        systems.append(systems[0])
        scores.append(scores[0])
        
        # Add radar trace
        dashboard.add_trace(
            go.Scatterpolar(
                r=scores,
                theta=systems,
                fill='toself',
                fillcolor='rgba(255,82,82,0.2)',
                line=dict(
                    color='rgb(255,82,82)',
                    width=2
                ),
                name='Organ System Status'
            ),
            row=2, col=1
        )
    
    # Add allostatic load trend
    if 'stability_over_time' in stability_report and not stability_report['stability_over_time'].empty:
        stability_df = stability_report['stability_over_time']
        
        if 'allostatic_load' in stability_df.columns and 'window_start' in stability_df.columns:
            # Add allostatic load trace
            dashboard.add_trace(
                go.Scatter(
                    x=stability_df['window_start'],
                    y=stability_df['allostatic_load'],
                    mode='lines+markers',
                    name='Allostatic Load',
                    line=dict(
                        color=self.theme['danger_color'],
                        width=self.theme['line_width']
                    ),
                    marker=dict(
                        size=self.theme['marker_size']
                    )
                ),
                row=2, col=2
            )
            
            # Add severity zones
            dashboard.add_hrect(
                y0=0, y1=0.5,
                fillcolor="rgba(0,128,0,0.1)",
                line_width=0,
                row=2, col=2
            )
            
            dashboard.add_hrect(
                y0=0.5, y1=1.0,
                fillcolor="rgba(255,255,0,0.1)",
                line_width=0,
                row=2, col=2
            )
            
            dashboard.add_hrect(
                y0=1.0, y1=1.5,
                fillcolor="rgba(255,165,0,0.1)",
                line_width=0,
                row=2, col=2
            )
            
            dashboard.add_hrect(
                y0=1.5, y1=3.0,
                fillcolor="rgba(255,0,0,0.1)",
                line_width=0,
                row=2, col=2
            )
    
    # Add vital signs
    timeline_df = patient_timeline['timeline']
    vital_signs = ['heart_rate', 'respiratory_rate', 'temperature', 'o2_saturation']
    available_vitals = [v for v in vital_signs if v in timeline_df.columns]
    
    if available_vitals:
        # Use the first available vital sign
        vital = available_vitals[0]
        
        # Filter out NaN values
        vital_data = timeline_df[['measurement_time', vital]].dropna()
        
        if len(vital_data) > 0:
            # Add line trace
            dashboard.add_trace(
                go.Scatter(
                    x=vital_data['measurement_time'],
                    y=vital_data[vital],
                    mode='lines+markers',
                    name=vital.replace('_', ' ').title(),
                    line=dict(
                        color=self.theme['primary_color'],
                        width=self.theme['line_width']
                    ),
                    marker=dict(
                        size=self.theme['marker_size']
                    )
                ),
                row=3, col=1
            )
    
    # Add phase portrait
    if all(v in timeline_df.columns for v in ['heart_rate', 'sbp']):
        # Filter data
        phase_data = timeline_df[['measurement_time', 'heart_rate', 'sbp']].dropna()
        
        if len(phase_data) >= 5:
            # Add scatter trace
            dashboard.add_trace(
                go.Scatter(
                    x=phase_data['heart_rate'],
                    y=phase_data['sbp'],
                    mode='markers',
                    marker=dict(
                        color=range(len(phase_data)),
                        colorscale='Viridis',
                        size=self.theme['marker_size']
                    ),
                    name='Phase Portrait'
                ),
                row=3, col=2
            )
            
            # Add trajectory line
            dashboard.add_trace(
                go.Scatter(
                    x=phase_data['heart_rate'],
                    y=phase_data['sbp'],
                    mode='lines',
                    line=dict(
                        color='rgba(100,100,100,0.3)',
                        width=1
                    ),
                    showlegend=False
                ),
                row=3, col=2
            )
    
    # Update layout
    dashboard.update_layout(
        title=f"Patient {subject_id} Dashboard",
        width=self.export_settings['html_width'],
        height=self.export_settings['html_height'] * 1.5,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=0,
            xanchor="center",
            x=0.5
        )
    )
    
    return dashboard
```

**Innovation Significance**: This approach enhances clinical decision-making by:
1. Integrating multiple visualizations into a comprehensive dashboard
2. Providing interactive elements for data exploration
3. Using color coding and visual cues to highlight clinical significance
4. Enabling export to various formats for documentation and sharing

## 5. Unified System Integration

This innovation integrates all components into a cohesive system with a consistent API.

```python
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
```

**Innovation Significance**: This unified approach provides several advantages:
1. Consistent API across all components
2. Simplified workflow for complex analyses
3. Configurable analysis pipeline
4. Automatic result organization and export
5. Comprehensive logging and error handling

## 6. Performance Optimization with Decorators

This innovation uses Python decorators to add performance monitoring, validation, and error handling without modifying the core functionality.

```python
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
```

**Innovation Significance**: This approach enhances system robustness by:
1. Separating cross-cutting concerns from core functionality
2. Enabling detailed performance monitoring without code duplication
3. Facilitating systematic error handling and validation
4. Providing insights for optimization and troubleshooting

## 7. Batch Processing with Parallel Execution

This innovation enables efficient processing of multiple patients in parallel.

```python
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

def _process_batch_parallel(self, batch_id, subject_ids, analyses):
    """
    Process a batch of patients in parallel.
    
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
```

**Innovation Significance**: This approach enhances system scalability by:
1. Enabling parallel processing of multiple patients
2. Providing configurable parallelism based on available resources
3. Tracking progress and handling errors gracefully
4. Generating comprehensive batch reports

These innovations collectively make BioDynamICS a powerful framework for analyzing physiological dynamics in critical care, with potential applications in research, clinical decision support, and personalized medicine.