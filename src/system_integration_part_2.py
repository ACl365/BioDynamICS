# Check for relevant prescriptions
                    if 'subject_id' in prescriptions.columns:
                        patient_prescriptions = prescriptions[prescriptions['subject_id'] == subject_id]
                        
                        if len(patient_prescriptions) > 0:
                            # Filter for antibiotics
                            common_antibiotics = [
                                'VANCOMYCIN', 'CEFTRIAXONE', 'CIPROFLOXACIN', 'PIPERACILLIN',
                                'MEROPENEM', 'LEVOFLOXACIN', 'AZITHROMYCIN'
                            ]
                            
                            antibiotic_mask = patient_prescriptions['DRUG'].str.contains(
                                '|'.join(common_antibiotics), case=False, na=False
                            )
                            
                            antibiotics = patient_prescriptions[antibiotic_mask]['DRUG'].value_counts().to_dict()
                        else:
                            antibiotics = {}
                    else:
                        antibiotics = {}
                    
                    patient_infections[subject_id] = {
                        'organisms': organisms,
                        'antibiotics': antibiotics
                    }
            
            results['patient_infections'] = patient_infections
        
        self._log_operation("Microbiology data analysis complete")
        return results
    
    # =========================================================================
    # Visualization Methods
    # =========================================================================
    
    def create_patient_dashboard(self, patient_timeline):
        """
        Create a comprehensive patient dashboard with visualizations.
        
        Parameters:
        -----------
        patient_timeline : dict
            Patient timeline dictionary with 'info' and 'timeline' keys
            
        Returns:
        --------
        matplotlib.figure.Figure
            Dashboard figure
        """
        if patient_timeline is None or 'timeline' not in patient_timeline:
            self._log_operation("Error: Invalid patient timeline")
            return None
        
        # Process physiological signals
        self._log_operation("Processing signals for dashboard...")
        processing_results = self.process_physiological_signals(patient_timeline)
        
        if processing_results is None:
            self._log_operation("Error: Failed to process physiological signals")
            return None
        
        # Create stability report if not included in processing results
        if 'stability_report' not in processing_results:
            self._log_operation("Generating stability report...")
            stability_report = self.signal_processor.create_stability_report(patient_timeline)
        else:
            stability_report = processing_results['stability_report']
        
        # Create dashboard
        self._log_operation("Creating patient dashboard...")
        dashboard = self.visualizer.create_patient_dashboard(
            patient_timeline, stability_report
        )
        
        return dashboard
    
    def visualize_dynamical_analysis(self, time_series, analysis_results, vital_sign='heart_rate'):
        """
        Create visualizations for dynamical systems analysis.
        
        Parameters:
        -----------
        time_series : pandas.Series
            Time series data with datetime index
        analysis_results : dict
            Dynamical systems analysis results
        vital_sign : str, optional
            Name of the vital sign (default: 'heart_rate')
            
        Returns:
        --------
        list
            List of matplotlib figures
        """
        if time_series is None or len(time_series) < 10:
            self._log_operation("Error: Insufficient data for visualization")
            return []
        
        if analysis_results is None:
            self._log_operation("Error: No analysis results provided")
            return []
        
        self._log_operation(f"Creating dynamical analysis visualizations for {vital_sign}...")
        
        figures = []
        
        # Original time series
        fig_time_series = plt.figure(figsize=(12, 6))
        plt.plot(time_series.index, time_series.values, 'b-', alpha=0.7)
        plt.title(f'{vital_sign.replace("_", " ").title()} Time Series')
        plt.xlabel('Time')
        plt.ylabel(vital_sign.replace('_', ' ').title())
        plt.grid(True, linestyle='--', alpha=0.7)
        figures.append(fig_time_series)
        
        # 2D Phase space reconstruction
        embedded_2d = analysis_results['embedding']['embedded_2d']
        if embedded_2d is not None and len(embedded_2d) > 0:
            fig_phase_2d = plt.figure(figsize=(10, 8))
            plt.scatter(embedded_2d[:, 0], embedded_2d[:, 1], alpha=0.6, s=10)
            plt.plot(embedded_2d[:, 0], embedded_2d[:, 1], 'b-', alpha=0.3)
            
            delay = analysis_results['embedding']['optimal_time_delay']
            plt.title(f'2D Phase Space Reconstruction (delay={delay})')
            plt.xlabel(f'{vital_sign.replace("_", " ").title()}(t)')
            plt.ylabel(f'{vital_sign.replace("_", " ").title()}(t+{delay})')
            plt.grid(True, alpha=0.3)
            figures.append(fig_phase_2d)
        
        # 3D Phase space reconstruction if available
        embedded_3d = analysis_results['embedding']['embedded_3d']
        if embedded_3d is not None and len(embedded_3d) > 0:
            fig_phase_3d = plt.figure(figsize=(12, 10))
            ax = fig_phase_3d.add_subplot(111, projection='3d')
            
            ax.scatter(embedded_3d[:, 0], embedded_3d[:, 1], embedded_3d[:, 2], alpha=0.6, s=10)
            ax.plot(embedded_3d[:, 0], embedded_3d[:, 1], embedded_3d[:, 2], 'b-', alpha=0.3)
            
            delay = analysis_results['embedding']['optimal_time_delay']
            ax.set_title(f'3D Phase Space Reconstruction (delay={delay})')
            ax.set_xlabel(f'{vital_sign.replace("_", " ").title()}(t)')
            ax.set_ylabel(f'{vital_sign.replace("_", " ").title()}(t+{delay})')
            ax.set_zlabel(f'{vital_sign.replace("_", " ").title()}(t+{2*delay})')
            figures.append(fig_phase_3d)
        
        # Recurrence plot if available
        recurrence_matrix = analysis_results['stability']['recurrence_matrix']
        if recurrence_matrix is not None and len(recurrence_matrix) > 0:
            fig_recurrence = plt.figure(figsize=(10, 8))
            plt.imshow(recurrence_matrix, cmap='binary', origin='lower')
            plt.colorbar(label='Recurrence')
            plt.title(f'Recurrence Plot ({vital_sign.replace("_", " ").title()})')
            plt.xlabel('Time Index')
            plt.ylabel('Time Index')
            figures.append(fig_recurrence)
        
        # Summary visualization with interpretation
        fig_summary = plt.figure(figsize=(12, 8))
        plt.axis('off')
        
        interpretation = analysis_results.get('interpretation', {})
        stability = interpretation.get('stability', 'Unknown')
        attractors = interpretation.get('attractors', 'Unknown')
        overall = interpretation.get('overall', 'Unknown')
        
        lyapunov = analysis_results['stability']['lyapunov_exponent']
        
        summary_text = f"""
        Dynamical Systems Analysis Summary for {vital_sign.replace('_', ' ').title()}
        
        Stability Assessment: {stability}
        Attractor Type: {attractors}
        Maximal Lyapunov Exponent: {lyapunov:.4f if not np.isnan(lyapunov) else 'N/A'}
        
        Overall Assessment:
        {overall}
        """
        
        plt.text(0.5, 0.5, summary_text, ha='center', va='center', fontsize=12,
                transform=plt.gca().transAxes, bbox=dict(facecolor='aliceblue', alpha=0.5))
        
        figures.append(fig_summary)
        
        return figures
    
    def visualize_treatment_analysis(self, treatment_results):
        """
        Create visualizations for infection treatment analysis.
        
        Parameters:
        -----------
        treatment_results : dict
            Infection treatment analysis results
            
        Returns:
        --------
        list
            List of matplotlib figures
        """
        if treatment_results is None:
            self._log_operation("Error: No treatment results provided")
            return []
        
        pathogen = treatment_results['pathogen']
        antibiotic = treatment_results['antibiotic']
        
        self._log_operation(f"Creating treatment analysis visualizations for {pathogen} and {antibiotic}...")
        
        figures = []
        
        # Treatment comparison visualization
        comparison = treatment_results['treatment_comparison']
        best_regimens = comparison['best_regimens']
        all_regimens = comparison['all_regimens']
        
        fig_comparison = plt.figure(figsize=(14, 10))
        
        # Plot bacterial load over time for each regimen
        ax1 = fig_comparison.add_subplot(211)
        
        for i, regimen in enumerate(all_regimens):
            sim_results = regimen['simulation_results']
            times = sim_results['times']
            load = sim_results['bacterial_load']
            
            label = f"Regimen {i+1}: {regimen['dose']}mg q{regimen['interval']}h"
            style = '-'
            width = 2
            
            if i == best_regimens['overall']:
                style = '-'
                width = 3
            else:
                style = '--'
                width = 1.5
            
            ax1.semilogy(times, load, style, linewidth=width, label=label)
        
        ax1.set_title(f"Bacterial Load Dynamics with Different {antibiotic.title()} Regimens")
        ax1.set_xlabel("Time (hours)")
        ax1.set_ylabel("Bacterial Load (CFU/mL)")
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot antibiotic concentration
        ax2 = fig_comparison.add_subplot(212)
        
        for i, regimen in enumerate(all_regimens):
            sim_results = regimen['simulation_results']
            times = sim_results['times']
            conc = sim_results['antibiotic_concentration']
            
            label = f"Regimen {i+1}: {regimen['dose']}mg q{regimen['interval']}h"
            style = '-'
            width = 2
            
            if i == best_regimens['overall']:
                style = '-'
                width = 3
            else:
                style = '--'
                width = 1.5
            
            ax2.plot(times, conc, style, linewidth=width, label=label)
        
        # Add MIC line
        if isinstance(self.infection_modeler.antibiotic_params[antibiotic]['mic'], dict):
            mic = self.infection_modeler.antibiotic_params[antibiotic]['mic'].get(
                pathogen, 
                self.infection_modeler.antibiotic_params[antibiotic]['mic']['default']
            )
        else:
            mic = self.infection_modeler.antibiotic_params[antibiotic]['mic']
        
        ax2.axhline(y=mic, color='r', linestyle='--', label=f"MIC ({mic} mg/L)")
        
        ax2.set_title(f"{antibiotic.title()} Concentration Over Time")
        ax2.set_xlabel("Time (hours)")
        ax2.set_ylabel("Concentration (mg/L)")
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        figures.append(fig_comparison)
        
        # Optimization results visualization
        optimization = treatment_results['treatment_optimization']
        
        fig_optimization = plt.figure(figsize=(14, 8))
        
        # Plot bacterial load for optimized regimen
        ax1 = fig_optimization.add_subplot(211)
        
        opt_results = optimization['treatment_results']
        times = opt_results['times']
        load = opt_results['bacterial_load']
        
        ax1.semilogy(times, load, 'b-', linewidth=2)
        
        # Add dose times as vertical lines
        for dose_time in opt_results['dose_times']:
            ax1.axvline(x=dose_time, color='g', linestyle='--', alpha=0.5)
        
        ax1.set_title(f"Bacterial Load with Optimized {antibiotic.title()} Regimen")
        ax1.set_xlabel("Time (hours)")
        ax1.set_ylabel("Bacterial Load (CFU/mL)")
        ax1.grid(True, alpha=0.3)
        
        # Plot antibiotic concentration
        ax2 = fig_optimization.add_subplot(212)
        
        conc = opt_results['antibiotic_concentration']
        ax2.plot(times, conc, 'b-', linewidth=2)
        
        # Add MIC line
        ax2.axhline(y=mic, color='r', linestyle='--', label=f"MIC ({mic} mg/L)")
        
        # Add dose times as vertical lines
        for dose_time in opt_results['dose_times']:
            ax2.axvline(x=dose_time, color='g', linestyle='--', alpha=0.5)
        
        ax2.set_title(f"Optimized {antibiotic.title()} Regimen: {optimization['optimal_dose']:.0f}mg every {optimization['optimal_interval']:.1f}h")
        ax2.set_xlabel("Time (hours)")
        ax2.set_ylabel("Concentration (mg/L)")
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        figures.append(fig_optimization)
        
        # Summary visualization with interpretation
        fig_summary = plt.figure(figsize=(12, 8))
        plt.axis('off')
        
        interpretation = treatment_results.get('interpretation', {})
        effectiveness = interpretation.get('effectiveness', 'Unknown')
        key_parameter = interpretation.get('key_parameter', 'Unknown')
        recommendation = interpretation.get('recommendation', 'Unknown')
        expected_outcome = interpretation.get('expected_outcome', 'Unknown')
        
        summary_text = f"""
        Infection Treatment Analysis Summary
        
        Pathogen: {pathogen.replace('_', ' ').title()}
        Antibiotic: {antibiotic.title()}
        
        Effectiveness Assessment:
        {effectiveness}
        
        Key PK/PD Parameter:
        {key_parameter}
        
        Dosing Recommendation:
        {recommendation}
        
        Expected Outcome:
        {expected_outcome}
        """
        
        plt.text(0.5, 0.5, summary_text, ha='center', va='center', fontsize=12,
                transform=plt.gca().transAxes, bbox=dict(facecolor='aliceblue', alpha=0.5))
        
        figures.append(fig_summary)
        
        return figures
    
    # =========================================================================
    # Comprehensive Analysis Methods
    # =========================================================================
    
    def perform_comprehensive_analysis(self, subject_id=None):
        """
        Perform comprehensive analysis on a patient, including:
        - Physiological signal processing
        - Dynamical systems analysis
        - Treatment optimization (if applicable)
        
        Parameters:
        -----------
        subject_id : int, optional
            Patient subject ID (if None, tries to find a suitable patient)
            
        Returns:
        --------
        dict
            Comprehensive analysis results
        """
        # Load patient data
        patient_timeline = self.load_patient_data(subject_id)
        
        if patient_timeline is None:
            self._log_operation("Error: Failed to load patient data")
            return None
        
        subject_id = patient_timeline['info']['subject_id']
        self._log_operation(f"Performing comprehensive analysis for patient {subject_id}...")
        
        # Initialize results
        results = {
            'patient_info': patient_timeline['info'],
            'timeline_length': len(patient_timeline['timeline']),
            'signal_processing': None,
            'dynamical_analysis': {},
            'treatment_analysis': None,
            'visualizations': []
        }
        
        # Perform signal processing
        signal_results = self.process_physiological_signals(patient_timeline)
        if signal_results is not None:
            results['signal_processing'] = signal_results
        
        # Perform dynamical systems analysis on vital signs
        vital_signs = ['heart_rate', 'respiratory_rate', 'sbp', 'dbp', 'temperature']
        
        for vital in vital_signs:
            # Extract and analyze the vital sign
            time_series, _ = self.analyze_vital_sign(patient_timeline, vital)
            
            if time_series is not None and len(time_series) >= 20:
                # Perform dynamical analysis
                analysis = self.perform_dynamical_analysis(time_series, vital)
                
                if analysis is not None:
                    results['dynamical_analysis'][vital] = analysis
                    
                    # Check for critical transitions
                    if len(time_series) >= 50:
                        transition = self.detect_critical_transitions(time_series, vital)
                        results['dynamical_analysis'][vital]['critical_transitions'] = transition
        
        # Perform infection and treatment analysis if microbiology data exists
        if self.data_integrator is not None:
            if 'MICROBIOLOGYEVENTS' in self.data_integrator.tables:
                # Check if this patient has microbiology data
                micro = self.data_integrator.tables['MICROBIOLOGYEVENTS']
                
                if 'subject_id' in micro.columns:
                    patient_micro = micro[micro['subject_id'] == subject_id]
                    
                    if len(patient_micro) > 0 and 'ORGANISM' in patient_micro.columns:
                        # Get the most common organism for this patient
                        organisms = patient_micro['ORGANISM'].value_counts()
                        
                        if len(organisms) > 0:
                            top_organism = organisms.index[0]
                            
                            # Map MIMIC organism names to our model's pathogen types
                            organism_map = {
                                'STAPHYLOCOCCUS AUREUS': 's_aureus',
                                'ESCHERICHIA COLI': 'e_coli',
                                'PSEUDOMONAS AERUGINOSA': 'p_aeruginosa',
                                'KLEBSIELLA PNEUMONIAE': 'k_pneumoniae'
                            }
                            
                            pathogen = None
                            for mimic_name, model_name in organism_map.items():
                                if mimic_name in top_organism:
                                    pathogen = model_name
                                    break
                            
                            if pathogen is None:
                                pathogen = 'default'
                            
                            # Get antibiotic prescriptions for this patient
                            if 'PRESCRIPTIONS' in self.data_integrator.tables:
                                prescriptions = self.data_integrator.tables['PRESCRIPTIONS']
                                patient_prescriptions = prescriptions[prescriptions['subject_id'] == subject_id]
                                
                                # Map MIMIC antibiotic names to our model's antibiotic types
                                antibiotic_map = {
                                    'VANCOMYCIN': 'vancomycin',
                                    'CEFTRIAXONE': 'ceftriaxone',
                                    'CIPROFLOXACIN': 'ciprofloxacin',
                                    'PIPERACILLIN': 'piperacillin'
                                }
                                
                                # Find antibiotics prescribed to this patient
                                prescribed_antibiotics = []
                                
                                if 'DRUG' in patient_prescriptions.columns:
                                    for mimic_name, model_name in antibiotic_map.items():
                                        if patient_prescriptions['DRUG'].str.contains(mimic_name, case=False, na=False).any():
                                            prescribed_antibiotics.append(model_name)
                                
                                antibiotic = prescribed_antibiotics[0] if prescribed_antibiotics else 'default'
                                
                                # Perform treatment analysis
                                treatment_results = self.analyze_infection_treatment(pathogen, antibiotic)
                                results['treatment_analysis'] = treatment_results
        
        # Create visualizations
        self._log_operation("Creating visualizations...")
        
        # Patient dashboard
        dashboard = self.create_patient_dashboard(patient_timeline)
        if dashboard is not None:
            results['visualizations'].append(('dashboard', dashboard))
        
        # Dynamical analysis visualizations
        for vital, analysis in results['dynamical_analysis'].items():
            time_series, _ = self.analyze_vital_sign(patient_timeline, vital)
            if time_series is not None:
                figures = self.visualize_dynamical_analysis(time_series, analysis, vital)
                for i, fig in enumerate(figures):
                    results['visualizations'].append((f'dynamical_{vital}_{i}', fig))
        
        # Treatment analysis visualizations
        if results['treatment_analysis'] is not None:
            figures = self.visualize_treatment_analysis(results['treatment_analysis'])
            for i, fig in enumerate(figures):
                results['visualizations'].append((f'treatment_{i}', fig))
        
        self._log_operation("Comprehensive analysis complete")
        return results
    
    def generate_clinical_report(self, analysis_results):
        """
        Generate a clinical report with findings and recommendations.
        
        Parameters:
        -----------
        analysis_results : dict
            Comprehensive analysis results
            
        Returns:
        --------
        str
            Clinical report as a formatted string
        """
        if analysis_results is None:
            self._log_operation("Error: No analysis results provided")
            return "Error: No analysis results provided"
        
        patient_info = analysis_results.get('patient_info', {})
        subject_id = patient_info.get('subject_id', 'Unknown')
        
        self._log_operation(f"Generating clinical report for patient {subject_id}...")
        
        # Build the report
        report = []
        
        # Header
        report.append("=" * 80)
        report.append(f"BIODYNIMCS CLINICAL ANALYSIS REPORT - PATIENT {subject_id}")
        report.append("=" * 80)
        report.append("")
        
        # Patient information
        report.append("PATIENT INFORMATION")
        report.append("-" * 80)
        report.append(f"Patient ID: {subject_id}")
        report.append(f"Gender: {patient_info.get('gender', 'Unknown')}")
        report.append(f"Admissions: {patient_info.get('admissions', 'Unknown')}")
        report.append(f"ICU Stays: {patient_info.get('icustays', 'Unknown')}")
        report.append("")
        
        # Signal processing summary
        signal_results = analysis_results.get('signal_processing', {})
        stability_report = signal_results.get('stability_report', {})
        
        report.append("PHYSIOLOGICAL STABILITY ASSESSMENT")
        report.append("-" * 80)
        
        if 'overall_assessment' in stability_report:
            assessment = stability_report['overall_assessment']
            if 'stability_classification' in assessment:
                report.append(f"Overall Stability: {assessment['stability_classification']}")
                
                if 'average_lyapunov_exponent' in assessment and not np.isnan(assessment['average_lyapunov_exponent']):
                    report.append(f"Average Lyapunov Exponent: {assessment['average_lyapunov_exponent']:.4f}")
                
                if 'systems_with_transitions' in assessment and assessment['systems_with_transitions']:
                    report.append("WARNING: Critical transitions detected in the following systems:")
                    for system in assessment['systems_with_transitions']:
                        report.append(f"  - {system.replace('_', ' ').title()}")
        
        # Organ system status
        if 'organ_system_summary' in stability_report:
            report.append("\nOrgan System Status:")
            for system, status in stability_report['organ_system_summary'].items():
                system_name = system.replace('_', ' ').title()
                score = status.get('score', 0)
                abnormal = status.get('abnormal_measures', 0)
                total = status.get('total_measures', 0)
                
                if score > 1.0:
                    stability = "Unstable"
                elif score > 0.5:
                    stability = "Marginally Stable"
                else:
                    stability = "Stable"
                    
                report.append(f"  {system_name}: {stability} (Score: {score:.2f}, {abnormal}/{total} abnormal measures)")
        
        report.append("")
        
        # Critical values
        if 'critical_values' in stability_report and stability_report['critical_values']:
            report.append("CRITICAL VALUES")
            report.append("-" * 80)
            
            for cv in stability_report['critical_values']:
                measure = cv['measurement'].replace('_', ' ').title()
                value = cv['value']
                ref_range = cv['reference_range']
                deviation = cv['percent_deviation']
                
                report.append(f"  {measure}: {value:.1f} (Reference Range: {ref_range[0]}-{ref_range[1]}, Deviation: {deviation:.1f}%)")
            
            report.append("")
        
        # Dynamical systems analysis
        if 'dynamical_analysis' in analysis_results and analysis_results['dynamical_analysis']:
            report.append("DYNAMICAL SYSTEMS ANALYSIS")
            report.append("-" * 80)
            
            for vital, analysis in analysis_results['dynamical_analysis'].items():
                vital_name = vital.replace('_', ' ').title()
                
                if 'interpretation' in analysis:
                    interpretation = analysis['interpretation']
                    
                    report.append(f"{vital_name}:")
                    report.append(f"  Stability: {interpretation.get('stability', 'Unknown')}")
                    report.append(f"  Attractors: {interpretation.get('attractors', 'Unknown')}")
                    
                    if 'overall' in interpretation:
                        report.append(f"  Assessment: {interpretation['overall']}")
                    
                    # Check for critical transitions
                    if 'critical_transitions' in analysis:
                        transitions = analysis['critical_transitions']
                        
                        if transitions.get('transition_detection', {}).get('detected', False):
                            probability = transitions['transition_detection'].get('probability', 0)
                            report.append(f"  WARNING: Critical transition detected (Probability: {probability:.2f})")
                    
                    report.append("")
        
        # Treatment analysis
        if 'treatment_analysis' in analysis_results and analysis_results['treatment_analysis']:
            report.append("INFECTION TREATMENT ANALYSIS")
            report.append("-" * 80)
            
            treatment = analysis_results['treatment_analysis']
            pathogen = treatment.get('pathogen', 'Unknown').replace('_', ' ').title()
            antibiotic = treatment.get('antibiotic', 'Unknown').title()
            
            report.append(f"Pathogen: {pathogen}")
            report.append(f"Antibiotic: {antibiotic}")
            
            if 'interpretation' in treatment:
                interpretation = treatment['interpretation']
                
                report.append(f"\nEffectiveness: {interpretation.get('effectiveness', 'Unknown')}")
                report.append(f"Key Parameter: {interpretation.get('key_parameter', 'Unknown')}")
                report.append(f"Recommendation: {interpretation.get('recommendation', 'Unknown')}")
                report.append(f"Expected Outcome: {interpretation.get('expected_outcome', 'Unknown')}")
            
            report.append("")
        
        # Recommendations
        report.append("CLINICAL RECOMMENDATIONS")
        report.append("-" * 80)
        
        # Generate recommendations based on analysis results
        recommendations = []
        
        # Check for critical transitions
        critical_transitions_detected = False
        for vital, analysis in analysis_results.get('dynamical_analysis', {}).items():
            if 'critical_transitions' in analysis:
                transitions = analysis['critical_transitions']
                if transitions.get('transition_detection', {}).get('detected', False):
                    vital_name = vital.replace('_', ' ').title()
                    recommendations.append(f"Monitor {vital_name} closely due to signs of approaching critical transition")
                    critical_transitions_detected = True
        
        if critical_transitions_detected:
            recommendations.append("Consider proactive intervention to prevent physiological deterioration")
        
        # Check organ system status
        if 'stability_report' in signal_results and 'organ_system_summary' in stability_report:
            for system, status in stability_report['organ_system_summary'].items():
                system_name = system.replace('_', ' ').title()
                score = status.get('score', 0)
                
                if score > 1.0:
                    recommendations.append(f"Urgent intervention required for {system_name} system")
                elif score > 0.5:
                    recommendations.append(f"Close monitoring recommended for {system_name} system")
        
        # Treatment recommendations
        if 'treatment_analysis' in analysis_results and analysis_results['treatment_analysis']:
            treatment = analysis_results['treatment_analysis']
            
            if 'interpretation' in treatment and 'recommendation' in treatment['interpretation']:
                recommendations.append(treatment['interpretation']['recommendation'])
        
        # Add recommendations to report
        if recommendations:
            for recommendation in recommendations:
                report.append(f"* {recommendation}")
        else:
            report.append("No specific recommendations at this time")
        
        report.append("")
        report.append("=" * 80)
        report.append(f"Report generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("BioDynamICS Clinical Analysis System")
        report.append("=" * 80)
        
        return "\n".join(report)
    
    # =========================================================================
    # Batch Processing Methods
    # =========================================================================
    
    def process_multiple_patients(self, subject_ids=None, max_patients=5):
        """
        Process multiple patients for comparative analysis.
        
        Parameters:
        -----------
        subject_ids : list, optional
            List of patient subject IDs (if None, finds suitable patients)
        max_patients : int, optional
            Maximum number of patients to process if subject_ids is None
            
        Returns:
        --------
        dict
            Dictionary with analysis results for each patient
        """
        if self.data_integrator is None:
            self._log_operation("Error: Data integrator not initialized")
            return None
        
        # If no subject IDs provided, find suitable patients
        if subject_ids is None:
            self._log_operation(f"Finding up to {max_patients} suitable patients...")
            
            if 'PATIENTS' not in self.data_integrator.tables:
                self.data_integrator.load_core_tables()
            
            # Get all patient IDs
            all_patients = self.data_integrator.tables['PATIENTS']['subject_id'].unique()
            
            subject_ids = []
            for patient_id in all_patients:
                if len(subject_ids) >= max_patients:
                    break
                    
                # Check if this patient has sufficient data