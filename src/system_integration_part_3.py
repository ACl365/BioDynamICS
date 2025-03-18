# Check if this patient has sufficient data
                timeline = self.create_patient_timeline(patient_id)
                
                if timeline is not None and 'timeline' in timeline:
                    event_count = len(timeline['timeline'])
                    
                    if event_count >= 20:  # Minimum threshold for analysis
                        subject_ids.append(patient_id)
                        self._log_operation(f"Found suitable patient {patient_id} with {event_count} events")
        
        if not subject_ids:
            self._log_operation("Error: No suitable patients found")
            return None
        
        # Process each patient
        results = {}
        for i, subject_id in enumerate(subject_ids[:max_patients]):
            self._log_operation(f"Processing patient {i+1}/{min(len(subject_ids), max_patients)}: {subject_id}")
            
            # Perform comprehensive analysis
            patient_results = self.perform_comprehensive_analysis(subject_id)
            
            if patient_results is not None:
                results[subject_id] = patient_results
        
        self._log_operation(f"Completed analysis for {len(results)} patients")
        return results
    
    def generate_comparative_report(self, patient_results):
        """
        Generate a comparative report for multiple patients.
        
        Parameters:
        -----------
        patient_results : dict
            Dictionary with analysis results for each patient
            
        Returns:
        --------
        str
            Comparative report as a formatted string
        """
        if not patient_results:
            self._log_operation("Error: No patient results provided")
            return "Error: No patient results provided"
        
        # Build the report
        report = []
        
        # Header
        report.append("=" * 80)
        report.append("BIODYNIMCS COMPARATIVE ANALYSIS REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Patient summary
        report.append(f"Number of Patients: {len(patient_results)}")
        report.append("")
        
        # Compare allostatic load
        report.append("ALLOSTATIC LOAD COMPARISON")
        report.append("-" * 80)
        
        allostatic_loads = []
        
        for subject_id, results in patient_results.items():
            if 'signal_processing' in results and 'processing_results' in results['signal_processing']:
                allostatic_load = results['signal_processing']['processing_results'].get('allostatic_load', np.nan)
                allostatic_loads.append((subject_id, allostatic_load))
        
        # Sort by allostatic load (highest first)
        allostatic_loads.sort(key=lambda x: x[1] if not np.isnan(x[1]) else -1, reverse=True)
        
        for subject_id, load in allostatic_loads:
            if not np.isnan(load):
                if load > 1.5:
                    status = "Severe Stress"
                elif load > 1.0:
                    status = "Moderate Stress"
                elif load > 0.5:
                    status = "Mild Stress"
                else:
                    status = "Normal"
                    
                report.append(f"Patient {subject_id}: {load:.2f} ({status})")
            else:
                report.append(f"Patient {subject_id}: N/A (Insufficient data)")
        
        report.append("")
        
        # Compare organ system status
        report.append("ORGAN SYSTEM STATUS COMPARISON")
        report.append("-" * 80)
        
        # Get all unique organ systems
        all_systems = set()
        for results in patient_results.values():
            if 'signal_processing' in results and 'stability_report' in results['signal_processing']:
                stability_report = results['signal_processing']['stability_report']
                if 'organ_system_summary' in stability_report:
                    all_systems.update(stability_report['organ_system_summary'].keys())
        
        if all_systems:
            # Create a table header
            header = ["Patient ID"]
            header.extend([system.replace('_', ' ').title() for system in sorted(all_systems)])
            report.append("\t".join(header))
            report.append("-" * 100)
            
            # Add rows for each patient
            for subject_id, results in patient_results.items():
                row = [str(subject_id)]
                
                if 'signal_processing' in results and 'stability_report' in results['signal_processing']:
                    stability_report = results['signal_processing']['stability_report']
                    if 'organ_system_summary' in stability_report:
                        organ_summary = stability_report['organ_system_summary']
                        
                        for system in sorted(all_systems):
                            if system in organ_summary:
                                score = organ_summary[system].get('score', np.nan)
                                if not np.isnan(score):
                                    row.append(f"{score:.2f}")
                                else:
                                    row.append("N/A")
                            else:
                                row.append("N/A")
                    else:
                        row.extend(["N/A"] * len(all_systems))
                else:
                    row.extend(["N/A"] * len(all_systems))
                
                report.append("\t".join(row))
            
            report.append("")
        else:
            report.append("No organ system data available for comparison")
            report.append("")
        
        # Compare critical transitions
        report.append("CRITICAL TRANSITIONS DETECTED")
        report.append("-" * 80)
        
        transitions_detected = False
        
        for subject_id, results in patient_results.items():
            if 'dynamical_analysis' in results:
                patient_transitions = []
                
                for vital, analysis in results['dynamical_analysis'].items():
                    if 'critical_transitions' in analysis:
                        transition = analysis['critical_transitions']
                        if transition.get('detected', False):
                            probability = transition.get('probability', 0)
                            patient_transitions.append((vital, probability))
                
                if patient_transitions:
                    transitions_detected = True
                    report.append(f"Patient {subject_id}:")
                    for vital, probability in patient_transitions:
                        report.append(f"  - {vital.replace('_', ' ').title()}: {probability:.2f} probability")
                    report.append("")
        
        if not transitions_detected:
            report.append("No critical transitions detected in any patients")
            report.append("")
        
        # Compare infection treatment outcomes if available
        report.append("INFECTION TREATMENT COMPARISON")
        report.append("-" * 80)
        
        treatment_data = False
        
        for subject_id, results in patient_results.items():
            if 'treatment_analysis' in results and results['treatment_analysis'] is not None:
                treatment = results['treatment_analysis']
                pathogen = treatment.get('pathogen', 'Unknown').replace('_', ' ').title()
                antibiotic = treatment.get('antibiotic', 'Unknown').title()
                
                interpretation = treatment.get('interpretation', {})
                effectiveness = interpretation.get('effectiveness', 'Unknown')
                recommendation = interpretation.get('recommendation', 'Unknown')
                
                treatment_data = True
                report.append(f"Patient {subject_id}:")
                report.append(f"  Pathogen: {pathogen}")
                report.append(f"  Antibiotic: {antibiotic}")
                report.append(f"  Effectiveness: {effectiveness}")
                report.append(f"  Recommendation: {recommendation}")
                report.append("")
        
        if not treatment_data:
            report.append("No infection treatment data available for comparison")
            report.append("")
        
        # Overall patient risk assessment
        report.append("OVERALL PATIENT RISK ASSESSMENT")
        report.append("-" * 80)
        
        # Calculate a composite risk score for each patient
        risk_scores = []
        
        for subject_id, results in patient_results.items():
            risk_components = []
            
            # Component 1: Allostatic load
            if 'signal_processing' in results and 'processing_results' in results['signal_processing']:
                allostatic_load = results['signal_processing']['processing_results'].get('allostatic_load', np.nan)
                if not np.isnan(allostatic_load):
                    # Scale to 0-10 range
                    risk_components.append(min(10, allostatic_load * 5))
            
            # Component 2: Critical transitions
            if 'dynamical_analysis' in results:
                max_transition_prob = 0
                
                for vital, analysis in results['dynamical_analysis'].items():
                    if 'critical_transitions' in analysis:
                        transition = analysis['critical_transitions']
                        probability = transition.get('probability', 0)
                        max_transition_prob = max(max_transition_prob, probability)
                
                # Scale to 0-10 range
                risk_components.append(max_transition_prob * 10)
            
            # Component 3: Treatment effectiveness
            if 'treatment_analysis' in results and results['treatment_analysis'] is not None:
                treatment = results['treatment_analysis']
                
                # Get the best regimen metrics
                comparison = treatment.get('treatment_comparison', {})
                if 'best_regimens' in comparison and 'all_regimens' in comparison:
                    best_idx = comparison['best_regimens'].get('overall', 0)
                    best_regimen = comparison['all_regimens'][best_idx]
                    
                    # If treatment is not successful, add risk
                    if not best_regimen.get('is_successful', True):
                        risk_components.append(5)
            
            # Calculate overall risk score (average of components)
            if risk_components:
                risk_score = sum(risk_components) / len(risk_components)
                risk_scores.append((subject_id, risk_score))
        
        # Sort by risk score (highest first)
        risk_scores.sort(key=lambda x: x[1], reverse=True)
        
        for subject_id, score in risk_scores:
            if score > 7:
                risk_category = "High Risk"
            elif score > 4:
                risk_category = "Moderate Risk"
            else:
                risk_category = "Low Risk"
                
            report.append(f"Patient {subject_id}: {score:.1f}/10 ({risk_category})")
        
        if not risk_scores:
            report.append("Insufficient data for risk assessment")
        
        report.append("")
        report.append("=" * 80)
        report.append(f"Report generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("BioDynamICS Clinical Analysis System")
        report.append("=" * 80)
        
        return "\n".join(report)
    
    def export_results(self, analysis_results, export_path=None):
        """
        Export analysis results to files.
        
        Parameters:
        -----------
        analysis_results : dict
            Analysis results from perform_comprehensive_analysis
        export_path : str, optional
            Path for exported files (default: current directory)
            
        Returns:
        --------
        dict
            Information about exported files
        """
        if analysis_results is None:
            self._log_operation("Error: No analysis results provided")
            return {"error": "No analysis results provided"}
        
        if export_path is None:
            export_path = "."
            
        # Create export directory if it doesn't exist
        if not os.path.exists(export_path):
            os.makedirs(export_path)
            
        self._log_operation(f"Exporting results to {export_path}")
        
        # Get patient ID
        subject_id = analysis_results.get('patient_info', {}).get('subject_id', 'unknown')
        
        # Export report
        report = self.generate_clinical_report(analysis_results)
        report_file = os.path.join(export_path, f"patient_{subject_id}_report.txt")
        
        with open(report_file, 'w') as f:
            f.write(report)
            
        self._log_operation(f"Exported clinical report to {report_file}")
        
        # Export visualizations
        visualization_files = []
        
        for viz_name, fig in analysis_results.get('visualizations', []):
            viz_file = os.path.join(export_path, f"patient_{subject_id}_{viz_name}.png")
            fig.savefig(viz_file, dpi=300)
            visualization_files.append(viz_file)
            
        self._log_operation(f"Exported {len(visualization_files)} visualizations")
        
        # Export numerical data
        data_file = os.path.join(export_path, f"patient_{subject_id}_data.csv")
        
        # Prepare data for export
        export_data = []
        
        # Signal processing features
        if 'signal_processing' in analysis_results and 'processing_results' in analysis_results['signal_processing']:
            for key, value in analysis_results['signal_processing']['processing_results'].get('features', {}).items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    export_data.append({'category': 'signal_processing', 'metric': key, 'value': value})
        
        # Dynamical analysis features
        for vital, analysis in analysis_results.get('dynamical_analysis', {}).items():
            if 'stability' in analysis:
                stability = analysis['stability']
                if 'maximal_lyapunov_exponent' in stability:
                    export_data.append({
                        'category': 'dynamical_analysis',
                        'metric': f"{vital}_lyapunov_exponent",
                        'value': stability['maximal_lyapunov_exponent']
                    })
                
                # Recurrence metrics
                if 'recurrence_metrics' in stability:
                    for metric, value in stability['recurrence_metrics'].items():
                        if isinstance(value, (int, float)) and not isinstance(value, bool) and not np.isnan(value):
                            export_data.append({
                                'category': 'dynamical_analysis',
                                'metric': f"{vital}_{metric}",
                                'value': value
                            })
        
        # Create DataFrame and export
        if export_data:
            pd.DataFrame(export_data).to_csv(data_file, index=False)
            self._log_operation(f"Exported numerical data to {data_file}")
        else:
            self._log_operation("No numerical data to export")
        
        return {
            'report_file': report_file,
            'visualization_files': visualization_files,
            'data_file': data_file if export_data else None
        }
    
    def run_batch_analysis(self, subject_ids=None, max_patients=5, export_path=None):
        """
        Run batch analysis on multiple patients and generate comparative report.
        
        Parameters:
        -----------
        subject_ids : list, optional
            List of patient subject IDs (if None, finds suitable patients)
        max_patients : int, optional
            Maximum number of patients to process if subject_ids is None
        export_path : str, optional
            Path for exported files
            
        Returns:
        --------
        tuple
            (comparison report, individual patient results)
        """
        # Process multiple patients
        patient_results = self.process_multiple_patients(subject_ids, max_patients)
        
        if patient_results is None or len(patient_results) == 0:
            self._log_operation("Error: No patient results available")
            return None, None
        
        # Generate comparative report
        comparison_report = self.generate_comparative_report(patient_results)
        
        # Export results if path provided
        if export_path is not None:
            # Create export directory if it doesn't exist
            if not os.path.exists(export_path):
                os.makedirs(export_path)
                
            # Export comparison report
            report_file = os.path.join(export_path, "comparative_report.txt")
            with open(report_file, 'w') as f:
                f.write(comparison_report)
                
            self._log_operation(f"Exported comparative report to {report_file}")
            
            # Export individual patient results
            for subject_id, results in patient_results.items():
                patient_dir = os.path.join(export_path, f"patient_{subject_id}")
                if not os.path.exists(patient_dir):
                    os.makedirs(patient_dir)
                    
                self.export_results(results, patient_dir)
        
        return comparison_report, patient_results
