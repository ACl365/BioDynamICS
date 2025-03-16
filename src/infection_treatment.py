"""
BioDynamICS - Infection Treatment Modeling Module

This module implements models for infection dynamics and antimicrobial treatment
responses based on clinical data from the MIMIC-III dataset and theoretical models.
It provides methods for analyzing treatment effectiveness and optimizing treatment timing.

Author: Alexander Clarke
Date: March 16, 2025
"""

import numpy as np
import pandas as pd
from scipy import integrate, optimize, stats
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings

class InfectionTreatmentModeler:
    """
    Models infection dynamics and antibiotic treatment responses for clinical applications.
    
    This class implements pharmacokinetic/pharmacodynamic (PK/PD) models, bacterial growth
    and death dynamics, and treatment optimization algorithms.
    """
    
    def __init__(self):
        """Initialize the InfectionTreatmentModeler with default parameters."""
        # Configuration parameters for bacterial growth models
        self.bacterial_growth_params = {
            'default': {
                'growth_rate': 0.5,      # per hour
                'carrying_capacity': 1e9, # CFU/mL
                'initial_load': 1e4,     # CFU/mL
                'death_rate': 0.1        # per hour (natural death)
            },
            # Parameters for specific pathogens
            'e_coli': {
                'growth_rate': 0.6,
                'carrying_capacity': 2e9,
                'initial_load': 1e4,
                'death_rate': 0.05
            },
            's_aureus': {
                'growth_rate': 0.4,
                'carrying_capacity': 5e8,
                'initial_load': 1e3,
                'death_rate': 0.04
            },
            'p_aeruginosa': {
                'growth_rate': 0.3,
                'carrying_capacity': 1e9,
                'initial_load': 5e3,
                'death_rate': 0.03
            },
            'k_pneumoniae': {
                'growth_rate': 0.55,
                'carrying_capacity': 1.5e9,
                'initial_load': 8e3,
                'death_rate': 0.045
            }
        }
        
        # Configuration parameters for antibiotic PK/PD models
        self.antibiotic_params = {
            'default': {
                'half_life': 6.0,        # hours
                'peak_concentration': 30.0, # mg/L
                'mic': 1.0,              # Minimum Inhibitory Concentration (mg/L)
                'hill_coefficient': 1.5,  # Hill coefficient for dose-response
                'volume_distribution': 0.3, # L/kg
                'protein_binding': 0.3,   # fraction bound to proteins (inactive)
                'elimination_rate': 0.12   # per hour
            },
            # Parameters for specific antibiotics
            'vancomycin': {
                'half_life': 6.0,
                'peak_concentration': 25.0,
                'mic': {'s_aureus': 1.0, 'e_coli': 2.0, 'p_aeruginosa': 8.0, 'default': 2.0},
                'hill_coefficient': 1.2,
                'volume_distribution': 0.7,
                'protein_binding': 0.55,
                'elimination_rate': 0.115
            },
            'ceftriaxone': {
                'half_life': 8.0,
                'peak_concentration': 80.0,
                'mic': {'e_coli': 0.1, 's_aureus': 4.0, 'p_aeruginosa': 8.0, 'default': 1.0},
                'hill_coefficient': 1.3,
                'volume_distribution': 0.18,
                'protein_binding': 0.95,
                'elimination_rate': 0.086
            },
            'ciprofloxacin': {
                'half_life': 4.0,
                'peak_concentration': 4.0,
                'mic': {'e_coli': 0.015, 'p_aeruginosa': 0.5, 's_aureus': 1.0, 'default': 0.5},
                'hill_coefficient': 1.8,
                'volume_distribution': 2.5,
                'protein_binding': 0.2,
                'elimination_rate': 0.17
            },
            'piperacillin': {
                'half_life': 1.0,
                'peak_concentration': 200.0,
                'mic': {'e_coli': 2.0, 'p_aeruginosa': 8.0, 's_aureus': 4.0, 'default': 4.0},
                'hill_coefficient': 2.0,
                'volume_distribution': 0.18,
                'protein_binding': 0.3,
                'elimination_rate': 0.69
            }
        }
        
        # Parameters for immune system response
        self.immune_params = {
            'baseline_clearance': 0.02,      # per hour
            'max_response': 0.2,             # per hour
            'ec50': 1e6,                     # bacterial load at half-maximal response
            'response_delay': 24,            # hours for full immune response
            'hill_coefficient': 1.0          # steepness of immune response
        }
        
        # Indicators of treatment success
        self.treatment_success_thresholds = {
            'bacterial_reduction': 0.99,     # 99% reduction in bacterial load
            'clearance_load': 1e3,           # bacterial load below this is "cleared"
            'time_to_clearance': 72,         # clearance within this many hours is "success"
            'rebound_threshold': 1e5         # rebound above this level is "failure"
        }
        
        # Initialize logging
        self._log_initialization()
    
    def _log_initialization(self):
        """Log initialization information."""
        print("Initialized InfectionTreatmentModeler")
        print(f"- Pathogen models: {len(self.bacterial_growth_params)} different pathogens")
        print(f"- Antimicrobial models: {len(self.antibiotic_params)} different antibiotics")
    
    # =========================================================================
    # Infection Dynamics Models
    # =========================================================================
    
    def bacterial_growth_model(self, t, N, params=None):
        """
        Logistic bacterial growth model with natural death.
        
        Parameters:
        -----------
        t : float
            Time point
        N : float
            Bacterial population size (CFU/mL)
        params : dict, optional
            Model parameters (default: self.bacterial_growth_params['default'])
            
        Returns:
        --------
        float
            Rate of change of the bacterial population
        """
        if params is None:
            params = self.bacterial_growth_params['default']
            
        # Extract parameters
        r = params['growth_rate']        # Intrinsic growth rate
        K = params['carrying_capacity']  # Carrying capacity
        d = params['death_rate']         # Natural death rate
        
        # Logistic growth with natural death
        dN_dt = r * N * (1 - N / K) - d * N
        
        return dN_dt
    
    def simulate_bacterial_growth(self, duration_hours=72, time_step=0.1, pathogen='default'):
        """
        Simulate bacterial growth over time using logistic growth model.
        
        Parameters:
        -----------
        duration_hours : float, optional
            Duration of simulation in hours
        time_step : float, optional
            Time step in hours
        pathogen : str, optional
            Type of pathogen to simulate
            
        Returns:
        --------
        tuple
            (time_points, bacterial_load)
        """
        # Get parameters for the specified pathogen
        if pathogen in self.bacterial_growth_params:
            params = self.bacterial_growth_params[pathogen]
        else:
            params = self.bacterial_growth_params['default']
            warnings.warn(f"Pathogen '{pathogen}' not found. Using default parameters.")
        
        # Time points
        time_points = np.arange(0, duration_hours + time_step, time_step)
        
        # Initial bacterial load
        N0 = params['initial_load']
        
        # Define ODE system
        def system(t, N):
            return self.bacterial_growth_model(t, N, params)
        
        # Solve ODE
        solution = integrate.solve_ivp(
            system,
            [0, duration_hours],
            [N0],
            t_eval=time_points,
            method='RK45'
        )
        
        # Extract solution
        bacterial_load = solution.y[0]
        
        return time_points, bacterial_load
    
    def immune_response(self, bacterial_load, time_since_infection, params=None):
        """
        Model the immune system response to bacterial infection.
        
        Parameters:
        -----------
        bacterial_load : float
            Current bacterial load (CFU/mL)
        time_since_infection : float
            Time since infection started (hours)
        params : dict, optional
            Immune response parameters
            
        Returns:
        --------
        float
            Rate of bacterial clearance due to immune response
        """
        if params is None:
            params = self.immune_params
            
        # Extract parameters
        baseline = params['baseline_clearance']  # Baseline clearance rate
        max_resp = params['max_response']        # Maximum response
        ec50 = params['ec50']                    # Load at half-maximal response
        delay = params['response_delay']         # Response delay
        hill = params['hill_coefficient']        # Hill coefficient
        
        # Calculate time-dependent response scaling (delayed response)
        time_factor = min(time_since_infection / delay, 1.0)
        
        # Calculate load-dependent response (Hill function)
        load_factor = (bacterial_load**hill) / (ec50**hill + bacterial_load**hill)
        
        # Combine factors to get total immune clearance rate
        clearance_rate = baseline + time_factor * max_resp * load_factor
        
        return clearance_rate
    
    # =========================================================================
    # Pharmacokinetic/Pharmacodynamic (PK/PD) Models
    # =========================================================================
    
    def antibiotic_concentration(self, t, dose, administration_times, antibiotic='default'):
        """
        Calculate antibiotic concentration at time t using a one-compartment PK model.
        
        Parameters:
        -----------
        t : float
            Time point (hours)
        dose : float
            Dose amount (mg)
        administration_times : list
            List of times when doses were administered (hours)
        antibiotic : str, optional
            Type of antibiotic
            
        Returns:
        --------
        float
            Antibiotic concentration (mg/L)
        """
        # Get parameters for the specified antibiotic
        if antibiotic in self.antibiotic_params:
            params = self.antibiotic_params[antibiotic]
        else:
            params = self.antibiotic_params['default']
            warnings.warn(f"Antibiotic '{antibiotic}' not found. Using default parameters.")
        
        # Extract parameters
        k_elim = params['elimination_rate']  # Elimination rate constant
        vd = params['volume_distribution']   # Volume of distribution (L/kg)
        pb = params['protein_binding']       # Protein binding fraction
        
        # Assume standard 70 kg patient if not specified
        patient_weight = 70  # kg
        
        # Calculate bioavailable concentration
        conc = 0
        for admin_time in administration_times:
            if t >= admin_time:
                # One-compartment model with first-order elimination
                elapsed = t - admin_time
                conc += (dose / (vd * patient_weight)) * np.exp(-k_elim * elapsed)
        
        # Adjust for protein binding (only free drug is active)
        free_fraction = 1 - pb
        free_conc = conc * free_fraction
        
        return free_conc
    
    def antibiotic_effect(self, concentration, pathogen='default', antibiotic='default'):
        """
        Calculate antibiotic effect on bacterial death rate using an Emax model.
        
        Parameters:
        -----------
        concentration : float
            Antibiotic concentration (mg/L)
        pathogen : str, optional
            Type of pathogen
        antibiotic : str, optional
            Type of antibiotic
            
        Returns:
        --------
        float
            Death rate due to antibiotic effect (per hour)
        """
        # Get parameters for the specified antibiotic
        if antibiotic in self.antibiotic_params:
            params = self.antibiotic_params[antibiotic]
        else:
            params = self.antibiotic_params['default']
            warnings.warn(f"Antibiotic '{antibiotic}' not found. Using default parameters.")
        
        # Get MIC for the specified pathogen
        if isinstance(params['mic'], dict) and pathogen in params['mic']:
            mic = params['mic'][pathogen]
        elif isinstance(params['mic'], dict):
            mic = params['mic']['default']
        else:
            mic = params['mic']
        
        # Extract parameters
        hill = params['hill_coefficient']  # Hill coefficient
        
        # Maximum effect - typically 4-5x the natural death rate at high concentrations
        if pathogen in self.bacterial_growth_params:
            Emax = 5 * self.bacterial_growth_params[pathogen]['death_rate']
        else:
            Emax = 5 * self.bacterial_growth_params['default']['death_rate']
        
        # Calculate effect using Hill equation (Emax model)
        # Effect increases with concentration/MIC ratio
        if concentration <= 0:
            return 0
        
        effect = Emax * (concentration / mic)**hill / ((concentration / mic)**hill + 1)
        
        return effect
    
    def bacterial_dynamics_with_treatment(self, t, state, antibiotic_dose, dose_times, 
                                          pathogen='default', antibiotic='default'):
        """
        Model bacterial dynamics under antibiotic treatment.
        
        Parameters:
        -----------
        t : float
            Time point (hours)
        state : array-like
            Current state [bacterial_load, time_since_infection]
        antibiotic_dose : float
            Dose amount (mg)
        dose_times : list
            List of administration times (hours)
        pathogen : str, optional
            Type of pathogen
        antibiotic : str, optional
            Type of antibiotic
            
        Returns:
        --------
        list
            Rate of change of state variables
        """
        # Extract state variables
        N, time_since_infection = state
        
        # Get growth parameters for the specified pathogen
        if pathogen in self.bacterial_growth_params:
            params = self.bacterial_growth_params[pathogen]
        else:
            params = self.bacterial_growth_params['default']
        
        # Natural growth and death
        growth_term = self.bacterial_growth_model(t, N, params)
        
        # Immune response
        immune_clearance_rate = self.immune_response(N, time_since_infection)
        immune_term = -immune_clearance_rate * N
        
        # Antibiotic effect
        conc = self.antibiotic_concentration(t, antibiotic_dose, dose_times, antibiotic)
        abx_death_rate = self.antibiotic_effect(conc, pathogen, antibiotic)
        abx_term = -abx_death_rate * N
        
        # Combine all effects
        dN_dt = growth_term + immune_term + abx_term
        dt_infection = 1.0  # Time passes at normal rate
        
        return [dN_dt, dt_infection]
    
    def simulate_treatment(self, antibiotic, pathogen, dose, dosing_interval, 
                          duration_hours=168, initial_delay=0, patient_weight=70):
        """
        Simulate bacterial dynamics under antibiotic treatment.
        
        Parameters:
        -----------
        antibiotic : str
            Type of antibiotic
        pathogen : str
            Type of pathogen
        dose : float
            Dose per administration (mg)
        dosing_interval : float
            Hours between doses
        duration_hours : float, optional
            Simulation duration in hours
        initial_delay : float, optional
            Delay before first antibiotic dose (hours)
        patient_weight : float, optional
            Patient weight in kg
            
        Returns:
        --------
        dict
            Simulation results including time points, bacterial load, antibiotic concentration,
            and treatment metrics
        """
        # Adjust dose for patient weight if needed
        adjusted_dose = dose
        if "dose_per_kg" in self.antibiotic_params.get(antibiotic, {}):
            if self.antibiotic_params[antibiotic].get("dose_per_kg", False):
                adjusted_dose = dose * patient_weight
        
        # Generate administration times
        dose_times = []
        current_time = initial_delay
        while current_time < duration_hours:
            dose_times.append(current_time)
            current_time += dosing_interval
        
        # Time points for solving the ODE
        time_points = np.linspace(0, duration_hours, int(duration_hours * 10) + 1)
        
        # Initial state
        if pathogen in self.bacterial_growth_params:
            initial_load = self.bacterial_growth_params[pathogen]['initial_load']
        else:
            initial_load = self.bacterial_growth_params['default']['initial_load']
        
        initial_state = [initial_load, 0]  # [bacterial_load, time_since_infection]
        
        # Define the ODE system
        def system(t, state):
            return self.bacterial_dynamics_with_treatment(
                t, state, adjusted_dose, dose_times, pathogen, antibiotic
            )
        
        # Solve the ODE
        solution = integrate.solve_ivp(
            system,
            [0, duration_hours],
            initial_state,
            t_eval=time_points,
            method='RK45'
        )
        
        # Extract results
        times = solution.t
        bacterial_load = solution.y[0]
        
        # Calculate antibiotic concentration at each time point
        antibiotic_conc = np.array([
            self.antibiotic_concentration(t, adjusted_dose, dose_times, antibiotic)
            for t in times
        ])
        
        # Calculate treatment metrics
        metrics = self.calculate_treatment_metrics(
            times, bacterial_load, antibiotic_conc, pathogen, antibiotic
        )
        
        # Prepare results
        results = {
            'times': times,
            'bacterial_load': bacterial_load,
            'antibiotic_concentration': antibiotic_conc,
            'dose_times': np.array(dose_times),
            'metrics': metrics
        }
        
        return results
    
    def calculate_treatment_metrics(self, times, bacterial_load, antibiotic_conc, 
                                   pathogen, antibiotic):
        """
        Calculate metrics to evaluate treatment effectiveness.
        
        Parameters:
        -----------
        times : array-like
            Time points
        bacterial_load : array-like
            Bacterial load at each time point
        antibiotic_conc : array-like
            Antibiotic concentration at each time point
        pathogen : str
            Type of pathogen
        antibiotic : str
            Type of antibiotic
            
        Returns:
        --------
        dict
            Dictionary of treatment metrics
        """
        # Get MIC for the specified pathogen and antibiotic
        if antibiotic in self.antibiotic_params:
            params = self.antibiotic_params[antibiotic]
        else:
            params = self.antibiotic_params['default']
        
        if isinstance(params['mic'], dict) and pathogen in params['mic']:
            mic = params['mic'][pathogen]
        elif isinstance(params['mic'], dict):
            mic = params['mic']['default']
        else:
            mic = params['mic']
        
        # Initial and final bacterial load
        initial_load = bacterial_load[0]
        final_load = bacterial_load[-1]
        
        # Maximum bacterial load
        max_load = np.max(bacterial_load)
        
        # Load reduction (fold change)
        if initial_load > 0:
            load_reduction = initial_load / max(final_load, 1)
        else:
            load_reduction = 0
        
        # Time to x% reduction
        percent_reductions = [50, 90, 99]
        times_to_reduction = {}
        
        for percent in percent_reductions:
            target_load = initial_load * (1 - percent/100)
            idx = np.argmax(bacterial_load <= target_load)
            if idx > 0 and bacterial_load[idx] <= target_load:
                times_to_reduction[f"{percent}%"] = times[idx]
            else:
                times_to_reduction[f"{percent}%"] = None
        
        # Check for clearance (load below threshold)
        clearance_threshold = self.treatment_success_thresholds['clearance_load']
        clearance_indices = np.where(bacterial_load < clearance_threshold)[0]
        
        if len(clearance_indices) > 0:
            time_to_clearance = times[clearance_indices[0]]
            is_cleared = True
        else:
            time_to_clearance = None
            is_cleared = False
        
        # Check for rebound (load increases after initial decrease)
        min_idx = np.argmin(bacterial_load)
        if min_idx < len(bacterial_load) - 1:
            min_load = bacterial_load[min_idx]
            subsequent_max = np.max(bacterial_load[min_idx:])
            rebound_factor = subsequent_max / max(min_load, 1) if min_load > 0 else 0
            has_rebound = rebound_factor > 2  # Arbitrary threshold
        else:
            rebound_factor = 1
            has_rebound = False
        
        # Calculate PK/PD indices
        # Time above MIC (percent of dosing interval)
        time_above_mic = np.sum(antibiotic_conc > mic) / len(antibiotic_conc) * 100
        
        # Peak/MIC ratio
        peak_conc = np.max(antibiotic_conc)
        peak_mic_ratio = peak_conc / mic if mic > 0 else float('inf')
        
        # AUC/MIC ratio (roughly estimated)
        auc = np.trapz(antibiotic_conc, times)
        auc_mic_ratio = auc / mic if mic > 0 else float('inf')
        
        # Overall treatment success
        success_threshold = self.treatment_success_thresholds['bacterial_reduction']
        max_time = self.treatment_success_thresholds['time_to_clearance']
        is_successful = (load_reduction >= success_threshold and 
                        (time_to_clearance is not None and time_to_clearance <= max_time))
        
        # Compile metrics
        metrics = {
            'initial_load': initial_load,
            'final_load': final_load,
            'max_load': max_load,
            'load_reduction': load_reduction,
            'time_to_reduction': times_to_reduction,
            'time_to_clearance': time_to_clearance,
            'is_cleared': is_cleared,
            'rebound_factor': rebound_factor,
            'has_rebound': has_rebound,
            'time_above_mic': time_above_mic,
            'peak_mic_ratio': peak_mic_ratio,
            'auc_mic_ratio': auc_mic_ratio,
            'is_successful': is_successful
        }
        
        return metrics
    
    # =========================================================================
    # Treatment Optimization
    # =========================================================================
    
    def optimize_dosing_regimen(self, antibiotic, pathogen, dose_range, interval_range, 
                               duration_hours=168, initial_delay=0, patient_weight=70,
                               objective='clearance'):
        """
        Optimize antibiotic dosing regimen to maximize treatment effectiveness.
        
        Parameters:
        -----------
        antibiotic : str
            Type of antibiotic
        pathogen : str
            Type of pathogen
        dose_range : tuple
            Range of possible doses (min, max) in mg
        interval_range : tuple
            Range of possible dosing intervals (min, max) in hours
        duration_hours : float, optional
            Simulation duration in hours
        initial_delay : float, optional
            Delay before first antibiotic dose (hours)
        patient_weight : float, optional
            Patient weight in kg
        objective : str, optional
            Optimization objective ('clearance', 'reduction', or 'time_above_mic')
            
        Returns:
        --------
        dict
            Optimal dosing regimen and treatment metrics
        """
        # Define the objective function to minimize
        def objective_function(params):
            dose, interval = params
            
            # Check if the dose and interval are within bounds
            if (dose < dose_range[0] or dose > dose_range[1] or
                interval < interval_range[0] or interval > interval_range[1]):
                return float('inf')
            
            # Simulate treatment with the current parameters
            results = self.simulate_treatment(
                antibiotic, pathogen, dose, interval, 
                duration_hours, initial_delay, patient_weight
            )
            
            # Extract metrics
            metrics = results['metrics']
            
            # Define the objective value based on the chosen objective
            if objective == 'clearance':
                # Minimize time to clearance (set high value if not cleared)
                if metrics['is_cleared']:
                    obj_value = metrics['time_to_clearance']
                else:
                    obj_value = duration_hours * 2  # Penalize for not clearing
            
            elif objective == 'reduction':
                # Maximize load reduction (minimize negative log reduction)
                obj_value = -np.log10(max(metrics['load_reduction'], 1))
            
            elif objective == 'time_above_mic':
                # Maximize time above MIC (minimize negative value)
                obj_value = -metrics['time_above_mic']
            
            else:
                raise ValueError(f"Unknown objective: {objective}")
            
            return obj_value
        
        # Initial guess (middle of ranges)
        initial_guess = [
            (dose_range[0] + dose_range[1]) / 2,
            (interval_range[0] + interval_range[1]) / 2
        ]
        
        # Run optimization
        result = optimize.minimize(
            objective_function,
            initial_guess,
            method='Nelder-Mead',
            bounds=[dose_range, interval_range],
            options={'maxiter': 50}
        )
        
        # Extract optimal parameters
        optimal_dose, optimal_interval = result.x
        
        # Simulate treatment with optimal parameters
        optimal_results = self.simulate_treatment(
            antibiotic, pathogen, optimal_dose, optimal_interval,
            duration_hours, initial_delay, patient_weight
        )
        
        # Return optimal parameters and results
        optimization_results = {
            'optimal_dose': optimal_dose,
            'optimal_interval': optimal_interval,
            'objective_value': result.fun,
            'convergence': result.success,
            'iterations': result.nit,
            'treatment_results': optimal_results
        }
        
        return optimization_results
    
    def evaluate_multiple_regimens(self, antibiotic, pathogen, regimens, 
                                 duration_hours=168, initial_delay=0, patient_weight=70):
        """
        Evaluate and compare multiple treatment regimens.
        
        Parameters:
        -----------
        antibiotic : str
            Type of antibiotic
        pathogen : str
            Type of pathogen
        regimens : list of dict
            List of treatment regimens, each with 'dose' and 'interval' keys
        duration_hours : float, optional
            Simulation duration in hours
        initial_delay : float, optional
            Delay before first antibiotic dose (hours)
        patient_weight : float, optional
            Patient weight in kg
            
        Returns:
        --------
        dict
            Comparison of treatment regimens
        """
        results = []
        
        for i, regimen in enumerate(regimens):
            dose = regimen['dose']
            interval = regimen['interval']
            
            # Simulate treatment
            sim_results = self.simulate_treatment(
                antibiotic, pathogen, dose, interval,
                duration_hours, initial_delay, patient_weight
            )
            
            # Add regimen details to metrics
            metrics = sim_results['metrics'].copy()
            metrics['regimen_id'] = i
            metrics['dose'] = dose
            metrics['interval'] = interval
            metrics['simulation_results'] = sim_results
            
            results.append(metrics)
        
        # Find the best regimen based on different criteria
        best_regimens = {}
        
        # Best for bacterial clearance
        clearance_metric = [
            (i, r['time_to_clearance'] if r['is_cleared'] else float('inf'))
            for i, r in enumerate(results)
        ]
        best_clearance_idx = min(clearance_metric, key=lambda x: x[1])[0]
        best_regimens['clearance'] = best_clearance_idx
        
        # Best for bacterial reduction
        reduction_metric = [(i, r['load_reduction']) for i, r in enumerate(results)]
        best_reduction_idx = max(reduction_metric, key=lambda x: x[1])[0]
        best_regimens['reduction'] = best_reduction_idx
        
        # Best for time above MIC
        time_above_mic_metric = [(i, r['time_above_mic']) for i, r in enumerate(results)]
        best_time_above_mic_idx = max(time_above_mic_metric, key=lambda x: x[1])[0]
        best_regimens['time_above_mic'] = best_time_above_mic_idx
        
        # Best overall (simple weighted score)
        def calculate_score(result):
            score = 0
            if result['is_cleared']:
                score += 10
                score += max(0, 48 - result['time_to_clearance']/10)
            score += np.log10(max(result['load_reduction'], 1)) * 3
            score += result['time_above_mic'] / 20
            return score
        
        scores = [(i, calculate_score(r)) for i, r in enumerate(results)]
        best_overall_idx = max(scores, key=lambda x: x[1])[0]
        best_regimens['overall'] = best_overall_idx
        
        # Return comparison results
        comparison = {
            'all_regimens': results,
            'best_regimens': best_regimens
        }
        
        return comparison
    
    # =========================================================================
    # Integration with MIMIC-III Data
    # =========================================================================
    