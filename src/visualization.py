"""
BioDynamICS - Clinical Visualization Module

This module creates visualizations of physiological data and stability metrics
to support clinical interpretation and decision-making.

Author: Alexander Clarke
Date: March 14, 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import warnings

class ClinicalVisualizer:
    """
    Creates visualizations of patient physiological data and stability metrics.
    """
    
    def __init__(self):
        """Initialize the ClinicalVisualizer."""
        # Set default style
        plt.style.use('seaborn-v0_8-whitegrid')
        
        # Custom color palettes
        self.organ_colors = {
            'cardiovascular': '#FF5E5B',  # Red
            'respiratory': '#5282FF',     # Blue
            'renal': '#FFBF42',           # Orange
            'hepatic': '#8C52FF',         # Purple
            'hematologic': '#FF42A1',     # Pink
            'metabolic': '#75D701',       # Green
            'neurologic': '#01D7C7',      # Teal
            'infectious': '#FF8A47'       # Coral
        }
        
        # Define normal reference ranges for visualization
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
            'lactate': (0.5, 2.2),                # mmol/L
            'ph': (7.35, 7.45),                   # pH units
        }
        
        # Create a custom diverging colormap for stability (blue = stable, red = unstable)
        self.stability_cmap = LinearSegmentedColormap.from_list(
            "stability", 
            [(0, '#2979FF'), (0.5, '#FFFFFF'), (1, '#FF5252')]
        )
        
        # Print initialization message
        print("Initialized ClinicalVisualizer")
    
    def plot_vital_sign_timeline(self, timeline, vital_sign, figsize=(12, 6), show_range=True):
        """
        Plot a single vital sign over time.
        
        Parameters:
        -----------
        timeline : pandas.DataFrame
            Patient timeline DataFrame with 'measurement_time' column
        vital_sign : str
            Name of the vital sign to plot
        figsize : tuple, optional
            Figure size
        show_range : bool, optional
            Whether to show normal range as shaded area
            
        Returns:
        --------
        matplotlib.figure.Figure
            The created figure
        """
        if vital_sign not in self.reference_ranges:
            print(f"Warning: No reference range for {vital_sign}, won't show normal range")
            show_range = False
        
        # Filter data for the vital sign
        if 'itemid' in timeline.columns and 'valuenum' in timeline.columns:
            # Need to map itemid to vital sign name
            # This would require a mapping from vital_sign to itemids
            print("Please provide filtered data for this vital sign")
            return None
        elif vital_sign in timeline.columns:
            data = timeline[[vital_sign, 'measurement_time']].copy()
            data = data.dropna(subset=[vital_sign])
        else:
            print(f"Error: {vital_sign} not found in timeline")
            return None
        
        # Sort by time
        data = data.sort_values('measurement_time')
        
        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot the vital sign
        ax.plot(data['measurement_time'], data[vital_sign], 
                marker='o', linestyle='-', color='#2979FF', 
                markersize=5, linewidth=1.5)
        
        # Add reference range if requested
        if show_range and vital_sign in self.reference_ranges:
            lower, upper = self.reference_ranges[vital_sign]
            ax.axhspan(lower, upper, alpha=0.2, color='green', label='Normal Range')
            
            # Add lines at boundaries
            ax.axhline(lower, linestyle='--', color='green', alpha=0.7, linewidth=1)
            ax.axhline(upper, linestyle='--', color='green', alpha=0.7, linewidth=1)
        
        # Format the plot
        ax.set_title(f"{vital_sign.replace('_', ' ').title()} Over Time", fontsize=14)
        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel(vital_sign.replace('_', ' ').title(), fontsize=12)
        
        # Format date axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
        plt.xticks(rotation=45)
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        return fig
    
    def plot_multi_vital_timeline(self, timeline, vital_signs, figsize=(14, 10)):
        """
        Plot multiple vital signs on separate subplots with aligned time axis.
        
        Parameters:
        -----------
        timeline : pandas.DataFrame
            Patient timeline DataFrame with 'measurement_time' column
        vital_signs : list
            List of vital sign names to plot
        figsize : tuple, optional
            Figure size
            
        Returns:
        --------
        matplotlib.figure.Figure
            The created figure
        """
        # Filter and prepare data
        data = timeline.copy()
        if not isinstance(data.index, pd.DatetimeIndex) and 'measurement_time' in data.columns:
            data['measurement_time'] = pd.to_datetime(data['measurement_time'])
        
        # Create figure with subplots
        fig, axes = plt.subplots(len(vital_signs), 1, figsize=figsize, sharex=True)
        
        # Handle case of single vital sign
        if len(vital_signs) == 1:
            axes = [axes]
            
        # Plot each vital sign
        for i, vital in enumerate(vital_signs):
            ax = axes[i]
            
            # Check if vital sign exists in data
            if vital in data.columns:
                # Filter out NaN values
                vital_data = data[['measurement_time', vital]].dropna(subset=[vital])
                
                # Plot data
                ax.plot(vital_data['measurement_time'], vital_data[vital],
                       marker='o', linestyle='-', markersize=4)
                
                # Add reference range if available
                if vital in self.reference_ranges:
                    lower, upper = self.reference_ranges[vital]
                    ax.axhspan(lower, upper, alpha=0.2, color='green')
                    ax.axhline(lower, linestyle='--', color='green', alpha=0.7, linewidth=1)
                    ax.axhline(upper, linestyle='--', color='green', alpha=0.7, linewidth=1)
                
                # Set labels
                ax.set_ylabel(vital.replace('_', ' ').title())
                
                # Add grid
                ax.grid(True, linestyle='--', alpha=0.7)
            else:
                ax.text(0.5, 0.5, f"{vital} not available",
                       horizontalalignment='center',
                       verticalalignment='center',
                       transform=ax.transAxes)
        
        # Format date axis on the bottom subplot
        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
        axes[-1].set_xlabel('Time')
        plt.xticks(rotation=45)
        
        # Add title
        fig.suptitle('Vital Signs Timeline', fontsize=16, y=0.98)
        
        plt.tight_layout()
        fig.subplots_adjust(top=0.95)
        
        return fig
    
    def plot_organ_system_radar(self, stability_report, figsize=(10, 8)):
        """
        Create a radar plot showing organ system status.
        
        Parameters:
        -----------
        stability_report : dict
            Stability report from PhysiologicalSignalProcessor
        figsize : tuple, optional
            Figure size
            
        Returns:
        --------
        matplotlib.figure.Figure
            The created figure
        """
        # Extract organ system scores
        if 'organ_system_summary' not in stability_report:
            print("Error: No organ system summary in stability report")
            return None
            
        organ_summary = stability_report['organ_system_summary']
        
        # Get systems and scores
        systems = []
        scores = []
        colors = []
        
        for system, data in organ_summary.items():
            systems.append(system.replace('_', ' ').title())
            scores.append(data['score'])
            colors.append(self.organ_colors.get(system, '#999999'))
        
        # If no systems, return None
        if not systems:
            print("No organ systems available for radar plot")
            return None
        
        # Create radar plot
        fig = plt.figure(figsize=figsize)
        
        # Calculate angles for each axis
        angles = np.linspace(0, 2*np.pi, len(systems), endpoint=False).tolist()
        
        # Close the polygon
        angles.append(angles[0])
        scores.append(scores[0])
        colors.append(colors[0])
        
        # Repeat labels and set up polar plot
        systems.append(systems[0])
        
        ax = fig.add_subplot(111, polar=True)
        
        # Plot the scores
        ax.plot(angles, scores, 'o-', linewidth=2, color='#FF5252')
        ax.fill(angles, scores, alpha=0.25, color='#FF5252')
        
        # Set labels
        ax.set_thetagrids(np.degrees(angles[:-1]), systems[:-1])
        
        # Add grid
        ax.grid(True)
        
        # Set y-limits (scores should be between 0 and 2 typically)
        ax.set_ylim(0, max(2, max(scores) * 1.2))
        
        # Add title
        plt.title('Organ System Status', size=15)
        
        # Add score levels
        severity_levels = [0.5, 1.0, 1.5]
        severity_labels = ['Mild', 'Moderate', 'Severe']
        severity_colors = ['#FFECB3', '#FFD54F', '#FF8F00']
        
        for level, label, color in zip(severity_levels, severity_labels, severity_colors):
            circle = plt.Circle((0, 0), level, transform=ax.transData._b, 
                               fill=True, alpha=0.1, color=color)
            ax.add_artist(circle)
            
            # Add text label at right side of the circle
            ax.text(0, level, f' {label}', verticalalignment='center')
        
        plt.tight_layout()
        return fig
    
    def plot_allostatic_load_trend(self, stability_over_time, figsize=(12, 6)):
        """
        Plot allostatic load trends over time.
        
        Parameters:
        -----------
        stability_over_time : pandas.DataFrame
            DataFrame with stability metrics over time
        figsize : tuple, optional
            Figure size
            
        Returns:
        --------
        matplotlib.figure.Figure
            The created figure
        """
        if stability_over_time is None or stability_over_time.empty:
            print("Error: No stability data provided")
            return None
            
        if 'allostatic_load' not in stability_over_time.columns:
            print("Error: No allostatic load in stability data")
            return None
            
        if 'window_start' not in stability_over_time.columns:
            print("Error: No window timestamps in stability data")
            return None
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot allostatic load
        ax.plot(stability_over_time['window_start'], 
                stability_over_time['allostatic_load'],
                marker='o', linestyle='-', color='#FF5252', 
                markersize=8, linewidth=2)
        
        # Add severity zones
        ax.axhspan(0, 0.5, alpha=0.1, color='green', label='Normal')
        ax.axhspan(0.5, 1.0, alpha=0.1, color='yellow', label='Mild Stress')
        ax.axhspan(1.0, 1.5, alpha=0.1, color='orange', label='Moderate Stress')
        ax.axhspan(1.5, 3.0, alpha=0.1, color='red', label='Severe Stress')
        
        # Add trend line if we have enough points
        if len(stability_over_time) > 2:
            try:
                # Calculate trend
                x = np.arange(len(stability_over_time))
                y = stability_over_time['allostatic_load'].values
                
                # Remove NaN values for regression
                mask = ~np.isnan(y)
                if np.sum(mask) > 1:
                    from scipy import stats
                    slope, intercept, r_value, p_value, _ = stats.linregress(x[mask], y[mask])
                    
                    # Plot trend line
                    trend_x = np.array([x[mask].min(), x[mask].max()])
                    trend_y = intercept + slope * trend_x
                    trend_dates = [stability_over_time.iloc[trend_x[0]]['window_start'],
                                 stability_over_time.iloc[trend_x[1]]['window_start']]
                    
                    ax.plot(trend_dates, trend_y, '--', color='black', 
                           linewidth=1.5, label=f'Trend (R²={r_value**2:.2f})')
                    
                    # Highlight trend direction
                    if slope > 0.05:
                        ax.text(0.02, 0.98, '⚠️ Worsening', transform=ax.transAxes,
                               fontsize=12, verticalalignment='top',
                               bbox=dict(boxstyle='round', facecolor='#FFCDD2', alpha=0.5))
                    elif slope < -0.05:
                        ax.text(0.02, 0.98, '✓ Improving', transform=ax.transAxes,
                               fontsize=12, verticalalignment='top',
                               bbox=dict(boxstyle='round', facecolor='#C8E6C9', alpha=0.5))
            except Exception as e:
                print(f"Error calculating trend: {e}")
        
        # Format axes
        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel('Allostatic Load', fontsize=12)
        ax.set_title('Physiological Stress Over Time', fontsize=14)
        
        # Format date axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
        plt.xticks(rotation=45)
        
        # Add grid and legend
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(loc='upper left')
        
        plt.tight_layout()
        return fig
    
    def plot_phase_portrait(self, timeline, x_measure, y_measure, z_measure=None, 
                           figsize=(10, 8), alpha=0.7):
        """
        Create a phase portrait (2D or 3D) to visualize physiological trajectories.
        
        Parameters:
        -----------
        timeline : pandas.DataFrame
            Patient timeline DataFrame
        x_measure : str
            Measurement for x-axis
        y_measure : str
            Measurement for y-axis
        z_measure : str, optional
            Measurement for z-axis (if provided, makes a 3D plot)
        figsize : tuple, optional
            Figure size
        alpha : float, optional
            Transparency of points
            
        Returns:
        --------
        matplotlib.figure.Figure
            The created figure
        """
        # Check if measures exist in the timeline
        required_measures = [x_measure, y_measure]
        if z_measure is not None:
            required_measures.append(z_measure)
            
        missing_measures = [m for m in required_measures if m not in timeline.columns]
        if missing_measures:
            print(f"Error: Missing measures in timeline: {', '.join(missing_measures)}")
            return None
        
        # Filter to rows with all required measures
        data = timeline[required_measures + ['measurement_time']].dropna()
        
        if len(data) < 5:
            print("Error: Not enough data points after filtering")
            return None
        
        # Create figure
        fig = plt.figure(figsize=figsize)
        
        # Determine if 2D or 3D
        if z_measure is None:
            # 2D phase portrait
            ax = fig.add_subplot(111)
            
            # Plot trajectory
            points = ax.scatter(data[x_measure], data[y_measure], 
                              c=range(len(data)), cmap='viridis', 
                              alpha=alpha, s=30)
            
            # Connect points with lines to show trajectory
            ax.plot(data[x_measure], data[y_measure], 
                   '-', color='gray', alpha=0.3, linewidth=1)
            
            # Mark start and end points
            ax.scatter(data[x_measure].iloc[0], data[y_measure].iloc[0], 
                      color='green', s=100, label='Start', zorder=5)
            ax.scatter(data[x_measure].iloc[-1], data[y_measure].iloc[-1], 
                      color='red', s=100, label='End', zorder=5)
            
            # Add reference ranges if available
            if x_measure in self.reference_ranges:
                ax.axvspan(self.reference_ranges[x_measure][0], 
                         self.reference_ranges[x_measure][1],
                         alpha=0.1, color='green')
            
            if y_measure in self.reference_ranges:
                ax.axhspan(self.reference_ranges[y_measure][0], 
                         self.reference_ranges[y_measure][1],
                         alpha=0.1, color='green')
            
            # Set labels
            ax.set_xlabel(x_measure.replace('_', ' ').title(), fontsize=12)
            ax.set_ylabel(y_measure.replace('_', ' ').title(), fontsize=12)
            
            # Add colorbar to show time progression
            cbar = plt.colorbar(points)
            cbar.set_label('Time Progression')
            
            # Add grid and legend
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.legend()
        else:
            # 3D phase portrait
            from mpl_toolkits.mplot3d import Axes3D
            ax = fig.add_subplot(111, projection='3d')
            
            # Plot trajectory
            points = ax.scatter(data[x_measure], data[y_measure], data[z_measure],
                              c=range(len(data)), cmap='viridis', 
                              alpha=alpha, s=30)
            
            # Connect points with lines
            ax.plot(data[x_measure], data[y_measure], data[z_measure],
                   '-', color='gray', alpha=0.3, linewidth=1)
            
            # Mark start and end points
            ax.scatter(data[x_measure].iloc[0], data[y_measure].iloc[0], data[z_measure].iloc[0],
                      color='green', s=100, label='Start')
            ax.scatter(data[x_measure].iloc[-1], data[y_measure].iloc[-1], data[z_measure].iloc[-1],
                      color='red', s=100, label='End')
            
            # Set labels
            ax.set_xlabel(x_measure.replace('_', ' ').title(), fontsize=12)
            ax.set_ylabel(y_measure.replace('_', ' ').title(), fontsize=12)
            ax.set_zlabel(z_measure.replace('_', ' ').title(), fontsize=12)
            
            # Add colorbar
            cbar = plt.colorbar(points)
            cbar.set_label('Time Progression')
            
            # Add legend
            ax.legend()
        
        # Set title
        if z_measure is None:
            plt.title(f'Physiological Phase Portrait: {x_measure.title()} vs {y_measure.title()}',
                     fontsize=14)
        else:
            plt.title(f'3D Phase Portrait: {x_measure.title()}, {y_measure.title()}, {z_measure.title()}',
                     fontsize=14)
        
        plt.tight_layout()
        return fig
    
    def plot_organ_system_heatmap(self, stability_over_time, figsize=(14, 8)):
        """
        Create a heatmap showing organ system status over time.
        
        Parameters:
        -----------
        stability_over_time : pandas.DataFrame
            DataFrame with stability metrics over time
        figsize : tuple, optional
            Figure size
            
        Returns:
        --------
        matplotlib.figure.Figure
            The created figure
        """
        if stability_over_time is None or stability_over_time.empty:
            print("Error: No stability data provided")
            return None
        
        # Get columns for organ systems
        organ_cols = [col for col in stability_over_time.columns 
                     if col.endswith('_score') and 
                     col.split('_')[0] in self.organ_colors]
        
        if not organ_cols:
            print("Error: No organ system scores found in data")
            return None
        
        # Prepare data for heatmap
        heatmap_data = stability_over_time[['window_start'] + organ_cols].copy()
        
        # Convert timestamps to formatted strings
        heatmap_data['window_start'] = heatmap_data['window_start'].dt.strftime('%m/%d %H:%M')
        
        # Set window_start as index
        heatmap_data = heatmap_data.set_index('window_start')
        
        # Rename columns to readable names
        heatmap_data.columns = [col.split('_')[0].title() for col in heatmap_data.columns]
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create heatmap
        sns.heatmap(heatmap_data.T, annot=True, cmap=self.stability_cmap,
                   linewidths=0.5, ax=ax, vmin=0, vmax=2, fmt='.2f')
        
        # Set labels
        ax.set_xlabel('Time Window', fontsize=12)
        ax.set_ylabel('Organ System', fontsize=12)
        
        # Set title
        ax.set_title('Organ System Status Over Time', fontsize=14)
        
        plt.tight_layout()
        return fig
    
    def create_patient_dashboard(self, patient_timeline, stability_report, vital_signs=None,
                                figsize=(16, 20)):
        """
        Create a comprehensive patient dashboard with multiple visualizations.
        
        Parameters:
        -----------
        patient_timeline : dict
            Patient timeline dictionary with 'info' and 'timeline' keys
        stability_report : dict
            Stability report from PhysiologicalSignalProcessor
        vital_signs : list, optional
            List of vital signs to include (defaults to standard set)
        figsize : tuple, optional
            Figure size
            
        Returns:
        --------
        matplotlib.figure.Figure
            The created figure
        """
        if not patient_timeline or 'timeline' not in patient_timeline:
            print("Error: Invalid patient timeline")
            return None
            
        if not stability_report:
            print("Error: No stability report provided")
            return None
            
        # Use default vital signs if not specified
        if vital_signs is None:
            vital_signs = ['heart_rate', 'respiratory_rate', 'sbp', 'dbp', 'temperature', 'o2_saturation']
            
        # Filter vital signs to those available in the data
        timeline_df = patient_timeline['timeline']
        available_vitals = [v for v in vital_signs if v in timeline_df.columns]
        
        if not available_vitals:
            print("Warning: None of the specified vital signs are available")
            
        # Create figure with GridSpec for flexible layout
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(12, 10, figure=fig)
        
        # Patient info header
        ax_info = fig.add_subplot(gs[0:1, :])
        patient_id = patient_timeline.get('info', {}).get('subject_id', 'Unknown')
        patient_gender = patient_timeline.get('info', {}).get('gender', 'Unknown')
        
        ax_info.text(0.5, 0.5, 
                   f"Patient ID: {patient_id} | Gender: {patient_gender} | " + 
                   f"Allostatic Load: {stability_report.get('overall_results', {}).get('allostatic_load', 'N/A'):.2f}",
                   ha='center', va='center', fontsize=14,
                   bbox=dict(boxstyle='round', facecolor='#E3F2FD', alpha=0.5))
        ax_info.axis('off')
        
        # Organ system radar plot
        ax_radar = fig.add_subplot(gs[1:5, 0:5])
        self.plot_organ_system_radar(stability_report)
        plt.close()  # Close the independent figure
        ax_radar = fig.add_subplot(gs[1:5, 0:5], polar=True)
        
        # Extract organ system scores
        organ_summary = stability_report.get('organ_system_summary', {})
        systems = []
        scores = []
        
        for system, data in organ_summary.items():
            systems.append(system.replace('_', ' ').title())
            scores.append(data['score'])
            
        # If no systems, leave plot empty
        if systems:
            # Calculate angles for radar plot
            angles = np.linspace(0, 2*np.pi, len(systems), endpoint=False).tolist()
            
            # Close the polygon
            angles.append(angles[0])
            scores.append(scores[0])
            systems.append(systems[0])
            
            # Plot the scores
            ax_radar.plot(angles, scores, 'o-', linewidth=2, color='#FF5252')
            ax_radar.fill(angles, scores, alpha=0.25, color='#FF5252')
            
            # Set labels
            ax_radar.set_thetagrids(np.degrees(angles[:-1]), systems[:-1])
            
            # Add grid
            ax_radar.grid(True)
            
            # Set y-limits
            ax_radar.set_ylim(0, max(2, max(scores) * 1.2))
            
            # Add title
            ax_radar.set_title('Organ System Status', size=15)
            
            # Add severity levels
            severity_levels = [0.5, 1.0, 1.5]
            for level in severity_levels:
                circle = plt.Circle((0, 0), level, transform=ax_radar.transData._b, 
                                  fill=True, alpha=0.1, color='#FFD54F')
                ax_radar.add_artist(circle)
        
        # Allostatic load trend
        ax_trend = fig.add_subplot(gs[1:3, 5:])
        if 'stability_over_time' in stability_report and not stability_report['stability_over_time'].empty:
            stability_df = stability_report['stability_over_time']
            
            # Plot allostatic load
            ax_trend.plot(stability_df['window_start'], 
                       stability_df['allostatic_load'],
                       marker='o', linestyle='-', color='#FF5252', 
                       markersize=6, linewidth=1.5)
            
            # Add severity zones
            ax_trend.axhspan(0, 0.5, alpha=0.1, color='green')
            ax_trend.axhspan(0.5, 1.0, alpha=0.1, color='yellow')
            ax_trend.axhspan(1.0, 1.5, alpha=0.1, color='orange')
            ax_trend.axhspan(1.5, 3.0, alpha=0.1, color='red')
            
            # Format date axis
            ax_trend.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
            plt.setp(ax_trend.xaxis.get_majorticklabels(), rotation=45)
            
            # Set labels
            ax_trend.set_title('Physiological Stress Over Time', fontsize=12)
            ax_trend.set_ylabel('Allostatic Load')
            
            # Add grid
            ax_trend.grid(True, linestyle='--', alpha=0.7)
        else:
            ax_trend.text(0.5, 0.5, "Insufficient data for trend analysis",
                       ha='center', va='center', fontsize=12)
            ax_trend.axis('off')
        
        # Vital signs timeline
        ax_vitals = []
        for i, vital in enumerate(available_vitals[:4]):  # Show up to 4 vital signs
            ax_vital = fig.add_subplot(gs[3+i:4+i, 5:])
            ax_vitals.append(ax_vital)
            
            # Filter data for this vital sign
            vital_data = timeline_df[['measurement_time', vital]].dropna(subset=[vital])
            
            if len(vital_data) > 0:
                # Plot vital sign
                ax_vital.plot(vital_data['measurement_time'], vital_data[vital],
                           marker='o', linestyle='-', markersize=3)
                
                # Add reference range if available
                if vital in self.reference_ranges:
                    lower, upper = self.reference_ranges[vital]
                    ax_vital.axhspan(lower, upper, alpha=0.1, color='green')
                
                # Format x-axis (only for bottom plot)
                if i == len(available_vitals[:4]) - 1:
                    ax_vital.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
                    plt.setp(ax_vital.xaxis.get_majorticklabels(), rotation=45)
                else:
                    # Hide x-tick labels for all but the bottom plot
                    plt.setp(ax_vital.get_xticklabels(), visible=False)
                
                # Set labels
                ax_vital.set_ylabel(vital.replace('_', ' ').title())
                
                # Add grid
                ax_vital.grid(True, linestyle='--', alpha=0.5)
            else:
                ax_vital.text(0.5, 0.5, f"No data for {vital}",
                           ha='center', va='center', fontsize=10)
                ax_vital.axis('off')
        
        # Critical values table
        ax_critical = fig.add_subplot(gs[5:7, 0:5])
        critical_values = stability_report.get('critical_values', [])
        
        if critical_values:
            # Create table data
            table_data = [['Measurement', 'Value', 'Reference', 'Deviation']]
            for cv in critical_values[:5]:  # Show at most 5
                measurement = cv['measurement'].replace('_', ' ').title()
                value = f"{cv['value']:.1f}"
                ref_range = f"{cv['reference_range'][0]:.1f} - {cv['reference_range'][1]:.1f}"
                deviation = f"{cv['percent_deviation']:.1f}%"
                table_data.append([measurement, value, ref_range, deviation])
            
            # Create table
            table = ax_critical.table(cellText=table_data, 
                                    loc='center', 
                                    cellLoc='center',
                                    colWidths=[0.3, 0.2, 0.3, 0.2])
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 1.5)
            
            # Style table - header row
            for j in range(len(table_data[0])):
                table[(0, j)].set_facecolor('#E3F2FD')
                table[(0, j)].set_text_props(weight='bold')
            
            # Style table - highlight deviations
            for i in range(1, len(table_data)):
                # Make the deviation cell red for large deviations
                try:
                    dev_value = float(table_data[i][3].replace('%', ''))
                    if dev_value > 30:
                        table[(i, 3)].set_facecolor('#FFCDD2')
                except:
                    pass
            
            ax_critical.set_title('Critical Values', fontsize=12)
            ax_critical.axis('off')
        else:
            ax_critical.text(0.5, 0.5, "No critical values detected",
                          ha='center', va='center', fontsize=12)
            ax_critical.axis('off')
        
        # Organ system heatmap
        ax_heatmap = fig.add_subplot(gs[7:12, 0:5])
        if 'stability_over_time' in stability_report and not stability_report['stability_over_time'].empty:
            stability_df = stability_report['stability_over_time']
            
            # Get columns for organ systems
            organ_cols = [col for col in stability_df.columns 
                         if col.endswith('_score') and 
                         col.split('_')[0] in self.organ_colors]
            
            if organ_cols:
                # Prepare data for heatmap
                heatmap_data = stability_df[['window_start'] + organ_cols].copy()
                
                # Convert timestamps to formatted strings
                heatmap_data['window_start'] = heatmap_data['window_start'].dt.strftime('%m/%d %H:%M')
                
                # Set window_start as index
                heatmap_data = heatmap_data.set_index('window_start')
                
                # Rename columns to readable names
                heatmap_data.columns = [col.split('_')[0].title() for col in heatmap_data.columns]
                
                # Create heatmap
                sns.heatmap(heatmap_data.T, annot=True, cmap=self.stability_cmap,
                           linewidths=0.5, ax=ax_heatmap, vmin=0, vmax=2, fmt='.2f')
                
                # Set labels
                ax_heatmap.set_xlabel('Time Window')
                ax_heatmap.set_ylabel('Organ System')
                
                # Set title
                ax_heatmap.set_title('Organ System Status Over Time', fontsize=12)
            else:
                ax_heatmap.text(0.5, 0.5, "No organ system data available for heatmap",
                              ha='center', va='center', fontsize=12)
                ax_heatmap.axis('off')
        else:
            ax_heatmap.text(0.5, 0.5, "Insufficient data for organ system heatmap",
                          ha='center', va='center', fontsize=12)
            ax_heatmap.axis('off')
        
        # Phase portrait (if we have heart rate and blood pressure)
        ax_phase = fig.add_subplot(gs[7:12, 5:])
        if all(v in timeline_df.columns for v in ['heart_rate', 'sbp']):
            # Filter data
            phase_data = timeline_df[['measurement_time', 'heart_rate', 'sbp']].dropna()
            
            if len(phase_data) >= 5:
                # Plot trajectory
                sc = ax_phase.scatter(phase_data['heart_rate'], phase_data['sbp'], 
                                    c=range(len(phase_data)), cmap='viridis', 
                                    alpha=0.7, s=30)
                
                # Connect points with lines
                ax_phase.plot(phase_data['heart_rate'], phase_data['sbp'], 
                           '-', color='gray', alpha=0.3, linewidth=1)
                
                # Mark start and end points
                ax_phase.scatter(phase_data['heart_rate'].iloc[0], phase_data['sbp'].iloc[0], 
                              color='green', s=80, label='Start')
                ax_phase.scatter(phase_data['heart_rate'].iloc[-1], phase_data['sbp'].iloc[-1], 
                              color='red', s=80, label='End')
                
                # Add reference ranges
                if 'heart_rate' in self.reference_ranges:
                    ax_phase.axvspan(self.reference_ranges['heart_rate'][0], 
                                  self.reference_ranges['heart_rate'][1],
                                  alpha=0.1, color='green')
                
                if 'sbp' in self.reference_ranges:
                    ax_phase.axhspan(self.reference_ranges['sbp'][0], 
                                  self.reference_ranges['sbp'][1],
                                  alpha=0.1, color='green')
                
                # Set labels
                ax_phase.set_xlabel('Heart Rate')
                ax_phase.set_ylabel('Systolic BP')
                
                # Add colorbar to show time progression
                cbar = plt.colorbar(sc, ax=ax_phase)
                cbar.set_label('Time Progression')
                
                # Add grid and legend
                ax_phase.grid(True, linestyle='--', alpha=0.5)
                ax_phase.legend(loc='upper left')
                
                # Set title
                ax_phase.set_title('Heart Rate vs Blood Pressure Trajectory', fontsize=12)
            else:
                ax_phase.text(0.5, 0.5, "Insufficient data for phase portrait",
                           ha='center', va='center', fontsize=12)
                ax_phase.axis('off')
        else:
            ax_phase.text(0.5, 0.5, "Missing heart rate or blood pressure data",
                       ha='center', va='center', fontsize=12)
            ax_phase.axis('off')
        
        # Adjust layout
        fig.tight_layout()
        fig.subplots_adjust(hspace=0.5, wspace=0.3)
        
        # Add figure title
        fig.suptitle('Patient Physiological Dynamics Dashboard', fontsize=16, y=0.99)
        
        return fig