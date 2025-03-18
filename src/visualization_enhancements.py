"""
BioDynamICS - Visualization Enhancements Module

This module extends the basic visualization capabilities with interactive
components, exportable report formats, and improved aesthetics for portfolio
presentation.

Author: Alexander Clarke
Date: March 18, 2025
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio
from datetime import datetime
import json
import base64
from pathlib import Path
import logging
from typing import Dict, List, Union, Optional, Tuple, Any

# Import BioDynamICS modules
from src.visualization import ClinicalVisualizer

class EnhancedVisualizer(ClinicalVisualizer):
    """
    Enhanced visualization class with interactive components,
    exportable report formats, and improved aesthetics.
    """
    
    def __init__(self):
        """Initialize the EnhancedVisualizer."""
        # Call parent constructor
        super().__init__()
        
        # Set up logging
        self.logger = logging.getLogger("EnhancedVisualizer")
        
        # Set default theme for enhanced visualizations
        self.theme = {
            'background_color': '#F8F9FA',
            'text_color': '#212529',
            'primary_color': '#007BFF',
            'secondary_color': '#6C757D',
            'success_color': '#28A745',
            'danger_color': '#DC3545',
            'warning_color': '#FFC107',
            'info_color': '#17A2B8',
            'font_family': 'Arial, sans-serif',
            'title_font_size': 18,
            'axis_font_size': 12,
            'label_font_size': 10,
            'grid_alpha': 0.2,
            'line_width': 2,
            'marker_size': 6
        }
        
        # Set default export settings
        self.export_settings = {
            'image_format': 'png',
            'image_dpi': 300,
            'html_width': 1000,
            'html_height': 600,
            'pdf_width': 8.5,
            'pdf_height': 11,
            'include_code': False,
            'include_data': False
        }
        
        # Initialize templates
        self._initialize_templates()
        
        self.logger.info("Initialized EnhancedVisualizer")
    
    def _initialize_templates(self):
        """Initialize templates for visualizations."""
        # Create custom plotly template
        custom_template = go.layout.Template()
        
        # Set font
        custom_template.layout.font = {
            'family': self.theme['font_family'],
            'color': self.theme['text_color']
        }
        
        # Set colors
        custom_template.layout.colorway = [
            self.theme['primary_color'],
            self.theme['success_color'],
            self.theme['danger_color'],
            self.theme['warning_color'],
            self.theme['info_color'],
            self.theme['secondary_color']
        ]
        
        # Set paper and plot background
        custom_template.layout.paper_bgcolor = self.theme['background_color']
        custom_template.layout.plot_bgcolor = self.theme['background_color']
        
        # Set title font
        custom_template.layout.title = {
            'font': {
                'size': self.theme['title_font_size'],
                'color': self.theme['text_color']
            },
            'x': 0.5,
            'xanchor': 'center'
        }
        
        # Set axis properties
        axis_template = {
            'gridcolor': f"rgba(0,0,0,{self.theme['grid_alpha']})",
            'linecolor': self.theme['secondary_color'],
            'titlefont': {'size': self.theme['axis_font_size']},
            'tickfont': {'size': self.theme['label_font_size']},
            'zeroline': True,
            'zerolinecolor': self.theme['secondary_color'],
            'zerolinewidth': 1
        }
        
        custom_template.layout.xaxis = axis_template
        custom_template.layout.yaxis = axis_template
        
        # Set legend properties
        custom_template.layout.legend = {
            'font': {'size': self.theme['label_font_size']},
            'bgcolor': f"rgba(255,255,255,0.5)",
            'bordercolor': self.theme['secondary_color'],
            'borderwidth': 1
        }
        
        # Set margin
        custom_template.layout.margin = {'l': 60, 'r': 40, 't': 60, 'b': 60}
        
        # Register template
        pio.templates['biodynamics'] = custom_template
        pio.templates.default = 'biodynamics'
    
    def update_theme(self, theme_updates):
        """
        Update visualization theme.
        
        Parameters:
        -----------
        theme_updates : dict
            Dictionary with theme updates
        """
        # Update theme
        self.theme.update(theme_updates)
        
        # Reinitialize templates
        self._initialize_templates()
        
        self.logger.info("Updated visualization theme")
    
    def update_export_settings(self, export_updates):
        """
        Update export settings.
        
        Parameters:
        -----------
        export_updates : dict
            Dictionary with export setting updates
        """
        # Update export settings
        self.export_settings.update(export_updates)
        
        self.logger.info("Updated export settings")
    
    # =========================================================================
    # Interactive Vital Sign Visualizations
    # =========================================================================
    
    def create_interactive_vital_signs(self, timeline, vital_signs=None):
        """
        Create interactive visualization of vital signs.
        
        Parameters:
        -----------
        timeline : pandas.DataFrame
            Patient timeline DataFrame with 'measurement_time' column
        vital_signs : list, optional
            List of vital signs to visualize
            
        Returns:
        --------
        plotly.graph_objects.Figure
            Interactive figure
        """
        # Default vital signs if not specified
        if vital_signs is None:
            vital_signs = ['heart_rate', 'respiratory_rate', 'sbp', 'dbp', 'temperature', 'o2_saturation']
        
        # Filter vital signs to those available in the data
        available_vitals = [v for v in vital_signs if v in timeline.columns]
        
        if not available_vitals:
            self.logger.warning("No vital signs available for visualization")
            return None
        
        # Create subplots
        fig = make_subplots(
            rows=len(available_vitals),
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=[v.replace('_', ' ').title() for v in available_vitals]
        )
        
        # Add traces for each vital sign
        for i, vital in enumerate(available_vitals):
            # Filter out NaN values
            vital_data = timeline[['measurement_time', vital]].dropna()
            
            if len(vital_data) == 0:
                continue
            
            # Add line trace
            fig.add_trace(
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
                row=i+1,
                col=1
            )
            
            # Add reference range if available
            if vital in self.reference_ranges:
                lower, upper = self.reference_ranges[vital]
                
                # Add reference range as a filled area
                fig.add_trace(
                    go.Scatter(
                        x=vital_data['measurement_time'],
                        y=[lower] * len(vital_data),
                        mode='lines',
                        line=dict(width=0),
                        showlegend=False
                    ),
                    row=i+1,
                    col=1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=vital_data['measurement_time'],
                        y=[upper] * len(vital_data),
                        mode='lines',
                        line=dict(width=0),
                        fill='tonexty',
                        fillcolor=f"rgba(0,128,0,0.2)",
                        name='Normal Range',
                        showlegend=(i == 0)  # Only show in legend for first plot
                    ),
                    row=i+1,
                    col=1
                )
        
        # Update layout
        fig.update_layout(
            title='Vital Signs Timeline',
            height=250 * len(available_vitals),
            width=self.export_settings['html_width'],
            hovermode='closest',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # Update x-axis
        fig.update_xaxes(
            title_text='Time',
            row=len(available_vitals),
            col=1
        )
        
        return fig
    
    def create_interactive_phase_portrait(self, timeline, x_measure, y_measure, z_measure=None, color_by=None):
        """
        Create interactive phase portrait visualization.
        
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
        color_by : str, optional
            Measurement to color points by
            
        Returns:
        --------
        plotly.graph_objects.Figure
            Interactive figure
        """
        # Check if measures exist in the timeline
        required_measures = [x_measure, y_measure]
        if z_measure is not None:
            required_measures.append(z_measure)
        if color_by is not None:
            required_measures.append(color_by)
            
        missing_measures = [m for m in required_measures if m not in timeline.columns]
        if missing_measures:
            self.logger.warning(f"Missing measures in timeline: {', '.join(missing_measures)}")
            return None
        
        # Filter to rows with all required measures
        data = timeline[required_measures + ['measurement_time']].dropna()
        
        if len(data) < 5:
            self.logger.warning("Not enough data points after filtering")
            return None
        
        # Create figure
        if z_measure is None:
            # 2D phase portrait
            if color_by is None:
                # Color by time
                fig = px.scatter(
                    data,
                    x=x_measure,
                    y=y_measure,
                    color=data.index,
                    color_continuous_scale='Viridis',
                    labels={
                        x_measure: x_measure.replace('_', ' ').title(),
                        y_measure: y_measure.replace('_', ' ').title()
                    },
                    title=f"Phase Portrait: {x_measure.replace('_', ' ').title()} vs {y_measure.replace('_', ' ').title()}"
                )
                
                # Update colorbar
                fig.update_colorbar(
                    title="Time Progression",
                    tickvals=[data.index.min(), data.index.max()],
                    ticktext=["Start", "End"]
                )
            else:
                # Color by specified measure
                fig = px.scatter(
                    data,
                    x=x_measure,
                    y=y_measure,
                    color=color_by,
                    labels={
                        x_measure: x_measure.replace('_', ' ').title(),
                        y_measure: y_measure.replace('_', ' ').title(),
                        color_by: color_by.replace('_', ' ').title()
                    },
                    title=f"Phase Portrait: {x_measure.replace('_', ' ').title()} vs {y_measure.replace('_', ' ').title()}"
                )
            
            # Add trajectory line
            fig.add_trace(
                go.Scatter(
                    x=data[x_measure],
                    y=data[y_measure],
                    mode='lines',
                    line=dict(
                        color='rgba(100,100,100,0.3)',
                        width=1
                    ),
                    showlegend=False
                )
            )
            
            # Add reference ranges if available
            if x_measure in self.reference_ranges:
                lower, upper = self.reference_ranges[x_measure]
                fig.add_shape(
                    type="rect",
                    x0=lower,
                    x1=upper,
                    y0=data[y_measure].min(),
                    y1=data[y_measure].max(),
                    fillcolor="rgba(0,128,0,0.1)",
                    line=dict(width=0),
                    layer="below"
                )
            
            if y_measure in self.reference_ranges:
                lower, upper = self.reference_ranges[y_measure]
                fig.add_shape(
                    type="rect",
                    x0=data[x_measure].min(),
                    x1=data[x_measure].max(),
                    y0=lower,
                    y1=upper,
                    fillcolor="rgba(0,128,0,0.1)",
                    line=dict(width=0),
                    layer="below"
                )
            
            # Mark start and end points
            fig.add_trace(
                go.Scatter(
                    x=[data[x_measure].iloc[0]],
                    y=[data[y_measure].iloc[0]],
                    mode='markers',
                    marker=dict(
                        color='green',
                        size=12,
                        symbol='circle'
                    ),
                    name='Start'
                )
            )
            
            fig.add_trace(
                go.Scatter(
                    x=[data[x_measure].iloc[-1]],
                    y=[data[y_measure].iloc[-1]],
                    mode='markers',
                    marker=dict(
                        color='red',
                        size=12,
                        symbol='circle'
                    ),
                    name='End'
                )
            )
        else:
            # 3D phase portrait
            if color_by is None:
                # Color by time
                fig = px.scatter_3d(
                    data,
                    x=x_measure,
                    y=y_measure,
                    z=z_measure,
                    color=data.index,
                    color_continuous_scale='Viridis',
                    labels={
                        x_measure: x_measure.replace('_', ' ').title(),
                        y_measure: y_measure.replace('_', ' ').title(),
                        z_measure: z_measure.replace('_', ' ').title()
                    },
                    title=f"3D Phase Portrait: {x_measure.replace('_', ' ').title()}, {y_measure.replace('_', ' ').title()}, {z_measure.replace('_', ' ').title()}"
                )
                
                # Update colorbar
                fig.update_colorbar(
                    title="Time Progression",
                    tickvals=[data.index.min(), data.index.max()],
                    ticktext=["Start", "End"]
                )
            else:
                # Color by specified measure
                fig = px.scatter_3d(
                    data,
                    x=x_measure,
                    y=y_measure,
                    z=z_measure,
                    color=color_by,
                    labels={
                        x_measure: x_measure.replace('_', ' ').title(),
                        y_measure: y_measure.replace('_', ' ').title(),
                        z_measure: z_measure.replace('_', ' ').title(),
                        color_by: color_by.replace('_', ' ').title()
                    },
                    title=f"3D Phase Portrait: {x_measure.replace('_', ' ').title()}, {y_measure.replace('_', ' ').title()}, {z_measure.replace('_', ' ').title()}"
                )
            
            # Add trajectory line
            fig.add_trace(
                go.Scatter3d(
                    x=data[x_measure],
                    y=data[y_measure],
                    z=data[z_measure],
                    mode='lines',
                    line=dict(
                        color='rgba(100,100,100,0.3)',
                        width=1
                    ),
                    showlegend=False
                )
            )
            
            # Mark start and end points
            fig.add_trace(
                go.Scatter3d(
                    x=[data[x_measure].iloc[0]],
                    y=[data[y_measure].iloc[0]],
                    z=[data[z_measure].iloc[0]],
                    mode='markers',
                    marker=dict(
                        color='green',
                        size=6,
                        symbol='circle'
                    ),
                    name='Start'
                )
            )
            
            fig.add_trace(
                go.Scatter3d(
                    x=[data[x_measure].iloc[-1]],
                    y=[data[y_measure].iloc[-1]],
                    z=[data[z_measure].iloc[-1]],
                    mode='markers',
                    marker=dict(
                        color='red',
                        size=6,
                        symbol='circle'
                    ),
                    name='End'
                )
            )
        
        # Update layout
        fig.update_layout(
            width=self.export_settings['html_width'],
            height=self.export_settings['html_height'],
            hovermode='closest'
        )
        
        return fig
    
    def create_interactive_organ_system_radar(self, stability_report):
        """
        Create interactive radar plot showing organ system status.
        
        Parameters:
        -----------
        stability_report : dict
            Stability report from PhysiologicalSignalProcessor
            
        Returns:
        --------
        plotly.graph_objects.Figure
            Interactive figure
        """
        # Extract organ system scores
        if 'organ_system_summary' not in stability_report:
            self.logger.warning("No organ system summary in stability report")
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
            self.logger.warning("No organ systems available for radar plot")
            return None
        
        # Close the polygon
        systems.append(systems[0])
        scores.append(scores[0])
        
        # Create radar plot
        fig = go.Figure()
        
        # Add radar trace
        fig.add_trace(
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
            )
        )
        
        # Add severity levels
        severity_levels = [0.5, 1.0, 1.5]
        severity_labels = ['Mild', 'Moderate', 'Severe']
        severity_colors = ['rgba(255,236,179,0.3)', 'rgba(255,213,79,0.3)', 'rgba(255,143,0,0.3)']
        
        for level, label, color in zip(severity_levels, severity_labels, severity_colors):
            fig.add_trace(
                go.Scatterpolar(
                    r=[level] * len(systems),
                    theta=systems,
                    fill='toself',
                    fillcolor=color,
                    line=dict(width=0),
                    name=label,
                    hoverinfo='skip'
                )
            )
        
        # Update layout
        fig.update_layout(
            title='Organ System Status',
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, max(2, max(scores) * 1.2)]
                )
            ),
            width=self.export_settings['html_width'],
            height=self.export_settings['html_height'],
            showlegend=True
        )
        
        return fig
    
    def create_interactive_allostatic_load_trend(self, stability_over_time):
        """
        Create interactive plot of allostatic load trends over time.
        
        Parameters:
        -----------
        stability_over_time : pandas.DataFrame
            DataFrame with stability metrics over time
            
        Returns:
        --------
        plotly.graph_objects.Figure
            Interactive figure
        """
        if stability_over_time is None or stability_over_time.empty:
            self.logger.warning("No stability data provided")
            return None
            
        if 'allostatic_load' not in stability_over_time.columns:
            self.logger.warning("No allostatic load in stability data")
            return None
            
        if 'window_start' not in stability_over_time.columns:
            self.logger.warning("No window timestamps in stability data")
            return None
        
        # Create figure
        fig = go.Figure()
        
        # Add allostatic load trace
        fig.add_trace(
            go.Scatter(
                x=stability_over_time['window_start'],
                y=stability_over_time['allostatic_load'],
                mode='lines+markers',
                name='Allostatic Load',
                line=dict(
                    color=self.theme['danger_color'],
                    width=self.theme['line_width']
                ),
                marker=dict(
                    size=self.theme['marker_size']
                )
            )
        )
        
        # Add severity zones
        fig.add_hrect(
            y0=0, y1=0.5,
            fillcolor="rgba(0,128,0,0.1)",
            line_width=0,
            annotation_text="Normal",
            annotation_position="right"
        )
        
        fig.add_hrect(
            y0=0.5, y1=1.0,
            fillcolor="rgba(255,255,0,0.1)",
            line_width=0,
            annotation_text="Mild Stress",
            annotation_position="right"
        )
        
        fig.add_hrect(
            y0=1.0, y1=1.5,
            fillcolor="rgba(255,165,0,0.1)",
            line_width=0,
            annotation_text="Moderate Stress",
            annotation_position="right"
        )
        
        fig.add_hrect(
            y0=1.5, y1=3.0,
            fillcolor="rgba(255,0,0,0.1)",
            line_width=0,
            annotation_text="Severe Stress",
            annotation_position="right"
        )
        
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
                    
                    fig.add_trace(
                        go.Scatter(
                            x=trend_dates,
                            y=trend_y,
                            mode='lines',
                            name=f'Trend (R²={r_value**2:.2f})',
                            line=dict(
                                color='black',
                                width=1.5,
                                dash='dash'
                            )
                        )
                    )
                    
                    # Add trend annotation
                    if slope > 0.05:
                        fig.add_annotation(
                            x=0.02,
                            y=0.98,
                            xref="paper",
                            yref="paper",
                            text="⚠️ Worsening",
                            showarrow=False,
                            font=dict(
                                size=14,
                                color="black"
                            ),
                            bgcolor="rgba(255,205,210,0.5)",
                            bordercolor="rgba(255,205,210,1)",
                            borderwidth=1,
                            borderpad=4,
                            align="left"
                        )
                    elif slope < -0.05:
                        fig.add_annotation(
                            x=0.02,
                            y=0.98,
                            xref="paper",
                            yref="paper",
                            text="✓ Improving",
                            showarrow=False,
                            font=dict(
                                size=14,
                                color="black"
                            ),
                            bgcolor="rgba(200,230,201,0.5)",
                            bordercolor="rgba(200,230,201,1)",
                            borderwidth=1,
                            borderpad=4,
                            align="left"
                        )
            except Exception as e:
                self.logger.warning(f"Error calculating trend: {e}")
        
        # Update layout
        fig.update_layout(
            title='Physiological Stress Over Time',
            xaxis_title='Time',
            yaxis_title='Allostatic Load',
            width=self.export_settings['html_width'],
            height=self.export_settings['html_height'],
            hovermode='closest'
        )
        
        return fig
    
    # =========================================================================
    # Interactive Dashboard
    # =========================================================================
    
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
        if not patient_timeline or 'timeline' not in patient_timeline:
            self.logger.warning("Invalid patient timeline")
            return None
            
        if not stability_report:
            self.logger.warning("No stability report provided")
            return None
        
        # Extract patient info
        patient_info = patient_timeline.get('info', {})
        subject_id = patient_info.get('subject_id', 'Unknown')
        gender = patient_info.get('gender', 'Unknown')
        
        # Extract timeline data
        timeline_df = patient_timeline['timeline']
        
        # Create dashboard
        dashboard = make_subplots(
            rows=3, cols=2,
            specs=[
                [{"colspan": 2}, None],
                [{"type": "polar"}, {"type": "xy"}],
                [{"type": "xy"}, {"type": "xy"}]
            ],
            subplot_titles=[
                f"Patient {subject_id} Dashboard",
                "Organ System Status", "Allostatic Load Trend",
                "Vital Signs", "Heart Rate vs Blood Pressure"
            ],
            vertical_spacing=0.1,
            horizontal_spacing=0.05
        )
        
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
            
            # Update polar layout
            dashboard.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, max(2, max(scores) * 1.2)]
                    )
                )
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
                
                # Add reference range if available
                if vital in self.reference_ranges:
                    lower, upper = self.reference_ranges[vital]
                    
                    dashboard.add_hrect(
                        y0=lower, y1=upper,
                        fillcolor="rgba(0,128,0,0.1)",
                        line_width=0,
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
                
                # Mark start and end points
                dashboard.add_trace(
                    go.Scatter(
                        x=[phase_data['heart_rate'].iloc[0]],
                        y=[phase_data['sbp'].iloc[0]],
                        mode='markers',
                        marker=dict(
                            color='green',
                            size=10,
                            symbol='circle'
                        ),
                        name='Start'
                    ),
                    row=3, col=2
                )
                
                dashboard.add_trace(
                    go.Scatter(
                        x=[phase_data['heart_rate'].iloc[-1]],
                        y=[phase_data['sbp'].iloc[-1]],
                        mode='markers',
                        marker=dict(
                            color='red',
                            size=10,
                            symbol='circle'
                        ),
                        name='End'
                    ),
                    row=3, col=2
                )
                
                # Add reference ranges
                if 'heart_rate' in self.reference_ranges:
                    lower, upper = self.reference_ranges['heart_rate']
                    dashboard.add_vrect(
                        x0=lower, x1=upper,
                        fillcolor="rgba(0,128,0,0.1)",
                        line_width=0,
                        row=3, col=2
                    )
                
                if 'sbp' in self.reference_ranges:
                    lower, upper = self.reference_ranges['sbp']
                    dashboard.add_hrect(
                        y0=lower, y1=upper,
                        fillcolor="rgba(0,128,0,0.1)",
                        line_width=0,
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
        
        # Update axes labels
        dashboard.update_xaxes(title_text="Time", row=2, col=2)
        dashboard.update_yaxes(title_text="Allostatic Load", row=2, col=2)
        
        dashboard.update_xaxes(title_text="Time", row=3, col=1)
        dashboard.update_yaxes(title_text=vital.replace('_', ' ').title() if available_vitals else "", row=3, col=1)
        
        dashboard.update_xaxes(title_text="Heart Rate", row=3, col=2)
        dashboard.update_yaxes(title_text="Systolic BP", row=3, col=2)
        
        return dashboard
    
    # =========================================================================
    # Exportable Report Formats
    # =========================================================================
    
    def export_visualization(self, fig, export_path, format=None):
        """
        Export visualization to file.
        
        Parameters:
        -----------
        fig : matplotlib.figure.Figure or plotly.graph_objects.Figure
            Figure to export
        export_path : str
            Path to export the visualization
        format : str, optional
            Export format (default: from export_settings)
            
        Returns:
        --------
        bool
            Success status
        """
        if fig is None:
            self.logger.warning("No figure to export")
            return False
        
        # Use specified format or default
        format = format or self.export_settings['image_format']
        
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(export_path), exist_ok=True)
            
            # Export based on figure type
            if isinstance(fig, plt.Figure):
                # Matplotlib figure
                fig.savefig(
                    export_path,
                    format=format,
                    dpi=self.export_settings['image_dpi'],
                    bbox_inches='tight'
                )
                self.logger.info(f"Exported matplotlib figure to {export_path}")
                return True
                
            elif 'plotly.graph_objs' in str(type(fig)):
                # Plotly figure
                if format.lower() in ['png', 'jpg', 'jpeg', 'webp', 'svg', 'pdf']:
                    # Export as static image
                    pio.write_image(
                        fig,
                        export_path,
                        format=format,
                        width=self.export_settings['html_width'],
                        height=self.export_settings['html_height'],
                        scale=2  # Higher resolution
                    )
                    self.logger.info(f"Exported plotly figure as {format} to {export_path}")
                    return True
                    
                elif format.lower() == 'html':
                    # Export as interactive HTML
                    config = {
                        'displayModeBar': True,
                        'responsive': True,
                        'toImageButtonOptions': {
                            'format': 'png',
                            'filename': os.path.basename(export_path).replace('.html', ''),
                            'height': self.export_settings['html_height'],
                            'width': self.export_settings['html_width'],
                            'scale': 2
                        }
                    }
                    
                    pio.write_html(
                        fig,
                        export_path,
                        config=config,
                        include_plotlyjs='cdn',
                        include_mathjax='cdn',
                        auto_open=False
                    )
                    self.logger.info(f"Exported plotly figure as HTML to {export_path}")
                    return True
                    
                elif format.lower() == 'json':
                    # Export as JSON
                    with open(export_path, 'w') as f:
                        f.write(fig.to_json())
                    self.logger.info(f"Exported plotly figure as JSON to {export_path}")
                    return True
                    
                else:
                    self.logger.warning(f"Unsupported format for plotly figure: {format}")
                    return False
            else:
                self.logger.warning(f"Unsupported figure type: {type(fig)}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error exporting visualization: {e}")
            return False
    
    def create_html_report(self, patient_timeline, stability_report, export_path):
        """
        Create comprehensive HTML report for a patient.
        
        Parameters:
        -----------
        patient_timeline : dict
            Patient timeline dictionary with 'info' and 'timeline' keys
        stability_report : dict
            Stability report from PhysiologicalSignalProcessor
        export_path : str
            Path to export the report
            
        Returns:
        --------
        bool
            Success status
        """
        if not patient_timeline or 'timeline' not in patient_timeline:
            self.logger.warning("Invalid patient timeline")
            return False
            
        if not stability_report:
            self.logger.warning("No stability report provided")
            return False
        
        try:
            # Extract patient info
            patient_info = patient_timeline.get('info', {})
            subject_id = patient_info.get('subject_id', 'Unknown')
            
            # Create visualizations
            dashboard = self.create_interactive_dashboard(patient_timeline, stability_report)
            vital_signs = self.create_interactive_vital_signs(patient_timeline['timeline'])
            organ_radar = self.create_interactive_organ_system_radar(stability_report)
            
            # Convert to HTML
            dashboard_html = ""
            vital_signs_html = ""
            organ_radar_html = ""
            
            if dashboard is not None:
                dashboard_html = dashboard.to_html(
                    full_html=False,
                    include_plotlyjs=False,
                    config={'responsive': True}
                )
            
            if vital_signs is not None:
                vital_signs_html = vital_signs.to_html(
                    full_html=False,
                    include_plotlyjs=False,
                    config={'responsive': True}
                )
            
            if organ_radar is not None:
                organ_radar_html = organ_radar.to_html(
                    full_html=False,
                    include_plotlyjs=False,
                    config={'responsive': True}
                )
            
            # Create HTML report
            html_content = f"""
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Patient {subject_id} Report</title>
                <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
                <style>
                    body {{
                        font-family: {self.theme['font_family']};
                        color: {self.theme['text_color']};
                        background-color: {self.theme['background_color']};
                        margin: 0;
                        padding: 20px;
                    }}
                    .container {{
                        max-width: 1200px;
                        margin: 0 auto;
                    }}
                    .header {{
                        background-color: {self.theme['primary_color']};
                        color: white;
                        padding: 20px;
                        margin-bottom: 20px;
                        border-radius: 5px;
                    }}
                    .section {{
                        background-color: white;
                        padding: 20px;
                        margin-bottom: 20px;
                        border-radius: 5px;
                        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                    }}
                    .section h2 {{
                        color: {self.theme['primary_color']};
                        border-bottom: 1px solid #eee;
                        padding-bottom: 10px;
                    }}
                    .patient-info {{
                        display: grid;
                        grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
                        gap: 10px;
                    }}
                    .info-item {{
                        padding: 10px;
                        background-color: #f8f9fa;
                        border-radius: 5px;
                    }}
                    .info-label {{
                        font-weight: bold;
                        color: {self.theme['secondary_color']};
                    }}
                    .critical {{
                        color: {self.theme['danger_color']};
                        font-weight: bold;
                    }}
                    .warning {{
                        color: {self.theme['warning_color']};
                        font-weight: bold;
                    }}
                    .normal {{
                        color: {self.theme['success_color']};
                    }}
                    .footer {{
                        text-align: center;
                        margin-top: 30px;
                        color: {self.theme['secondary_color']};
                        font-size: 0.8em;
                    }}
                </style>
            </head>
            <body>
                <div class="container">
                    <div class="header">
                        <h1>BioDynamICS Patient Report</h1>
                        <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                    </div>
                    
                    <div class="section">
                        <h2>Patient Information</h2>
                        <div class="patient-info">
                            <div class="info-item">
                                <div class="info-label">Patient ID</div>
                                <div>{patient_info.get('subject_id', 'Unknown')}</div>
                            </div>
                            <div class="info-item">
                                <div class="info-label">Gender</div>
                                <div>{patient_info.get('gender', 'Unknown')}</div>
                            </div>
                            <div class="info-item">
                                <div class="info-label">Date of Birth</div>
                                <div>{patient_info.get('dob', 'Unknown')}</div>
                            </div>
                            <div class="info-item">
                                <div class="info-label">Admissions</div>
                                <div>{patient_info.get('admissions', 'Unknown')}</div>
                            </div>
                            <div class="info-item">
                                <div class="info-label">ICU Stays</div>
                                <div>{patient_info.get('icustays', 'Unknown')}</div>
                            </div>
                        </div>
                    </div>
                    
                    <div class="section">
                        <h2>Physiological Stability Dashboard</h2>
                        <p>Comprehensive view of patient's physiological stability.</p>
                        {dashboard_html}
                    </div>
                    
                    <div class="section">
                        <h2>Vital Signs Timeline</h2>
                        <p>Detailed view of vital sign measurements over time.</p>
                        {vital_signs_html}
                    </div>
                    
                    <div class="section">
                        <h2>Organ System Status</h2>
                        <p>Radar chart showing the status of different organ systems.</p>
                        {organ_radar_html}
                    </div>
                    
                    <div class="section">
                        <h2>Stability Assessment</h2>
                        <p>Summary of patient's physiological stability.</p>
                        <div class="patient-info">
            """
            
            # Add stability metrics
            allostatic_load = stability_report.get('overall_results', {}).get('allostatic_load', 'N/A')
            
            if isinstance(allostatic_load, (int, float)):
                if allostatic_load < 0.5:
                    stability_class = "normal"
                    stability_text = "Normal"
                elif allostatic_load < 1.0:
                    stability_class = "normal"
                    stability_text = "Mild Stress"
                elif allostatic_load < 1.5:
                    stability_class = "warning"
                    stability_text = "Moderate Stress"
                else:
                    stability_class = "critical"
                    stability_text = "Severe Stress"
                
                html_content += f"""
                            <div class="info-item">
                                <div class="info-label">Allostatic Load</div>
                                <div class="{stability_class}">{allostatic_load:.2f} ({stability_text})</div>
                            </div>
                """
            
            # Add organ system status
            for system, status in stability_report.get('organ_system_summary', {}).items():
                system_score = status.get('score', 0)
                
                if system_score < 0.5:
                    system_class = "normal"
                    system_text = "Normal"
                elif system_score < 1.0:
                    system_class = "normal"
                    system_text = "Mild Dysfunction"
                elif system_score < 1.5:
                    system_class = "warning"
                    system_text = "Moderate Dysfunction"
                else:
                    system_class = "critical"
                    system_text = "Severe Dysfunction"
                
                html_content += f"""
                            <div class="info-item">
                                <div class="info-label">{system.replace('_', ' ').title()} System</div>
                                <div class="{system_class}">{system_score:.2f} ({system_text})</div>
                            </div>
                """
            
            # Add critical values
            html_content += """
                        </div>
                    </div>
                    
                    <div class="section">
                        <h2>Critical Values</h2>
                        <p>Measurements that are significantly outside normal ranges.</p>
                        <table style="width:100%; border-collapse: collapse;">
                            <tr style="background-color: #f8f9fa;">
                                <th style="padding: 10px; text-align: left; border-bottom: 1px solid #ddd;">Measurement</th>
                                <th style="padding: 10px; text-align: left; border-bottom: 1px solid #ddd;">Value</th>
                                <th style="padding: 10px; text-align: left; border-bottom: 1px solid #ddd;">Reference Range</th>
                                <th style="padding: 10px; text-align: left; border-bottom: 1px solid #ddd;">Deviation</th>
                            </tr>
            """
            
            for cv in stability_report.get('critical_values', []):
                measurement = cv.get('measurement', '').replace('_', ' ').title()
                value = cv.get('value', 0)
                ref_range = cv.get('reference_range', [0, 0])
                deviation = cv.get('percent_deviation', 0)
                
                if deviation > 50:
                    row_class = "critical"
                elif deviation > 25:
                    row_class = "warning"
                else:
                    row_class = ""
                
                html_content += f"""
                            <tr class="{row_class}">
                                <td style="padding: 10px; border-bottom: 1px solid #ddd;">{measurement}</td>
                                <td style="padding: 10px; border-bottom: 1px solid #ddd;">{value:.1f}</td>
                                <td style="padding: 10px; border-bottom: 1px solid #ddd;">{ref_range[0]:.1f} - {ref_range[1]:.1f}</td>
                                <td style="padding: 10px; border-bottom: 1px solid #ddd;">{deviation:.1f}%</td>
                            </tr>
                """
            
            # Close HTML
            html_content += """
                        </table>
                    </div>
                    
                    <div class="footer">
                        <p>Generated by BioDynamICS System</p>
                        <p>© 2025 BioDynamICS</p>
                    </div>
                </div>
            </body>
            </html>
            """
            
            # Write HTML to file
            os.makedirs(os.path.dirname(export_path), exist_ok=True)
            with open(export_path, 'w') as f:
                f.write(html_content)
            
            self.logger.info(f"Created HTML report at {export_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error creating HTML report: {e}")
            return False
    
    def create_pdf_report(self, patient_timeline, stability_report, export_path):
        """
        Create PDF report for a patient.
        
        Parameters:
        -----------
        patient_timeline : dict
            Patient timeline dictionary with 'info' and 'timeline' keys
        stability_report : dict
            Stability report from PhysiologicalSignalProcessor
        export_path : str
            Path to export the report
            
        Returns:
        --------
        bool
            Success status
        """
        try:
            # First create HTML report
            html_path = export_path.replace('.pdf', '.html')
            self.create_html_report(patient_timeline, stability_report, html_path)
            
            # Convert HTML to PDF using weasyprint
            try:
                from weasyprint import HTML
                HTML(html_path).write_pdf(export_path)
                self.logger.info(f"Created PDF report at {export_path}")
                return True
            except ImportError:
                self.logger.warning("weasyprint not installed, cannot create PDF report")
                self.logger.info("Install weasyprint with: pip install weasyprint")
                return False
                
        except Exception as e:
            self.logger.error(f"Error creating PDF report: {e}")
            return False

# Example usage
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create enhanced visualizer
    visualizer = EnhancedVisualizer()
    
    # Example data
    timeline = pd.DataFrame({
        'measurement_time': pd.date_range(start='2025-01-01', periods=100, freq='H'),
        'heart_rate': np.random.normal(80, 10, 100),
        'sbp': np.random.normal(120, 15, 100),
        'dbp': np.random.normal(80, 10, 100),
        'respiratory_rate': np.random.normal(16, 3, 100),
        'temperature': np.random.normal(37, 0.5, 100),
        'o2_saturation': np.random.normal(98, 2, 100)
    })
    
    # Create interactive vital signs visualization
    fig = visualizer.create_interactive_vital_signs(timeline)
    
    # Export visualization
    visualizer.export_visualization(fig, "results/interactive_vital_signs.html", format="html")