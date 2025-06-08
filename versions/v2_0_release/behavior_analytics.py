"""
Behavior Analytics Module for SereniTeach
Generates trend visualizations and data insights from incident reports and behavior logs
"""

import json
import logging
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)

class BehaviorAnalytics:
    """Handles behavior trend analysis and visualization generation"""
    
    def __init__(self):
        self.behavior_logs_dir = "behavior_logs"
        self.incident_reports_dir = "incident_reports"
        self.charts_dir = "temp_charts"
        self._ensure_charts_directory()
        logger.info("Behavior analytics module initialized")
    
    def _ensure_charts_directory(self):
        """Ensure temporary charts directory exists"""
        if not os.path.exists(self.charts_dir):
            os.makedirs(self.charts_dir)
    
    def collect_all_behavior_data(self, student_id: Optional[str] = None) -> pd.DataFrame:
        """
        Collect behavior data from all logs or specific student
        
        Parameters:
        - student_id: Optional student ID to filter data, None for all students
        
        Returns:
        - DataFrame with behavior incident data
        """
        all_data = []
        
        try:
            # Collect data from behavior logs
            if os.path.exists(self.behavior_logs_dir):
                for filename in os.listdir(self.behavior_logs_dir):
                    if filename.endswith('.json'):
                        # Filter by student if specified
                        if student_id and not filename.startswith(f"{student_id}_"):
                            continue
                            
                        log_path = os.path.join(self.behavior_logs_dir, filename)
                        try:
                            with open(log_path, 'r') as f:
                                log_data = json.load(f)
                                
                            # Extract relevant data
                            incident_data = log_data.get('incident_data', {})
                            incident_time = datetime.fromisoformat(log_data['incident_time'])
                            
                            behavior_entry = {
                                'student_id': log_data.get('student_id', 'unknown'),
                                'timestamp': incident_time,
                                'date': incident_time.date(),
                                'hour': incident_time.hour,
                                'day_of_week': incident_time.strftime('%A'),
                                'time_period': incident_data.get('time_period', 'unknown'),
                                'severity': incident_data.get('severity', 'medium'),
                                'noise_level_db': incident_data.get('noise_level_db', -70),
                                'location': incident_data.get('location', 'classroom'),
                                'behavior_description': incident_data.get('behavior_description', ''),
                                'keywords': incident_data.get('keywords', []),
                                'is_transition': incident_data.get('is_transition', False),
                                'has_bip': log_data.get('has_bip', False),
                                'outcome': log_data.get('outcome', 'unknown'),
                                'duration_minutes': self._estimate_duration(log_data)
                            }
                            
                            all_data.append(behavior_entry)
                            
                        except Exception as e:
                            logger.warning(f"Error reading behavior log {filename}: {e}")
            
            # Convert to DataFrame
            if all_data:
                df = pd.DataFrame(all_data)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                logger.info(f"Collected {len(df)} behavior incidents for analysis")
                return df
            else:
                logger.info("No behavior data found for analysis")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error collecting behavior data: {e}")
            return pd.DataFrame()
    
    def _estimate_duration(self, log_data: Dict) -> int:
        """Estimate incident duration from log data"""
        # Simple estimation based on severity and outcome
        severity = log_data.get('incident_data', {}).get('severity', 'medium')
        outcome = log_data.get('outcome', 'unknown')
        
        if 'successfully' in outcome.lower():
            if severity == 'high':
                return 15  # 15 minutes for high severity resolved
            elif severity == 'medium':
                return 8   # 8 minutes for medium severity
            else:
                return 5   # 5 minutes for low severity
        else:
            if severity == 'high':
                return 25  # 25 minutes for unresolved high severity
            elif severity == 'medium':
                return 12  # 12 minutes for unresolved medium
            else:
                return 7   # 7 minutes for unresolved low
    
    def generate_behavior_frequency_chart(self, df: pd.DataFrame, student_specific: bool = False) -> str:
        """
        Generate behavior frequency bar chart
        
        Parameters:
        - df: DataFrame with behavior data
        - student_specific: Whether this is for a specific student
        
        Returns:
        - Path to generated chart image
        """
        try:
            if df.empty:
                return None
            
            # Extract behavior types from keywords
            all_keywords = []
            for keywords in df['keywords']:
                if isinstance(keywords, list):
                    all_keywords.extend(keywords)
                elif isinstance(keywords, str):
                    all_keywords.append(keywords)
            
            # Count frequency of behavior types
            if all_keywords:
                keyword_counts = pd.Series(all_keywords).value_counts().head(5)
            else:
                # Fallback to severity if no keywords
                keyword_counts = df['severity'].value_counts().head(5)
            
            # Create chart
            plt.figure(figsize=(10, 6))
            bars = plt.bar(keyword_counts.index, keyword_counts.values, color='skyblue', alpha=0.8)
            plt.title(f'Top 5 Behavior Patterns {"(Student-Specific)" if student_specific else "(Class-Wide)"}',
                     fontsize=14, fontweight='bold')
            plt.xlabel('Behavior Type', fontsize=12)
            plt.ylabel('Frequency', fontsize=12)
            plt.xticks(rotation=45, ha='right')
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{int(height)}', ha='center', va='bottom')
            
            plt.tight_layout()
            
            # Save chart
            chart_path = os.path.join(self.charts_dir, 'behavior_frequency.png')
            plt.savefig(chart_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Generated behavior frequency chart: {chart_path}")
            return chart_path
            
        except Exception as e:
            logger.error(f"Error generating behavior frequency chart: {e}")
            return None
    
    def generate_time_heatmap(self, df: pd.DataFrame, student_specific: bool = False) -> str:
        """
        Generate time of day vs day of week heatmap
        
        Parameters:
        - df: DataFrame with behavior data
        - student_specific: Whether this is for a specific student
        
        Returns:
        - Path to generated chart image
        """
        try:
            if df.empty:
                return None
            
            # Create hour bins
            df['hour_bin'] = df['hour'].apply(lambda x: f"{x:02d}:00-{x+1:02d}:00")
            
            # Create pivot table for heatmap
            heatmap_data = df.pivot_table(
                index='hour_bin',
                columns='day_of_week',
                values='student_id',
                aggfunc='count',
                fill_value=0
            )
            
            # Reorder columns to show weekdays properly
            weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            heatmap_data = heatmap_data.reindex(columns=[day for day in weekday_order if day in heatmap_data.columns])
            
            # Create heatmap
            plt.figure(figsize=(12, 8))
            sns.heatmap(heatmap_data, annot=True, fmt='d', cmap='Reds', cbar_kws={'label': 'Incident Count'})
            plt.title(f'Behavioral Incidents by Time and Day {"(Student-Specific)" if student_specific else "(Class-Wide)"}',
                     fontsize=14, fontweight='bold')
            plt.xlabel('Day of Week', fontsize=12)
            plt.ylabel('Time of Day', fontsize=12)
            plt.tight_layout()
            
            # Save chart
            chart_path = os.path.join(self.charts_dir, 'time_heatmap.png')
            plt.savefig(chart_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Generated time heatmap: {chart_path}")
            return chart_path
            
        except Exception as e:
            logger.error(f"Error generating time heatmap: {e}")
            return None
    
    def generate_noise_escalation_scatter(self, df: pd.DataFrame, student_specific: bool = False) -> str:
        """
        Generate noise level vs escalation scatterplot
        
        Parameters:
        - df: DataFrame with behavior data
        - student_specific: Whether this is for a specific student
        
        Returns:
        - Path to generated chart image
        """
        try:
            if df.empty:
                return None
            
            # Map severity to numeric values
            severity_map = {'low': 1, 'medium': 2, 'high': 3}
            df['severity_numeric'] = df['severity'].map(severity_map).fillna(2)
            
            # Create scatter plot
            plt.figure(figsize=(10, 6))
            colors = {'low': 'green', 'medium': 'orange', 'high': 'red'}
            
            for severity in df['severity'].unique():
                subset = df[df['severity'] == severity]
                plt.scatter(subset['noise_level_db'], subset['duration_minutes'],
                           c=colors.get(severity, 'blue'), label=f'{severity.title()} Severity',
                           alpha=0.7, s=60)
            
            plt.xlabel('Ambient Noise Level (dB)', fontsize=12)
            plt.ylabel('Estimated Duration (minutes)', fontsize=12)
            plt.title(f'Noise Level vs Incident Duration {"(Student-Specific)" if student_specific else "(Class-Wide)"}',
                     fontsize=14, fontweight='bold')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Add trend line
            if len(df) > 1:
                z = np.polyfit(df['noise_level_db'], df['duration_minutes'], 1)
                p = np.poly1d(z)
                plt.plot(df['noise_level_db'], p(df['noise_level_db']), "r--", alpha=0.8, linewidth=2)
            
            plt.tight_layout()
            
            # Save chart
            chart_path = os.path.join(self.charts_dir, 'noise_escalation_scatter.png')
            plt.savefig(chart_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Generated noise escalation scatter plot: {chart_path}")
            return chart_path
            
        except Exception as e:
            logger.error(f"Error generating noise escalation scatter plot: {e}")
            return None
    
    def generate_trend_summary(self, df: pd.DataFrame, student_id: Optional[str] = None) -> Dict:
        """
        Generate summary statistics and insights
        
        Parameters:
        - df: DataFrame with behavior data
        - student_id: Optional student ID for specific analysis
        
        Returns:
        - Dictionary with summary insights
        """
        try:
            if df.empty:
                return {
                    'total_incidents': 0,
                    'insights': ['No behavioral data available for analysis.']
                }
            
            insights = []
            total_incidents = len(df)
            
            # Basic statistics
            insights.append(f"Total incidents analyzed: {total_incidents}")
            
            # Most common behavior
            all_keywords = []
            for keywords in df['keywords']:
                if isinstance(keywords, list):
                    all_keywords.extend(keywords)
            
            if all_keywords:
                most_common = pd.Series(all_keywords).value_counts().index[0]
                insights.append(f"Most common behavior pattern: {most_common}")
            
            # Time patterns
            busiest_hour = df['hour'].mode().values[0] if not df['hour'].empty else 'Unknown'
            busiest_day = df['day_of_week'].mode().values[0] if not df['day_of_week'].empty else 'Unknown'
            insights.append(f"Peak incident time: {busiest_hour}:00 on {busiest_day}s")
            
            # Severity analysis
            severity_dist = df['severity'].value_counts()
            high_severity_pct = (severity_dist.get('high', 0) / total_incidents) * 100
            insights.append(f"High severity incidents: {high_severity_pct:.1f}% of total")
            
            # BIP effectiveness
            if 'has_bip' in df.columns:
                bip_students = df[df['has_bip'] == True]
                if not bip_students.empty:
                    bip_success_rate = len(bip_students[bip_students['outcome'].str.contains('successfully', na=False)]) / len(bip_students) * 100
                    insights.append(f"BIP intervention success rate: {bip_success_rate:.1f}%")
            
            # Environmental factors
            avg_noise = df['noise_level_db'].mean()
            transition_incidents = len(df[df['is_transition'] == True])
            transition_pct = (transition_incidents / total_incidents) * 100
            insights.append(f"Average noise level during incidents: {avg_noise:.1f} dB")
            insights.append(f"Incidents during transitions: {transition_pct:.1f}%")
            
            return {
                'total_incidents': total_incidents,
                'date_range': f"{df['date'].min()} to {df['date'].max()}",
                'student_specific': student_id is not None,
                'insights': insights
            }
            
        except Exception as e:
            logger.error(f"Error generating trend summary: {e}")
            return {
                'total_incidents': 0,
                'insights': [f'Error analyzing data: {str(e)}']
            }
    
    def generate_all_visualizations(self, student_id: Optional[str] = None) -> Dict:
        """
        Generate complete set of visualizations and analysis
        
        Parameters:
        - student_id: Optional student ID for specific analysis
        
        Returns:
        - Dictionary with chart paths and summary data
        """
        try:
            # Collect data
            df = self.collect_all_behavior_data(student_id)
            student_specific = student_id is not None
            
            # Generate charts
            chart_paths = []
            
            freq_chart = self.generate_behavior_frequency_chart(df, student_specific)
            if freq_chart:
                chart_paths.append(freq_chart)
            
            time_chart = self.generate_time_heatmap(df, student_specific)
            if time_chart:
                chart_paths.append(time_chart)
            
            noise_chart = self.generate_noise_escalation_scatter(df, student_specific)
            if noise_chart:
                chart_paths.append(noise_chart)
            
            # Generate summary
            summary = self.generate_trend_summary(df, student_id)
            
            logger.info(f"Generated {len(chart_paths)} visualization(s) for behavior analysis")
            
            return {
                'success': True,
                'chart_paths': chart_paths,
                'summary': summary,
                'student_specific': student_specific,
                'data_available': not df.empty
            }
            
        except Exception as e:
            logger.error(f"Error generating visualizations: {e}")
            return {
                'success': False,
                'error': str(e),
                'chart_paths': [],
                'summary': {'total_incidents': 0, 'insights': ['Error generating analysis']},
                'student_specific': student_id is not None,
                'data_available': False
            }
    
    def cleanup_charts(self):
        """Remove generated chart files to avoid clutter"""
        try:
            if os.path.exists(self.charts_dir):
                for filename in os.listdir(self.charts_dir):
                    if filename.endswith('.png'):
                        file_path = os.path.join(self.charts_dir, filename)
                        os.remove(file_path)
                        logger.info(f"Cleaned up chart: {filename}")
        except Exception as e:
            logger.warning(f"Error cleaning up charts: {e}")