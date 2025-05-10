"""
Context sensing module for behavioral decision support
- Provides time-of-day tagging
- Provides simulated ambient noise level data
"""

import datetime
import logging
import random
from typing import Dict, List, Optional

# Set up logging
logger = logging.getLogger(__name__)

# Default bell schedule (configurable)
DEFAULT_BELL_SCHEDULE = [
    {"name": "pre-class", "start": "07:00", "end": "08:00"},
    {"name": "morning-block-1", "start": "08:00", "end": "09:30"},
    {"name": "morning-break", "start": "09:30", "end": "09:45"},
    {"name": "morning-block-2", "start": "09:45", "end": "11:15"},
    {"name": "lunch", "start": "11:15", "end": "12:00"},
    {"name": "afternoon-block-1", "start": "12:00", "end": "13:30"},
    {"name": "afternoon-break", "start": "13:30", "end": "13:45"},
    {"name": "afternoon-block-2", "start": "13:45", "end": "15:15"},
    {"name": "end-of-day", "start": "15:15", "end": "16:00"},
    {"name": "after-school", "start": "16:00", "end": "18:00"}
]

class ContextSensor:
    """
    Class to capture contextual information from the environment:
    - Time of day (mapped to school periods)
    - Ambient noise levels
    """
    
    def __init__(self, bell_schedule: List[Dict] = None):
        """Initialize the context sensor with optional custom bell schedule"""
        self.bell_schedule = bell_schedule if bell_schedule else DEFAULT_BELL_SCHEDULE
        logger.info(f"ContextSensor initialized with {len(self.bell_schedule)} time periods")
    
    def get_current_time_period(self) -> Dict:
        """
        Get the current time period based on the bell schedule
        
        Returns:
            Dict: containing the period name, start and end times, and a boolean
                 indicating if it's a transition period
        """
        now = datetime.datetime.now()
        current_time_str = now.strftime("%H:%M")
        
        # Convert current time to minutes since midnight for easy comparison
        current_minutes = now.hour * 60 + now.minute
        
        for period in self.bell_schedule:
            # Convert period times to minutes since midnight
            start_h, start_m = map(int, period["start"].split(":"))
            end_h, end_m = map(int, period["end"].split(":"))
            
            start_minutes = start_h * 60 + start_m
            end_minutes = end_h * 60 + end_m
            
            # Check if current time falls within this period
            if start_minutes <= current_minutes < end_minutes:
                # Determine if this is a transition period (breaks, lunch, etc.)
                is_transition = any(keyword in period["name"].lower() 
                                   for keyword in ["break", "lunch", "recess", "transition"])
                
                return {
                    "name": period["name"],
                    "start": period["start"],
                    "end": period["end"],
                    "is_transition": is_transition,
                    "current_time": current_time_str
                }
        
        # If no matching period is found, return a default period
        return {
            "name": "outside-school-hours",
            "start": "00:00",
            "end": "23:59",
            "is_transition": False,
            "current_time": current_time_str
        }
    
    def sample_ambient_noise(self) -> Optional[float]:
        """
        Simulate ambient noise level based on time of day and random variation
        
        Returns:
            float: Simulated noise level in decibels (dB), or None if simulation fails
        """
        try:
            # Get current time period for contextual noise simulation
            time_period = self.get_current_time_period()
            
            # Base noise levels by period type (dB scale, higher = louder)
            # These are simulated values based on typical school environments
            noise_by_period_type = {
                "pre-class": [-65, -45],        # Low-moderate noise before classes start
                "morning-block-1": [-60, -40],  # Moderate classroom noise
                "morning-break": [-40, -20],    # Louder during breaks
                "morning-block-2": [-60, -40],  # Moderate classroom noise
                "lunch": [-35, -15],            # Very loud during lunch
                "afternoon-block-1": [-55, -35],# Moderate classroom noise, slightly louder after lunch
                "afternoon-break": [-40, -20],  # Louder during breaks
                "afternoon-block-2": [-60, -40],# Moderate classroom noise
                "end-of-day": [-45, -25],       # Louder as students prepare to leave
                "after-school": [-65, -45],     # Quieter after most students have left
                "outside-school-hours": [-80, -60]  # Very quiet outside of school hours
            }
            
            # Get noise range for current period
            period_name = time_period["name"]
            noise_range = noise_by_period_type.get(
                period_name, [-60, -40]  # Default range if period not found
            )
            
            # Generate a noise level within the appropriate range for this period
            db = round(random.uniform(noise_range[0], noise_range[1]), 2)
            
            # Determine if this is a transition period and adjust accordingly
            if "break" in period_name or "lunch" in period_name or "transition" in period_name:
                # Add additional noise for transition periods
                db = min(db + random.uniform(5, 15), -10)  # Cap at -10 dB
            
            logger.info(f"Simulated ambient noise for {period_name}: {db:.2f} dB")
            return db
            
        except Exception as e:
            logger.error(f"Error simulating ambient noise: {str(e)}")
            return -50.0  # Return a reasonable default if simulation fails
    
    def get_context_data(self) -> Dict:
        """
        Get complete context data: time period and noise level
        
        Returns:
            Dict: Dictionary containing time_period and noise_level_db
        """
        time_period = self.get_current_time_period()
        noise_level = self.sample_ambient_noise()
        
        return {
            "time_period": time_period,
            "noise_level_db": noise_level
        }


# Create a global instance for use throughout the application
context_sensor = ContextSensor()