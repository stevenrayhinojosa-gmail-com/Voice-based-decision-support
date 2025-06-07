"""
Generate test behavioral data for demonstrating the trend visualization system
"""

import json
import os
import random
from datetime import datetime, timedelta
from student_manager import student_manager

def generate_test_behavior_logs():
    """Generate realistic test behavior logs for multiple students"""
    
    # Ensure behavior_logs directory exists
    if not os.path.exists('behavior_logs'):
        os.makedirs('behavior_logs')
    
    # Get student roster
    students = student_manager.get_all_students()
    
    # Behavior patterns for realistic data
    behavior_types = {
        'disruption': ['talking', 'noise', 'interrupting', 'calling out'],
        'defiance': ['refusing', 'arguing', 'non-compliance'],
        'aggression': ['hitting', 'throwing', 'yelling', 'pushing'],
        'off-task': ['wandering', 'distracted', 'daydreaming'],
        'social': ['teasing', 'inappropriate comments', 'exclusion']
    }
    
    severity_levels = ['low', 'medium', 'high']
    locations = ['classroom', 'hallway', 'cafeteria', 'playground']
    outcomes = [
        'Crisis resolved successfully with de-escalation',
        'Student responded well to intervention',
        'Situation required additional support',
        'Crisis ended with student calming down'
    ]
    
    # Generate 15-20 incidents across the past 2 weeks
    base_date = datetime.now() - timedelta(days=14)
    
    for i in range(18):
        # Select random student (some more likely than others for realistic patterns)
        student_weights = [3 if s.get('has_bip') else 1 for s in students]
        student = random.choices(students, weights=student_weights)[0]
        
        # Generate incident time (more likely during school hours)
        days_offset = random.randint(0, 13)
        hour = random.choices(
            range(8, 16),  # School hours 8 AM to 4 PM
            weights=[1, 2, 3, 4, 3, 2, 2, 1]  # Peak around 11 AM - 1 PM
        )[0]
        minute = random.randint(0, 59)
        
        incident_time = base_date + timedelta(days=days_offset, hours=hour, minutes=minute)
        
        # Generate behavior data
        behavior_category = random.choice(list(behavior_types.keys()))
        keywords = random.sample(behavior_types[behavior_category], random.randint(1, 2))
        
        severity = random.choices(
            severity_levels,
            weights=[4, 3, 1]  # More low/medium severity
        )[0]
        
        # Generate environmental data
        noise_level = random.uniform(-80, -50)  # dB range
        if hour in [11, 12, 13]:  # Lunch time - noisier
            noise_level += 10
        
        location = random.choice(locations)
        is_transition = random.random() < 0.2  # 20% during transitions
        
        # Create behavior log entry
        log_entry = {
            'incident_id': f'test_{i+1:03d}',
            'session_id': f'session_test_{i+1}',
            'student_id': student['student_id'],
            'has_bip': student.get('has_bip', False),
            'incident_time': incident_time.isoformat(),
            'end_time': (incident_time + timedelta(minutes=random.randint(5, 25))).isoformat(),
            'location': location,
            'outcome': random.choice(outcomes),
            'incident_data': {
                'behavior_description': f"{behavior_category.title()} behavior involving {', '.join(keywords)}",
                'keywords': keywords,
                'severity': severity,
                'noise_level_db': round(noise_level, 2),
                'time_period': f"Period {random.randint(1, 7)}",
                'is_transition': is_transition,
                'location': location
            }
        }
        
        # Save to individual student file
        filename = f"{student['student_id']}_behavior_{incident_time.strftime('%Y%m%d_%H%M%S')}.json"
        filepath = os.path.join('behavior_logs', filename)
        
        with open(filepath, 'w') as f:
            json.dump(log_entry, f, indent=2)
        
        print(f"Generated behavior log: {filename}")
    
    print(f"\nGenerated {18} test behavior incidents across {len(students)} students")
    print("Data includes realistic patterns:")
    print("- Students with BIPs have more incidents")
    print("- Peak incidents during lunch hours")
    print("- Various severity levels and behavior types")
    print("- Environmental noise correlation")

if __name__ == "__main__":
    generate_test_behavior_logs()