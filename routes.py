import logging
import os
import json
from datetime import datetime
from flask import render_template, request, redirect, url_for, flash, session, jsonify
from sqlalchemy import func, and_, or_
from sqlalchemy.ext.declarative import DeclarativeMeta

from app import app, db
from forms import *
from models import *
from ml_models import BehavioralDecisionModel
from voice_recognition import analyze_speech_for_decision, extract_keywords_from_speech
from advanced_nlp import BehaviorQueryProcessor
from context_sensors import ContextSensor, context_sensor
from crisis_alert import crisis_alert_system
from incident_reporting import incident_reporter
from feedback_system import TeacherFeedbackSystem
from teacher_encouragement import TeacherEncouragementSystem
from student_manager import StudentManager
from localization import localization_manager, get_localized_phrase
from voice_clarification import voice_clarification_system
from database_migration import DatabaseMigrationService

# Helper class for JSON serialization of SQLAlchemy objects
class AlchemyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj.__class__, DeclarativeMeta):
            # Handle SQLAlchemy objects
            fields = {}
            for field in [x for x in dir(obj) if not x.startswith('_') and x != 'metadata']:
                data = obj.__getattribute__(field)
                try:
                    # Try to serialize the data
                    json.dumps(data)
                    fields[field] = data
                except TypeError:
                    # Skip non-serializable fields
                    fields[field] = str(data)
            return fields
        return json.JSONEncoder.default(self, obj)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('routes')

# Initialize feedback system
feedback_system = TeacherFeedbackSystem()

# Initialize teacher encouragement system
encouragement_system = TeacherEncouragementSystem()

# Initialize student management system
student_manager = StudentManager()

# Remove this route as we now have a direct route to voice_only at '/'

@app.route('/dashboard')
def dashboard():
    """Dashboard page route"""
    # Count number of behavioral data entries
    data_count = BehavioralData.query.count()
    
    # Count number of protocols
    protocol_count = Protocol.query.count()
    
    # Get recent entries
    recent_entries = BehavioralData.query.order_by(BehavioralData.created_at.desc()).limit(5).all()
    
    return render_template('dashboard.html',
                          data_count=data_count,
                          protocol_count=protocol_count,
                          recent_entries=recent_entries,
                          title="Dashboard")

@app.route('/data_entry', methods=['GET', 'POST'])
def data_entry():
    """Route for entering behavioral data"""
    form = BehavioralDataForm()
    
    # Get list of protocols for the dropdown
    protocols = Protocol.query.all()
    form.protocol_used.choices = [(0, 'None')] + [(p.id, p.name) for p in protocols]
    
    if form.validate_on_submit():
        # Create new behavioral data entry
        new_data = BehavioralData(
            subject_id=form.subject_id.data,
            age=form.age.data,
            gender=form.gender.data,
            context=form.context.data,
            behavior_description=form.behavior_description.data,
            intensity=form.intensity.data,
            frequency=form.frequency.data,
            duration=form.duration.data,
            triggers=form.triggers.data,
            consequences=form.consequences.data,
            outcome=form.outcome.data,
            protocol_used=form.protocol_used.data if form.protocol_used.data > 0 else None
        )
        
        db.session.add(new_data)
        db.session.commit()
        
        flash('Behavioral data saved successfully!', 'success')
        return redirect(url_for('data_entry'))
    
    return render_template('data_entry.html', 
                          form=form, 
                          title="Enter Behavioral Data")

@app.route('/protocols')
def protocols():
    """Route for viewing and managing protocols"""
    all_protocols = Protocol.query.all()
    return render_template('protocols.html', 
                          protocols=all_protocols, 
                          title="Behavioral Protocols")

@app.route('/protocols/add', methods=['GET', 'POST'])
def add_protocol():
    """Route for adding a new protocol"""
    form = ProtocolForm()
    
    if form.validate_on_submit():
        new_protocol = Protocol(
            name=form.name.data,
            description=form.description.data,
            category=form.category.data
        )
        
        db.session.add(new_protocol)
        db.session.commit()
        
        flash(f'Protocol "{form.name.data}" created successfully!', 'success')
        return redirect(url_for('protocol_detail', protocol_id=new_protocol.id))
    
    return render_template('protocol_form.html', 
                          form=form, 
                          title="Create Protocol")

@app.route('/protocols/<int:protocol_id>')
def protocol_detail(protocol_id):
    """Route for viewing a protocol's details and decision points"""
    protocol = Protocol.query.get_or_404(protocol_id)
    decision_points = DecisionPoint.query.filter_by(protocol_id=protocol_id).order_by(DecisionPoint.order).all()
    
    return render_template('protocol_detail.html',
                          protocol=protocol,
                          decision_points=decision_points,
                          title=f"Protocol: {protocol.name}")

@app.route('/protocols/<int:protocol_id>/add_decision', methods=['GET', 'POST'])
def add_decision_point(protocol_id):
    """Route for adding a decision point to a protocol"""
    protocol = Protocol.query.get_or_404(protocol_id)
    form = DecisionPointForm()
    
    # Set default order to be after the last decision point
    last_dp = DecisionPoint.query.filter_by(protocol_id=protocol_id).order_by(DecisionPoint.order.desc()).first()
    if last_dp:
        form.order.data = last_dp.order + 1
    else:
        form.order.data = 1
    
    if form.validate_on_submit():
        new_dp = DecisionPoint(
            protocol_id=protocol_id,
            question=form.question.data,
            order=form.order.data
        )
        
        db.session.add(new_dp)
        db.session.commit()
        
        flash('Decision point added successfully!', 'success')
        return redirect(url_for('decision_point_detail', dp_id=new_dp.id))
    
    return render_template('decision_point_form.html',
                          form=form,
                          protocol=protocol,
                          title="Add Decision Point")

@app.route('/decision_points/<int:dp_id>')
def decision_point_detail(dp_id):
    """Route for viewing a decision point's details and options"""
    dp = DecisionPoint.query.get_or_404(dp_id)
    options = DecisionOption.query.filter_by(decision_point_id=dp_id).all()
    
    return render_template('decision_point_detail.html',
                          dp=dp,
                          options=options,
                          title=f"Decision Point: {dp.id}")

@app.route('/decision_points/<int:dp_id>/add_option', methods=['GET', 'POST'])
def add_option(dp_id):
    """Route for adding an option to a decision point"""
    dp = DecisionPoint.query.get_or_404(dp_id)
    form = DecisionOptionForm()
    
    # Get available decision points for next_decision_id (excluding current one)
    available_dps = DecisionPoint.query.filter(
        DecisionPoint.protocol_id == dp.protocol_id,
        DecisionPoint.id != dp_id
    ).all()
    
    form.next_decision_id.choices = [(0, 'None')] + [(d.id, f"Decision Point {d.id}: {d.question[:50]}...") for d in available_dps]
    
    if form.validate_on_submit():
        new_option = DecisionOption(
            decision_point_id=dp_id,
            text=form.text.data,
            is_terminal=form.is_terminal.data,
            recommendation=form.recommendation.data if form.is_terminal.data else None,
            next_decision_id=form.next_decision_id.data if not form.is_terminal.data and form.next_decision_id.data > 0 else None
        )
        
        db.session.add(new_option)
        db.session.commit()
        
        flash('Option added successfully!', 'success')
        return redirect(url_for('decision_point_detail', dp_id=dp_id))
    
    return render_template('option_form.html',
                          form=form,
                          dp=dp,
                          title="Add Option")

@app.route('/analyze')
def analyze():
    """Route for analyzing behavioral data and training models"""
    # Count data points for summary
    data_count = BehavioralData.query.count()
    
    # Get list of trained models
    ml_models = MLModel.query.all()
    
    return render_template('analyze.html',
                          data_count=data_count,
                          ml_models=ml_models,
                          title="Analyze Data")

@app.route('/decision_support')
def decision_support():
    """Route for the decision support system"""
    form = DecisionSupportForm()
    
    # Get available protocols
    protocols = Protocol.query.all()
    form.protocol_id.choices = [(p.id, p.name) for p in protocols]
    
    return render_template('decision_support.html',
                          form=form,
                          title="Decision Support")

@app.route('/decision_process', methods=['GET', 'POST'])
def decision_process():
    """Route for processing through a decision protocol"""
    # Check if a protocol is selected
    if 'current_protocol_id' not in session:
        if request.method == 'POST':
            protocol_id = request.form.get('protocol_id', type=int)
            if protocol_id:
                session['current_protocol_id'] = protocol_id
            else:
                flash('Please select a protocol', 'warning')
                return redirect(url_for('decision_support'))
        else:
            flash('Please select a protocol first', 'warning')
            return redirect(url_for('decision_support'))
    
    # Get the current protocol
    protocol_id = session['current_protocol_id']
    protocol = Protocol.query.get_or_404(protocol_id)
    
    # If we're just starting, get the first decision point
    if 'current_dp_id' not in session:
        first_dp = DecisionPoint.query.filter_by(protocol_id=protocol_id).order_by(DecisionPoint.order).first()
        if not first_dp:
            flash('This protocol has no decision points defined', 'danger')
            return redirect(url_for('decision_support'))
        session['current_dp_id'] = first_dp.id
    
    # Get the current decision point and its options
    dp_id = session['current_dp_id']
    dp = DecisionPoint.query.get_or_404(dp_id)
    options = DecisionOption.query.filter_by(decision_point_id=dp_id).all()
    
    # Process option selection
    if request.method == 'POST' and 'option_id' in request.form:
        option_id = request.form.get('option_id', type=int)
        selected_option = DecisionOption.query.get_or_404(option_id)
        
        if selected_option.is_terminal:
            # Set session variable for the recommendation
            session['recommendation'] = selected_option.recommendation
            # Clear the current DP ID to restart process next time
            session.pop('current_dp_id', None)
            # Redirect to results
            return redirect(url_for('decision_result'))
        else:
            # Move to the next decision point
            next_dp_id = selected_option.next_decision_id
            if next_dp_id:
                session['current_dp_id'] = next_dp_id
                return redirect(url_for('decision_process'))
            else:
                flash('No next decision point defined for this option', 'danger')
    
    return render_template('decision_process.html',
                          protocol=protocol,
                          decision_point=dp,  # Changed dp to decision_point to match template
                          options=options,
                          title="Decision Support Process")

@app.route('/decision_result')
def decision_result():
    """Route for showing the decision support result"""
    if 'recommendation' not in session:
        flash('No recommendation available. Please complete the decision process.', 'warning')
        return redirect(url_for('decision_support'))
    
    recommendation = session['recommendation']
    
    # Get the protocol info if available
    protocol = None
    if 'current_protocol_id' in session:
        protocol_id = session['current_protocol_id']
        protocol = Protocol.query.get(protocol_id)
    
    return render_template('results.html',
                          recommendation=recommendation,
                          protocol=protocol,
                          title="Decision Support Result")

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """Route for making predictions with trained models"""
    # Create a generic form for model selection
    form = PredictionForm()
    
    # Get available models
    models = MLModel.query.all()
    form.model_id.choices = [(m.id, f"{m.name} - {m.model_type} - {m.target}") for m in models]
    
    return render_template('predict.html',
                          form=form,
                          models=models,
                          title="Prediction")

@app.route('/teacher_input')
def teacher_input():
    """Route for teacher input page with both text and voice options"""
    return render_template('teacher_input.html',
                          title="Teacher Input")

@app.route('/process_teacher_input', methods=['POST'])
def process_teacher_input():
    """Process teacher input from either text or voice form"""
    if request.method == 'POST':
        input_type = request.form.get('input_type')
        
        if input_type == 'text':
            text_input = request.form.get('text_input')
            if not text_input:
                flash('Please enter some text describing the behavior', 'warning')
                return redirect(url_for('teacher_input'))
            
            # Process the text input with our NLP processor
            query_processor = BehaviorQueryProcessor()
            result = query_processor.get_response_for_query(text_input)
            
            # Store the result in session
            session['teacher_result'] = result
            
        elif input_type == 'voice':
            # In a real implementation, we'd process voice input here
            # For demo purposes, use the simulated text
            simulated_voice_text = request.form.get('simulated_voice_text')
            if not simulated_voice_text:
                flash('Please enter simulated voice text', 'warning')
                return redirect(url_for('teacher_input'))
            
            # Process the simulated voice text with our NLP processor
            query_processor = BehaviorQueryProcessor()
            result = query_processor.get_response_for_query(simulated_voice_text)
            
            # Store the result in session
            session['teacher_result'] = result
        
        return redirect(url_for('teacher_result'))
    
    return redirect(url_for('teacher_input'))

@app.route('/teacher_result')
def teacher_result():
    """Route for showing teacher input results"""
    if 'teacher_result' not in session:
        flash('No results available. Please submit input first.', 'warning')
        return redirect(url_for('teacher_input'))
    
    result = session['teacher_result']
    
    return render_template('teacher_result.html',
                          result=result,
                          title="Teacher Input Results")

@app.route('/natural_language_query', methods=['GET', 'POST'])
def natural_language_query():
    """
    Handle natural language queries from teachers using advanced NLP
    with context awareness from sensors
    """
    query_text = ""
    result = None
    
    # Get context data for enhanced analysis
    from context_sensors import context_sensor
    context_data = context_sensor.get_context_data()
    time_period = context_data['time_period']['name']
    noise_level = context_data['noise_level_db']
    is_transition = context_data['time_period']['is_transition']
    
    # Include context information in the template
    context_info = {
        'time_period': time_period.replace('-', ' ').title(),
        'noise_level': f"{noise_level:.1f} dB",
        'is_transition': 'Yes' if is_transition else 'No'
    }
    
    if request.method == 'POST':
        query_text = request.form.get('query', '')
        setting = request.form.get('setting', '')
        
        if query_text:
            # Process the query using our enhanced NLP processor with context
            query_processor = BehaviorQueryProcessor()
            result = query_processor.get_response_for_query(
                query_text,
                setting=setting,
                time_period=time_period,
                noise_level_db=noise_level,
                is_transition_period=is_transition
            )
            
            # Log the query with context for later analysis
            logger.info(f"NL Query: '{query_text}' processed with context - Time: {time_period}, Noise: {noise_level}dB")
            
    return render_template('natural_language_query.html',
                          query_text=query_text,
                          result=result,
                          context=context_info,
                          title="Context-Aware Natural Language Query")

@app.route('/voice_decision_support', methods=['GET', 'POST'])
def voice_decision_support():
    """Route for voice-based decision support"""
    # Create a form for protocol selection
    form = DecisionSupportForm()
    
    # Get available protocols
    protocols = Protocol.query.all()
    form.protocol_id.choices = [(p.id, p.name) for p in protocols]
    
    # Process the form submission
    if form.validate_on_submit():
        protocol_id = form.protocol_id.data
        session['current_protocol_id'] = protocol_id
        return redirect(url_for('voice_input_process'))
    
    # Check if we already have a selected protocol
    selected_protocol = None
    if 'current_protocol_id' in session:
        protocol_id = session['current_protocol_id']
        selected_protocol = Protocol.query.get(protocol_id)
        # Pre-select this protocol in the form
        form.protocol_id.data = protocol_id
    
    return render_template('voice_support.html',
                          form=form,
                          protocols=protocols,
                          selected_protocol=selected_protocol,
                          title="Voice-Based Decision Support")

@app.route('/voice_input_process', methods=['GET', 'POST'])
def voice_input_process():
    """Route for processing voice input for decision support"""
    # Check if a protocol is selected
    if 'current_protocol_id' not in session:
        flash('Please select a protocol first', 'warning')
        return redirect(url_for('voice_decision_support'))
    
    protocol_id = session['current_protocol_id']
    protocol = Protocol.query.get_or_404(protocol_id)
    
    if request.method == 'POST':
        # This is where we would process the voice input in a real environment
        # For now, we'll use a simulated voice input for testing
        
        if 'start_voice' in request.form:
            # Pretend to start voice recognition
            flash('Voice recognition started. Please describe the behavior...', 'info')
            
            # In a real environment, we would call voice_recognizer.listen_once() here
            # and process the result
            
            # For demo purposes, let's simulate a voice input result
            # (in production, this would come from the actual speech recognition)
            simulated_speech = request.form.get('simulated_speech', '')
            
            if simulated_speech:
                # Get context data from the context sensor
                from context_sensors import context_sensor
                context_data = context_sensor.get_context_data()
                time_period = context_data['time_period']['name']
                noise_level = context_data['noise_level_db']
                is_transition = context_data['time_period']['is_transition']
                
                # Log the context data
                logger.info(f"Context data during voice input: Time: {time_period}, Noise: {noise_level}dB, Is Transition: {is_transition}")
                
                # Analyze the speech with context data for decision mapping
                analysis_result = analyze_speech_for_decision(
                    simulated_speech, 
                    protocol_id,
                    time_period=time_period,
                    noise_level_db=noise_level,
                    is_transition_period=is_transition
                )
                
                if analysis_result['success']:
                    # Save behavioral data with context information
                    try:
                        behavior_data = BehavioralData(
                            subject_id='anonymous',  # Anonymous subject for voice input
                            behavior_description=simulated_speech,
                            protocol_used=protocol_id,
                            time_period=time_period,
                            noise_level_db=noise_level,
                            context=f"Voice input during {time_period} period",
                            intensity=8 if analysis_result.get('is_emergency', False) else 5  # Estimated intensity
                        )
                        db.session.add(behavior_data)
                        db.session.commit()
                        logger.info(f"Saved behavioral data with context: ID={behavior_data.id}")
                    except Exception as e:
                        logger.error(f"Error saving behavior data: {str(e)}")
                    
                    # If it's a terminal option, set the recommendation
                    if analysis_result['is_terminal']:
                        session['recommendation'] = analysis_result['recommendation']
                        # Add context data to session
                        session['time_period'] = time_period
                        session['noise_level_db'] = noise_level
                        session['is_transition_period'] = is_transition
                        return redirect(url_for('decision_result'))
                    else:
                        # If not terminal, proceed to the next decision point
                        session['current_dp_id'] = analysis_result['next_decision_id']
                        flash(f"Voice analyzed: '{simulated_speech}'. Proceeding with option: {analysis_result['selected_option']['text']}", 'success')
                        # Display contextual information
                        flash(f"Context: {time_period.replace('-', ' ').title()}, Noise level: {noise_level}dB", 'info')
                        return redirect(url_for('decision_process'))
                else:
                    # If analysis failed, show the error
                    flash(f"Could not process voice input: {analysis_result.get('error', 'Unknown error')}", 'danger')
    
    # If we get here, either it's a GET request or the POST processing didn't result in a redirect
    return render_template('voice_input.html',
                          protocol=protocol,
                          title="Voice Input Processing")

@app.route('/api/context_data', methods=['GET'])
def get_context_data():
    """API endpoint for retrieving current context data"""
    try:
        # Import the context_sensor singleton from the context_sensors module
        from context_sensors import context_sensor
        
        # Get fresh context data
        context_data = context_sensor.get_context_data()
        
        # Return the data as JSON with success flag
        return jsonify({
            'success': True,
            'time_period': context_data['time_period'],
            'noise_level_db': context_data['noise_level_db']
        })
    except Exception as e:
        logger.error(f"Error getting context data: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/voice_only', methods=['GET'])
@app.route('/', methods=['GET'])  # Make this the default landing page
def voice_only():
    """SereniTeach voice interface - primary interface"""
    return render_template('voice_only.html', title="SereniTeach Voice Assistant")
    
@app.route('/text_input', methods=['GET'])
def text_input():
    """Text-based interface for behavioral support with custom input"""
    return render_template('text_input.html', title="Text Input Behavioral Support")

@app.route('/api/context_data', methods=['GET'])
def api_context_data():
    """API endpoint to get current context data"""
    try:
        context_data = context_sensor.get_context_data()
        return jsonify(context_data)
    except Exception as e:
        logger.error(f"Error getting context data: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/voice_capture', methods=['POST'])
@app.route('/test_crisis_alert', methods=['POST'])
def test_crisis_alert():
    """Test route for crisis alert system"""
    try:
        # Create test crisis data
        test_data = {
            'keywords': ['emergency', 'dangerous', 'weapon'],
            'is_emergency': True,
            'severity': 'high',
            'behavior_type': 'violent behavior'
        }
        
        # Test the crisis alert system
        crisis_result = crisis_alert_system.process_behavior_incident(test_data, location="test classroom")
        
        return jsonify({
            'success': True,
            'crisis_detected': crisis_result['crisis_detected'],
            'alert_sent': crisis_result['alert_sent'],
            'details': crisis_result['alert_details']
        })
    except Exception as e:
        logger.error(f"Error testing crisis alert: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/test_incident_workflow', methods=['POST'])
def test_incident_workflow():
    """Test the complete incident reporting workflow"""
    try:
        session_id = "test_session_workflow"
        
        # Step 1: Start a crisis incident
        initial_data = {
            'location': 'test classroom',
            'behavior_description': 'Student showing aggressive behavior with dangerous object',
            'keywords': ['aggressive', 'dangerous', 'weapon'],
            'severity': 'high',
            'time_period': 'instructional',
            'noise_level_db': -45.0,
            'is_transition': False,
            'is_crisis': True
        }
        
        incident_id = incident_reporter.start_incident_log(session_id, initial_data)
        
        # Step 2: Log some interactions during the incident
        incident_reporter.log_voice_input(session_id, "The student is now throwing chairs")
        incident_reporter.log_behavior_update(session_id, "Student escalated to throwing furniture")
        incident_reporter.log_recommendation(session_id, "Remove other students from the area immediately")
        incident_reporter.log_voice_input(session_id, "Student has calmed down slightly")
        incident_reporter.log_recommendation(session_id, "Use de-escalation techniques, speak calmly")
        incident_reporter.log_behavior_update(session_id, "Student is now sitting but still agitated")
        
        # Step 3: End the incident and generate report
        report_result = incident_reporter.end_incident(session_id, outcome="de-escalated successfully")
        
        return jsonify({
            'success': True,
            'workflow_completed': True,
            'incident_id': incident_id,
            'report_generated': report_result['success'],
            'email_sent': report_result.get('email_sent', False),
            'report_filename': report_result.get('report_filename'),
            'report_preview': report_result.get('report_text', '')[:500] + '...' if report_result.get('report_text') else None
        })
        
    except Exception as e:
        logger.error(f"Error testing incident workflow: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/feedback/stats')
def feedback_stats():
    """View feedback statistics"""
    stats = feedback_system.get_feedback_stats()
    recent_feedback = feedback_system.get_recent_feedback(20)
    
    return render_template('feedback_stats.html', 
                         stats=stats, 
                         recent_feedback=recent_feedback)

@app.route('/test_feedback_workflow', methods=['POST'])
def test_feedback_workflow():
    """Test the complete feedback workflow including incident and feedback collection"""
    try:
        session_id = "test_feedback_session"
        
        # Step 1: Simulate crisis detection and incident report
        initial_data = {
            'location': 'classroom',
            'behavior_description': 'Student with aggressive behavior - testing feedback system',
            'keywords': ['aggressive', 'testing'],
            'severity': 'medium',
            'time_period': 'instructional',
            'noise_level_db': -50.0,
            'is_transition': False,
            'is_crisis': True
        }
        
        incident_id = incident_reporter.start_incident_log(session_id, initial_data)
        incident_reporter.log_voice_input(session_id, "Testing the feedback system with this incident")
        incident_reporter.log_recommendation(session_id, "Apply standard de-escalation protocols")
        
        # Step 2: End incident and request feedback
        report_result = incident_reporter.end_incident(session_id, outcome="resolved for testing")
        
        # Step 3: Request feedback
        feedback_request = feedback_system.request_feedback(
            session_id, 
            report_result["incident_id"],
            report_result.get("teacher_email")
        )
        
        # Step 4: Submit positive feedback
        feedback_result = feedback_system.submit_feedback(session_id, 'thumbs_up')
        
        return jsonify({
            'success': True,
            'test_completed': True,
            'incident_id': incident_id,
            'feedback_requested': feedback_request['success'],
            'feedback_submitted': feedback_result['success'],
            'feedback_stats': feedback_system.get_feedback_stats()
        })
        
    except Exception as e:
        logger.error(f"Error testing feedback workflow: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/test_encouragement_workflow', methods=['POST'])
def test_encouragement_workflow():
    """Test the complete teacher encouragement system during crisis scenarios"""
    try:
        session_id = "test_encouragement_session"
        
        # Step 1: Start a crisis incident with encouragement
        initial_data = {
            'location': 'classroom',
            'behavior_description': 'Student displaying escalating aggressive behavior - testing encouragement',
            'keywords': ['aggressive', 'escalating'],
            'severity': 'high',
            'time_period': 'instructional',
            'noise_level_db': -45.0,
            'is_transition': False,
            'is_crisis': True
        }
        
        incident_id = incident_reporter.start_incident_log(session_id, initial_data)
        
        # Step 2: Start teacher encouragement system
        encouragement_result = encouragement_system.start_encouragement(session_id, enabled=True)
        
        # Step 3: Simulate some time passing with encouragement messages
        import time
        time.sleep(2)  # Brief pause to simulate encouragement cycle
        
        # Step 4: Get encouragement status
        encouragement_status = encouragement_system.get_encouragement_status(session_id)
        
        # Step 5: Test toggling encouragement off and on
        toggle_off = encouragement_system.toggle_encouragement(session_id, enable=False)
        toggle_on = encouragement_system.toggle_encouragement(session_id, enable=True)
        
        # Step 6: End the crisis and stop encouragement
        report_result = incident_reporter.end_incident(session_id, outcome="successfully de-escalated")
        encouragement_stop = encouragement_system.stop_encouragement(session_id)
        
        return jsonify({
            'success': True,
            'test_completed': True,
            'incident_id': incident_id,
            'encouragement_started': encouragement_result.get('encouragement_started', False),
            'encouragement_status': encouragement_status,
            'toggle_off_success': toggle_off.get('encouragement_toggled', False),
            'toggle_on_success': toggle_on.get('encouragement_toggled', False),
            'encouragement_stopped': encouragement_stop.get('encouragement_stopped', False),
            'total_encouragements': encouragement_stop.get('total_encouragements', 0),
            'duration_seconds': encouragement_stop.get('duration_seconds', 0)
        })
        
    except Exception as e:
        logger.error(f"Error testing encouragement workflow: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/test_student_bip_workflow', methods=['POST'])
def test_student_bip_workflow():
    """Test the complete student BIP awareness system during behavioral incidents"""
    try:
        session_id = "test_bip_session"
        
        # Test with a student who has a BIP (Johnny Rivera - stu001)
        test_speech = "Johnny Rivera is having an aggressive outburst and throwing items around the classroom"
        
        # Step 1: Identify student from speech
        student_info = student_manager.identify_student_from_speech(test_speech)
        
        # Step 2: Create incident with student information
        if student_info:
            initial_data = {
                'location': 'classroom',
                'behavior_description': test_speech,
                'keywords': ['aggressive', 'throwing'],
                'severity': 'high',
                'time_period': 'instructional',
                'noise_level_db': -45.0,
                'is_transition': False,
                'is_crisis': True,
                'student_info': student_info
            }
            
            # Step 3: Start incident log
            incident_id = incident_reporter.start_incident_log(session_id, initial_data)
            
            # Step 4: Create personalized behavior log
            behavior_log_path = student_manager.create_behavior_log_entry(
                student_info['student_id'], 
                initial_data
            )
            
            # Step 5: Test BIP-enhanced recommendation
            base_recommendation = "Apply standard de-escalation protocols and maintain safe distance."
            enhanced_recommendation = student_manager.get_bip_enhanced_recommendation(
                base_recommendation, student_info
            )
            
            # Step 6: Update behavior log with recommendation
            if behavior_log_path:
                student_manager.update_behavior_log(
                    student_info['student_id'], 
                    behavior_log_path, 
                    {
                        'recommendation': enhanced_recommendation,
                        'bip_enhanced': True
                    }
                )
            
            # Step 7: End incident and update behavior log
            report_result = incident_reporter.end_incident(session_id, outcome="de-escalated using BIP strategies")
            
            if behavior_log_path:
                student_manager.update_behavior_log(
                    student_info['student_id'], 
                    behavior_log_path, 
                    {'outcome': 'successfully resolved with BIP strategies'}
                )
            
            return jsonify({
                'success': True,
                'test_completed': True,
                'student_identified': True,
                'student_info': student_info,
                'incident_id': incident_id,
                'behavior_log_created': behavior_log_path is not None,
                'behavior_log_path': behavior_log_path,
                'bip_enhancement_applied': True,
                'enhanced_recommendation': enhanced_recommendation,
                'test_speech': test_speech
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Student not identified from test speech',
                'test_speech': test_speech
            })
            
    except Exception as e:
        logger.error(f"Error testing student BIP workflow: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/student_roster')
def student_roster():
    """View student roster and BIP information"""
    try:
        # Get roster summary
        summary = student_manager.get_roster_summary()
        
        # Get students with BIPs
        students_with_bips = student_manager.get_all_students_with_bips()
        
        return jsonify({
            'success': True,
            'summary': summary,
            'students_with_bips': students_with_bips
        })
        
    except Exception as e:
        logger.error(f"Error retrieving student roster: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/student_behavior_history/<student_id>')
def student_behavior_history(student_id):
    """View behavior history for a specific student"""
    try:
        # Get behavior history
        history = student_manager.get_student_behavior_history(student_id, days_back=30)
        
        # Get student info
        student_info = None
        students_with_bips = student_manager.get_all_students_with_bips()
        for student in students_with_bips:
            if student['student_id'] == student_id:
                student_info = student
                break
        
        return jsonify({
            'success': True,
            'student_id': student_id,
            'student_info': student_info,
            'behavior_history': history,
            'total_incidents': len(history)
        })
        
    except Exception as e:
        logger.error(f"Error retrieving behavior history for {student_id}: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/student_dashboard')
def student_dashboard():
    """Student management dashboard with BIP awareness"""
    return render_template('student_dashboard.html')

@app.route('/voice_capture', methods=['POST'])
def voice_capture():
    """API endpoint for capturing voice input with context awareness"""
    try:
        # Check if the request contains JSON data
        if request.is_json:
            data = request.get_json()
            speech_text = data.get('text', 'The student is becoming agitated and disruptive in class')
            setting = data.get('setting', 'classroom')
        else:
            speech_text = request.form.get('voice_text', 'The student is becoming agitated and disruptive in class')
            setting = request.form.get('setting', 'classroom')
        
        # Check for language switch commands first
        new_language = localization_manager.detect_language_switch(speech_text)
        if new_language:
            localization_manager.set_language(new_language)
            response_message = get_localized_phrase("language_switched")
            
            return jsonify({
                "success": True,
                "language_switched": True,
                "new_language": new_language,
                "response": response_message,
                "current_language": localization_manager.current_language,
                "message": response_message
            })
        
        # Get context data for enhanced analysis
        context_data = context_sensor.get_context_data()
        time_period = context_data['time_period']['name']
        noise_level = context_data['noise_level_db']
        is_transition = context_data['time_period']['is_transition']
        
        # Log the context data
        logger.info(f"Context data during voice input: Time: {time_period}, Noise: {noise_level}dB, Is Transition: {is_transition}")
        
        # Check for "crisis is over" command in multiple languages
        crisis_over_phrases = localization_manager.get_voice_commands().get('crisis_over', [])
        is_crisis_end = any(phrase in speech_text.lower() for phrase in crisis_over_phrases)
        
        # Get session ID for incident tracking
        session_id = session.get('session_id', 'default_session')
        
        # Handle end of crisis
        if is_crisis_end and incident_reporter.has_active_incident(session_id):
            # End the incident and generate report
            report_result = incident_reporter.end_incident(session_id, outcome="resolved")
            
            localized_message = get_localized_phrase("crisis_over")
            return jsonify({
                "success": True,
                "crisis_ended": True,
                "incident_report": {
                    "generated": report_result["success"],
                    "email_sent": report_result.get("email_sent", False),
                    "filename": report_result.get("report_filename"),
                    "incident_id": report_result.get("incident_id")
                },
                "message": localized_message,
                "current_language": localization_manager.current_language
            })
        
        # Check for student identification in speech
        student_info = student_manager.identify_student_from_speech(speech_text)
        
        # Check if this is a response to a clarification prompt
        clarification_context = session.get('pending_clarification')
        if clarification_context:
            # Process clarification response
            clarification_result = voice_clarification_system.process_clarification_response(
                clarification_context, speech_text
            )
            
            # Clear pending clarification
            session.pop('pending_clarification', None)
            
            if clarification_result['success']:
                # Apply updates from clarification and continue processing
                updates = clarification_result['updates']
                
                # Log clarification interaction
                if incident_reporter.has_active_incident(session_id):
                    incident_reporter.log_interaction(
                        session_id, 
                        'clarification_response', 
                        f"Q: {clarification_context['prompt_text']} A: {speech_text}"
                    )
                
                return jsonify({
                    "success": True,
                    "clarification_processed": True,
                    "updates_applied": updates,
                    "clarification_log": clarification_result['clarification_log'],
                    "message": get_localized_phrase("analyzing"),
                    "current_language": localization_manager.current_language
                })
            else:
                # Clarification failed, ask for clearer response
                return jsonify({
                    "success": False,
                    "clarification_failed": True,
                    "error": clarification_result['error'],
                    "message": clarification_result.get('suggested_prompt', 
                                                       get_localized_phrase('error_processing')),
                    "current_language": localization_manager.current_language
                })
        
        # Process keywords and emergency detection with NLP analysis
        from advanced_nlp import BehaviorQueryProcessor
        nlp_processor = BehaviorQueryProcessor()
        nlp_analysis = nlp_processor.process_teacher_query(speech_text)
        
        keywords = nlp_analysis.get('keywords', [])
        is_emergency = nlp_analysis.get('is_emergency', False)
        
        # Check if clarification is needed based on NLP confidence
        if nlp_analysis.get('needs_clarification', False):
            clarification_needed = voice_clarification_system.needs_clarification(nlp_analysis)
            
            if clarification_needed:
                # Generate clarification prompt
                clarification_prompt = voice_clarification_system.get_clarification_prompt(
                    clarification_needed['type'], 
                    context={'keywords': keywords, 'description': speech_text}
                )
                
                if clarification_prompt:
                    # Store clarification context in session
                    session['pending_clarification'] = clarification_prompt
                    
                    # Log clarification request
                    if incident_reporter.has_active_incident(session_id):
                        incident_reporter.log_interaction(
                            session_id, 
                            'clarification_requested', 
                            f"Low confidence ({clarification_needed['confidence']:.2f}) - {clarification_prompt['prompt_text']}"
                        )
                    
                    status_message = voice_clarification_system.get_clarification_status_message(
                        clarification_needed['type']
                    )
                    
                    return jsonify({
                        "success": True,
                        "clarification_needed": True,
                        "clarification_type": clarification_needed['type'],
                        "confidence": clarification_needed['confidence'],
                        "reason": clarification_needed['reason'],
                        "status_message": status_message,
                        "prompt": clarification_prompt['prompt_text'],
                        "message": f"{status_message} {clarification_prompt['prompt_text']}",
                        "current_language": localization_manager.current_language
                    })
        
        # Continue with original emergency detection as fallback
        if not is_emergency:
            is_emergency = ('emergency' in speech_text.lower() or 'urgent' in speech_text.lower() or 
                          'immediate' in speech_text.lower() or 'danger' in speech_text.lower())
        
        # Adjust emergency detection based on context
        # Higher noise levels or transition periods might lead to misinterpretations
        if noise_level > -40 and not any(kw in speech_text.lower() for kw in ['emergency', 'urgent', 'danger']):
            # In very noisy environments, be more conservative about emergency detection
            is_emergency = False
        
        # If it's a transition period, certain behaviors might be more expected
        context_note = ""
        if is_transition:
            context_note = "Note: This is occurring during a transition period, which may affect behavior patterns."
        elif noise_level > -50:
            context_note = "Note: Current noise levels are elevated, which may impact behavior."
        
        # Save behavioral data with context information
        try:
            # Default protocol ID (if not specified)
            protocol_id = session.get('current_protocol_id', 1)
            
            behavior_data = BehavioralData(
                subject_id='anonymous',  # Anonymous subject for voice input
                behavior_description=speech_text,
                protocol_used=protocol_id,
                time_period=time_period,
                noise_level_db=noise_level,
                context=f"Voice input during {time_period} period, setting: {setting}",
                intensity=8 if is_emergency else 5  # Estimated intensity
            )
            db.session.add(behavior_data)
            db.session.commit()
            logger.info(f"Saved behavioral data with context: ID={behavior_data.id}")
        except Exception as e:
            logger.error(f"Error saving behavior data: {str(e)}")
        
        # Enhanced result with context data
        result = {
            "success": True,
            "text": speech_text,
            "context": {
                "time_period": time_period,
                "noise_level_db": noise_level,
                "is_transition_period": is_transition
            },
            "analysis": {
                "keywords": keywords,
                "is_emergency": is_emergency,
                "sentiment": "concerned" if "worried" in speech_text.lower() else "neutral",
                "context_note": context_note
            }
        }
        
        # Process with protocol if one is selected
        protocol_id = session.get('current_protocol_id', 1)
        
        # Analyze the speech for decision support with context data
        analysis = analyze_speech_for_decision(
            result["text"], 
            protocol_id,
            time_period=time_period,
            noise_level_db=noise_level,
            is_transition_period=is_transition,
            setting=setting
        )
        
        # Enhance analysis with BIP-aware recommendations if student identified
        if student_info and student_info.get('has_bip', False):
            if analysis and 'recommendation_text' in analysis:
                original_recommendation = analysis['recommendation_text']
                enhanced_recommendation = student_manager.get_bip_enhanced_recommendation(
                    original_recommendation, student_info
                )
                analysis['recommendation_text'] = enhanced_recommendation
                analysis['bip_enhanced'] = True
                logger.info(f"Applied BIP enhancement for student {student_info['first_name']} {student_info['last_name']}")
        
        # Add the analysis results to our response
        result["analysis"]["protocol_id"] = protocol_id
        result["analysis"]["protocol_analysis"] = analysis
        
        # Add student information to response
        if student_info:
            result["student_info"] = student_info
        
        # Check for crisis and send email alert if needed
        crisis_result = crisis_alert_system.process_behavior_incident(
            result["analysis"], 
            location=setting
        )
        
        # Add crisis alert information to response
        result["crisis_alert"] = crisis_result
        
        # Handle incident reporting
        if crisis_result["crisis_detected"]:
            logger.warning(f"Crisis detected and alert {'sent' if crisis_result['alert_sent'] else 'failed'}")
            
            # Start incident log if this is a new crisis
            if not incident_reporter.has_active_incident(session_id):
                initial_data = {
                    'location': setting,
                    'behavior_description': speech_text,
                    'keywords': keywords,
                    'severity': 'high' if is_emergency else 'medium',
                    'time_period': time_period,
                    'noise_level_db': noise_level,
                    'is_transition': is_transition,
                    'is_crisis': True,
                    'student_info': student_info  # Include student information
                }
                incident_id = incident_reporter.start_incident_log(session_id, initial_data)
                result["incident_started"] = True
                result["incident_id"] = incident_id
                logger.info(f"Started incident log {incident_id} for crisis")
                
                # Create personalized behavior log if student identified
                if student_info:
                    behavior_log_path = student_manager.create_behavior_log_entry(
                        student_info['student_id'], 
                        initial_data
                    )
                    result["behavior_log_created"] = behavior_log_path is not None
                    result["student_identified"] = True
                    logger.info(f"Created behavior log for student {student_info['first_name']} {student_info['last_name']}")
                
                # Start teacher encouragement system for crisis support
                encouragement_result = encouragement_system.start_encouragement(session_id, enabled=True)
                result["encouragement_started"] = encouragement_result.get("encouragement_started", False)
                logger.info(f"Started teacher encouragement for crisis session {session_id}")
        
        # Check for "crisis is over" command
        crisis_end_phrases = ['crisis is over', 'crisis over', 'incident is over', 'incident over', 'emergency is over', 'emergency over']
        if any(phrase in speech_text.lower() for phrase in crisis_end_phrases):
            if incident_reporter.has_active_incident(session_id):
                # End the incident and generate report
                report_result = incident_reporter.end_incident(session_id, outcome="teacher declared resolved")
                result["crisis_ended"] = True
                result["incident_report"] = report_result
                
                # Stop teacher encouragement system
                if encouragement_system.has_active_encouragement(session_id):
                    encouragement_stop = encouragement_system.stop_encouragement(session_id)
                    result["encouragement_stopped"] = encouragement_stop.get("encouragement_stopped", False)
                    logger.info(f"Stopped teacher encouragement for ended crisis")
                
                # Always request feedback after incident report is generated
                if report_result["success"]:
                    feedback_request = feedback_system.request_feedback(
                        session_id, 
                        report_result["incident_id"],
                        report_result.get("teacher_email", "stevenrayhinojosa@gmail.com")
                    )
                    result["feedback_request"] = feedback_request
                    
                    if report_result["email_sent"]:
                        result["message"] = "Crisis ended. Incident report generated and emailed. " + feedback_request.get("message", "")
                    else:
                        result["message"] = "Crisis ended. Incident report generated but email failed. " + feedback_request.get("message", "")
                else:
                    result["message"] = "Crisis ended but report generation failed."
            else:
                result["message"] = "No active crisis to end."
        
        # Check for encouragement toggle commands
        elif 'turn off encouragement' in speech_text.lower() or 'disable encouragement' in speech_text.lower():
            if encouragement_system.has_active_encouragement(session_id):
                toggle_result = encouragement_system.toggle_encouragement(session_id, enable=False)
                result["encouragement_toggled"] = toggle_result.get("encouragement_toggled", False)
                result["message"] = toggle_result.get("message", "Encouragement turned off.")
            else:
                result["message"] = "No active encouragement to turn off."
        
        elif 'turn on encouragement' in speech_text.lower() or 'enable encouragement' in speech_text.lower():
            if encouragement_system.has_active_encouragement(session_id):
                toggle_result = encouragement_system.toggle_encouragement(session_id, enable=True)
                result["encouragement_toggled"] = toggle_result.get("encouragement_toggled", False)
                result["message"] = toggle_result.get("message", "Encouragement turned on.")
            elif incident_reporter.has_active_incident(session_id):
                # Start encouragement if there's an active crisis
                start_result = encouragement_system.start_encouragement(session_id, enabled=True)
                result["encouragement_started"] = start_result.get("encouragement_started", False)
                result["message"] = "Encouragement started for active crisis."
            else:
                result["message"] = "No active crisis for encouragement."
        
        # Check for feedback responses
        elif feedback_system.has_pending_feedback(session_id):
            if 'thumbs up' in speech_text.lower() or 'positive' in speech_text.lower() or 'helpful' in speech_text.lower():
                feedback_result = feedback_system.submit_feedback(session_id, 'thumbs_up')
                result["feedback_submitted"] = feedback_result
                result["message"] = feedback_result.get("message", "Feedback recorded.")
            elif 'thumbs down' in speech_text.lower() or 'negative' in speech_text.lower() or 'not helpful' in speech_text.lower():
                # Ask for optional comment
                result["feedback_type"] = "negative"
                result["ask_for_comment"] = True
                result["message"] = "Would you like to leave a quick comment about what didn't work?"
            elif 'comment' in speech_text.lower() and len(speech_text.split()) > 2:
                # Extract comment (remove the word "comment" and submit feedback)
                comment_text = speech_text.replace('comment', '').strip()
                feedback_result = feedback_system.submit_feedback(session_id, 'thumbs_down', comment_text)
                result["feedback_submitted"] = feedback_result
                result["message"] = feedback_result.get("message", "Feedback and comment recorded.")
        
        # Log this interaction if there's an active incident
        elif incident_reporter.has_active_incident(session_id):
            incident_reporter.log_voice_input(session_id, speech_text)
            
            # Log any recommendations provided
            if analysis and 'recommendation' in analysis:
                recommendation_text = analysis.get('recommendation', '')
                if recommendation_text:
                    incident_reporter.log_recommendation(session_id, recommendation_text)
            
            # Log behavior updates if this describes a change in behavior
            behavior_change_indicators = ['now', 'started', 'stopped', 'began', 'is becoming', 'turned', 'changed']
            if any(indicator in speech_text.lower() for indicator in behavior_change_indicators):
                incident_reporter.log_behavior_update(session_id, speech_text)
        
        # Add encouragement status and latest message to response
        encouragement_status = encouragement_system.get_encouragement_status(session_id)
        result["encouragement_status"] = encouragement_status
        
        latest_encouragement = encouragement_system.get_latest_encouragement(session_id)
        if latest_encouragement:
            result["latest_encouragement"] = latest_encouragement
        
        # Return the results using our custom AlchemyEncoder for SQLAlchemy objects
        return app.response_class(
            response=json.dumps(result, cls=AlchemyEncoder),
            status=200,
            mimetype='application/json'
        )
            
    except Exception as e:
        logger.error(f"Error processing voice input: {str(e)}")
        return app.response_class(
            response=json.dumps({
                "success": False,
                "error": str(e)
            }),
            status=500,
            mimetype='application/json'
        )
        
        # Adjust emergency detection based on context
        # Higher noise levels or transition periods might lead to misinterpretations
        if noise_level > -40 and not any(kw in simulated_text.lower() for kw in ['emergency', 'urgent', 'danger']):
            # In very noisy environments, be more conservative about emergency detection
            is_emergency = False
        
        # If it's a transition period, certain behaviors might be more expected
        context_note = ""
        if is_transition:
            context_note = "Note: This is occurring during a transition period, which may affect behavior patterns."
        elif noise_level > -50:
            context_note = "Note: Current noise levels are elevated, which may impact behavior."
        
        # Enhanced result with context data
        result = {
            "success": True,
            "text": simulated_text,
            "context": {
                "time_period": time_period,
                "noise_level_db": noise_level,
                "is_transition_period": is_transition
            },
            "analysis": {
                "keywords": keywords,
                "is_emergency": is_emergency,
                "sentiment": "concerned" if "worried" in simulated_text.lower() else "neutral",
                "context_note": context_note
            }
        }
        
        # Process with protocol if one is selected
        protocol_id = session.get('current_protocol_id')
        if protocol_id:
            # Analyze the speech for decision support with context data
            analysis = analyze_speech_for_decision(
                result["text"], 
                protocol_id,
                time_period=time_period,
                noise_level_db=noise_level,
                is_transition_period=is_transition
            )
            
            # Add the analysis results to our response
            result["analysis"]["protocol_id"] = protocol_id
            result["analysis"]["protocol_analysis"] = analysis
        else:
            # If no protocol selected, add that information to the result
            result["analysis"]["protocol_status"] = "No protocol selected"
            
        # Return the results
        return jsonify(result)
            
    except Exception as e:
        logger.error(f"Error processing voice input: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        })

@app.route('/generate_test_data')
def generate_test_data():
    """Generate synthetic behavioral data for testing trend visualizations"""
    try:
        import random
        from datetime import timedelta
        
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
        generated_files = []
        
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
            
            generated_files.append(filename)
        
        return jsonify({
            'success': True,
            'message': f'Generated {len(generated_files)} test behavior incidents',
            'files_created': len(generated_files),
            'data_summary': {
                'incidents': len(generated_files),
                'students_involved': len(set([log_entry['student_id'] for _ in range(18)])),
                'date_range': f"{base_date.strftime('%Y-%m-%d')} to {datetime.now().strftime('%Y-%m-%d')}",
                'patterns': 'Students with BIPs have more incidents, peak during lunch hours'
            }
        })
        
    except Exception as e:
        logger.error(f"Error generating test data: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/test_analytics')
def test_analytics():
    """Test route for behavior analytics and visualization generation"""
    try:
        from behavior_analytics import BehaviorAnalytics
        
        analytics = BehaviorAnalytics()
        
        # Generate visualizations for all students
        result = analytics.generate_all_visualizations()
        
        return jsonify({
            'success': result['success'],
            'data_available': result['data_available'],
            'charts_generated': len(result.get('chart_paths', [])),
            'summary': result.get('summary', {}),
            'student_specific': result['student_specific'],
            'chart_files': [os.path.basename(path) for path in result.get('chart_paths', [])]
        })
        
    except Exception as e:
        logger.error(f"Error testing analytics: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/test_analytics/<student_id>')
def test_student_analytics(student_id):
    """Test route for student-specific behavior analytics"""
    try:
        from behavior_analytics import BehaviorAnalytics
        
        analytics = BehaviorAnalytics()
        
        # Generate visualizations for specific student
        result = analytics.generate_all_visualizations(student_id)
        
        return jsonify({
            'success': result['success'],
            'student_id': student_id,
            'data_available': result['data_available'],
            'charts_generated': len(result.get('chart_paths', [])),
            'summary': result.get('summary', {}),
            'student_specific': result['student_specific'],
            'chart_files': [os.path.basename(path) for path in result.get('chart_paths', [])]
        })
        
    except Exception as e:
        logger.error(f"Error testing student analytics: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/language_status')
def language_status():
    """Get current language status and available commands"""
    try:
        return jsonify({
            'success': True,
            'current_language': localization_manager.current_language,
            'supported_languages': localization_manager.supported_languages,
            'voice_commands': localization_manager.get_voice_commands(),
            'sample_phrases': {
                'crisis_detected': get_localized_phrase('crisis_detected'),
                'stay_calm': get_localized_phrase('stay_calm'),
                'feedback_request': get_localized_phrase('feedback_request'),
                'encouragement': localization_manager.get_encouragement()
            }
        })
    except Exception as e:
        logger.error(f"Error getting language status: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/test_language_switch', methods=['POST'])
def test_language_switch():
    """Test language switching functionality"""
    try:
        command = request.json.get('command', 'switch to spanish')
        detected_language = localization_manager.detect_language_switch(command)
        
        if detected_language:
            localization_manager.set_language(detected_language)
            response_message = get_localized_phrase("language_switched")
            
            return jsonify({
                'success': True,
                'command_tested': command,
                'detected_language': detected_language,
                'switched_to': localization_manager.current_language,
                'response_message': response_message,
                'sample_responses': {
                    'crisis_detected': get_localized_phrase('crisis_detected'),
                    'stay_calm': get_localized_phrase('stay_calm'),
                    'student_identified': get_localized_phrase('student_identified', 
                                                             name='María García', 
                                                             bip_status=get_localized_phrase('has_bip')),
                    'encouragement': localization_manager.get_encouragement()
                }
            })
        else:
            return jsonify({
                'success': False,
                'command_tested': command,
                'error': 'Language switch command not recognized',
                'current_language': localization_manager.current_language
            })
            
    except Exception as e:
        logger.error(f"Error testing language switch: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/test_multilingual_responses')
def test_multilingual_responses():
    """Test various system responses in both languages"""
    try:
        responses = {}
        
        for lang in ['en', 'es']:
            localization_manager.set_language(lang)
            
            responses[lang] = {
                'language_name': 'English' if lang == 'en' else 'Español',
                'crisis_responses': {
                    'crisis_detected': get_localized_phrase('crisis_detected'),
                    'stay_calm': get_localized_phrase('stay_calm'),
                    'safety_first': get_localized_phrase('safety_first'),
                    'crisis_over': get_localized_phrase('crisis_over')
                },
                'student_responses': {
                    'student_with_bip': get_localized_phrase('student_identified', 
                                                           name='Alex Johnson', 
                                                           bip_status=get_localized_phrase('has_bip')),
                    'student_no_bip': get_localized_phrase('student_identified', 
                                                         name='Sarah Wilson', 
                                                         bip_status=get_localized_phrase('no_bip'))
                },
                'behavior_types': {
                    'disruption': localization_manager.localize_behavior_type('disruption'),
                    'aggression': localization_manager.localize_behavior_type('aggression'),
                    'defiance': localization_manager.localize_behavior_type('defiance')
                },
                'severity_levels': {
                    'low': localization_manager.localize_severity('low'),
                    'medium': localization_manager.localize_severity('medium'),
                    'high': localization_manager.localize_severity('high')
                },
                'encouragement': localization_manager.get_encouragement(),
                'voice_commands': localization_manager.get_voice_commands()
            }
        
        return jsonify({
            'success': True,
            'responses_by_language': responses,
            'current_language': localization_manager.current_language
        })
        
    except Exception as e:
        logger.error(f"Error testing multilingual responses: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/test_voice_clarification', methods=['POST'])
def test_voice_clarification():
    """Test voice clarification system with ambiguous scenarios"""
    try:
        test_scenario = request.json.get('scenario', 'low_confidence_behavior')
        
        # Predefined test scenarios with low confidence
        test_scenarios = {
            'low_confidence_behavior': {
                'speech_text': "The student is doing something disruptive but I'm not sure what exactly",
                'expected_clarification': 'behavior_type',
                'description': 'Ambiguous behavior description requiring clarification'
            },
            'unclear_severity': {
                'speech_text': "There's some kind of issue with the student",
                'expected_clarification': 'severity',
                'description': 'Unclear severity level requiring assessment'
            },
            'potential_emergency': {
                'speech_text': "Something might be wrong, the student seems upset",
                'expected_clarification': 'emergency',
                'description': 'Uncertain emergency status requiring confirmation'
            },
            'physical_contact_unclear': {
                'speech_text': "The student was near another student and now they're upset",
                'expected_clarification': 'behavior_type',
                'description': 'Unclear if physical contact occurred'
            },
            'escalation_uncertain': {
                'speech_text': "The behavior seems to be changing but I can't tell if it's getting worse",
                'expected_clarification': 'severity',
                'description': 'Uncertain about behavior escalation'
            }
        }
        
        if test_scenario not in test_scenarios:
            return jsonify({
                'success': False,
                'error': f'Unknown test scenario: {test_scenario}',
                'available_scenarios': list(test_scenarios.keys())
            }), 400
        
        scenario_data = test_scenarios[test_scenario]
        speech_text = scenario_data['speech_text']
        
        # Process with NLP to generate low confidence analysis
        from advanced_nlp import BehaviorQueryProcessor
        nlp_processor = BehaviorQueryProcessor()
        nlp_analysis = nlp_processor.process_teacher_query(speech_text)
        
        # Simulate low confidence for testing by checking if clarification is needed
        clarification_needed = voice_clarification_system.needs_clarification({
            'confidence': {
                'behavior_type': 0.4,
                'severity': 0.4,
                'emergency': 0.4
            }
        })
        
        if clarification_needed:
            # Generate clarification prompt
            clarification_prompt = voice_clarification_system.get_clarification_prompt(
                clarification_needed['type'],
                context={'keywords': nlp_analysis.get('keywords', []), 'description': speech_text}
            )
            
            return jsonify({
                'success': True,
                'test_scenario': test_scenario,
                'scenario_description': scenario_data['description'],
                'original_speech': speech_text,
                'nlp_analysis': nlp_analysis,
                'clarification_needed': clarification_needed,
                'clarification_prompt': clarification_prompt,
                'expected_clarification_type': scenario_data['expected_clarification'],
                'test_result': 'PASS' if clarification_needed['type'] == scenario_data['expected_clarification'] else 'FAIL'
            })
        else:
            return jsonify({
                'success': True,
                'test_scenario': test_scenario,
                'scenario_description': scenario_data['description'],
                'original_speech': speech_text,
                'nlp_analysis': nlp_analysis,
                'clarification_needed': None,
                'test_result': 'FAIL - No clarification triggered when expected'
            })
        
    except Exception as e:
        logger.error(f"Error testing voice clarification: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/test_clarification_response', methods=['POST'])
def test_clarification_response():
    """Test processing of clarification responses"""
    try:
        clarification_type = request.json.get('clarification_type', 'behavior_type')
        prompt_key = request.json.get('prompt_key', 'aggression_physical')
        response_text = request.json.get('response', 'yes')
        
        # Create mock clarification context
        clarification_context = {
            'prompt_type': clarification_type,
            'prompt_key': prompt_key,
            'prompt_text': f"Test prompt for {clarification_type}:{prompt_key}",
            'language': localization_manager.current_language
        }
        
        # Process the response
        clarification_result = voice_clarification_system.process_clarification_response(
            clarification_context, response_text
        )
        
        return jsonify({
            'success': True,
            'clarification_context': clarification_context,
            'response_text': response_text,
            'processing_result': clarification_result,
            'test_status': 'PASS' if clarification_result['success'] else 'FAIL'
        })
        
    except Exception as e:
        logger.error(f"Error testing clarification response: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/clarification_status')
def clarification_status():
    """Get current clarification system status and configuration"""
    try:
        from voice_clarification import NLP_CONFIDENCE_THRESHOLD
        
        return jsonify({
            'success': True,
            'system_status': 'active',
            'confidence_threshold': NLP_CONFIDENCE_THRESHOLD,
            'supported_clarification_types': ['behavior_type', 'severity', 'emergency'],
            'current_language': localization_manager.current_language,
            'available_prompts': {
                'behavior_type': list(voice_clarification_system.clarification_prompts['behavior_type'].keys()),
                'severity': list(voice_clarification_system.clarification_prompts['severity'].keys()),
                'emergency': list(voice_clarification_system.clarification_prompts['emergency'].keys())
            },
            'sample_scenarios': [
                'low_confidence_behavior',
                'unclear_severity', 
                'potential_emergency',
                'physical_contact_unclear',
                'escalation_uncertain'
            ]
        })
        
    except Exception as e:
        logger.error(f"Error getting clarification status: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
