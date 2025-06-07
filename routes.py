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
        
        # Get context data for enhanced analysis
        context_data = context_sensor.get_context_data()
        time_period = context_data['time_period']['name']
        noise_level = context_data['noise_level_db']
        is_transition = context_data['time_period']['is_transition']
        
        # Log the context data
        logger.info(f"Context data during voice input: Time: {time_period}, Noise: {noise_level}dB, Is Transition: {is_transition}")
        
        # Check for "crisis is over" command
        crisis_over_phrases = ['crisis is over', 'crisis over', 'incident is over', 'incident over', 'situation resolved']
        is_crisis_end = any(phrase in speech_text.lower() for phrase in crisis_over_phrases)
        
        # Get session ID for incident tracking
        session_id = session.get('session_id', 'default_session')
        
        # Handle end of crisis
        if is_crisis_end and incident_reporter.has_active_incident(session_id):
            # End the incident and generate report
            report_result = incident_reporter.end_incident(session_id, outcome="resolved")
            
            return jsonify({
                "success": True,
                "crisis_ended": True,
                "incident_report": {
                    "generated": report_result["success"],
                    "email_sent": report_result.get("email_sent", False),
                    "filename": report_result.get("report_filename"),
                    "incident_id": report_result.get("incident_id")
                },
                "message": "Crisis ended. Incident report generated and emailed."
            })
        
        # Process keywords and emergency detection
        keywords = extract_keywords_from_speech(speech_text)
        
        # Determine if emergency based on keywords and context
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
        
        # Add the analysis results to our response
        result["analysis"]["protocol_id"] = protocol_id
        result["analysis"]["protocol_analysis"] = analysis
        
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
                    'is_crisis': True
                }
                incident_id = incident_reporter.start_incident_log(session_id, initial_data)
                result["incident_started"] = True
                result["incident_id"] = incident_id
                logger.info(f"Started incident log {incident_id} for crisis")
        
        # Check for "crisis is over" command
        crisis_end_phrases = ['crisis is over', 'crisis over', 'incident is over', 'incident over', 'emergency is over', 'emergency over']
        if any(phrase in speech_text.lower() for phrase in crisis_end_phrases):
            if incident_reporter.has_active_incident(session_id):
                # End the incident and generate report
                report_result = incident_reporter.end_incident(session_id, outcome="teacher declared resolved")
                result["crisis_ended"] = True
                result["incident_report"] = report_result
                
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
