import json
import logging
import pandas as pd
from flask import render_template, request, redirect, url_for, flash, jsonify, session
from app import app, db
from models import Protocol, DecisionPoint, DecisionOption, BehavioralData, MLModel
from forms import (BehavioralDataForm, ProtocolForm, DecisionPointForm, 
                  DecisionOptionForm, ModelTrainingForm, DecisionSupportForm,
                  PredictionForm)
from utils import load_behavioral_data_to_dataframe, preprocess_behavioral_data, get_protocol_tree, add_sample_protocol
from ml_models import BehavioralDecisionModel
from voice_recognition import voice_recognizer, analyze_speech_for_decision

# Setup logging
logger = logging.getLogger(__name__)

# Initialize the ML model
behavioral_model = BehavioralDecisionModel()

@app.route('/')
def index():
    """Home page route"""
    # Count protocols and behavioral data entries for the dashboard
    protocol_count = Protocol.query.count()
    data_count = BehavioralData.query.count()
    model_count = MLModel.query.count()
    
    # If there are no protocols yet, add a sample one
    if protocol_count == 0:
        add_sample_protocol()
        protocol_count = Protocol.query.count()
    
    return render_template('index.html', 
                           protocol_count=protocol_count, 
                           data_count=data_count,
                           model_count=model_count)

@app.route('/dashboard')
def dashboard():
    """Dashboard page route"""
    # Get counts for the dashboard
    protocol_count = Protocol.query.count()
    data_count = BehavioralData.query.count()
    model_count = MLModel.query.count()
    
    # Get latest protocols
    latest_protocols = Protocol.query.order_by(Protocol.created_at.desc()).limit(5).all()
    
    # Get latest behavioral data entries
    latest_data = BehavioralData.query.order_by(BehavioralData.created_at.desc()).limit(5).all()
    
    # Get latest models
    latest_models = MLModel.query.order_by(MLModel.created_at.desc()).limit(5).all()
    
    return render_template('dashboard.html',
                          protocol_count=protocol_count,
                          data_count=data_count,
                          model_count=model_count,
                          latest_protocols=latest_protocols,
                          latest_data=latest_data,
                          latest_models=latest_models)

@app.route('/data_entry', methods=['GET', 'POST'])
def data_entry():
    """Route for entering behavioral data"""
    form = BehavioralDataForm()
    
    # Add available protocols to the form
    protocols = Protocol.query.all()
    protocol_choices = [(0, 'None')] + [(p.id, p.name) for p in protocols]
    form.protocol_used = SelectField('Protocol Used (if any)', choices=protocol_choices, coerce=int, validators=[Optional()])
    
    if form.validate_on_submit():
        try:
            # Create a new behavioral data entry
            behavioral_data = BehavioralData(
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
                outcome=form.outcome.data
            )
            
            # Set protocol if selected
            if form.protocol_used.data > 0:
                behavioral_data.protocol_used = form.protocol_used.data
            
            db.session.add(behavioral_data)
            db.session.commit()
            
            flash('Behavioral data added successfully!', 'success')
            return redirect(url_for('dashboard'))
            
        except Exception as e:
            db.session.rollback()
            logger.error(f"Error adding behavioral data: {str(e)}")
            flash(f'Error adding data: {str(e)}', 'danger')
    
    return render_template('data_entry.html', form=form, title="Enter Behavioral Data")

@app.route('/protocols', methods=['GET'])
def protocols():
    """Route for viewing and managing protocols"""
    protocols = Protocol.query.all()
    form = ProtocolForm()
    
    return render_template('protocols.html', 
                          protocols=protocols, 
                          form=form,
                          title="Behavioral Protocols")

@app.route('/protocols/add', methods=['POST'])
def add_protocol():
    """Route for adding a new protocol"""
    form = ProtocolForm()
    
    if form.validate_on_submit():
        try:
            # Create a new protocol
            protocol = Protocol(
                name=form.name.data,
                description=form.description.data,
                category=form.category.data
            )
            
            db.session.add(protocol)
            db.session.commit()
            
            flash(f'Protocol "{form.name.data}" created successfully!', 'success')
            return redirect(url_for('protocol_detail', protocol_id=protocol.id))
            
        except Exception as e:
            db.session.rollback()
            logger.error(f"Error adding protocol: {str(e)}")
            flash(f'Error adding protocol: {str(e)}', 'danger')
    
    # If validation fails, return to the protocols page with errors
    protocols = Protocol.query.all()
    return render_template('protocols.html', 
                          protocols=protocols, 
                          form=form,
                          title="Behavioral Protocols")

@app.route('/protocols/<int:protocol_id>', methods=['GET'])
def protocol_detail(protocol_id):
    """Route for viewing a protocol's details and decision points"""
    protocol = Protocol.query.get_or_404(protocol_id)
    decision_points = DecisionPoint.query.filter_by(protocol_id=protocol_id).order_by(DecisionPoint.order).all()
    
    # Forms for adding decision points and options
    dp_form = DecisionPointForm()
    
    # If there are decision points, determine the next order number
    if decision_points:
        next_order = max([dp.order for dp in decision_points]) + 1
    else:
        next_order = 1
    
    dp_form.order.data = next_order
    
    return render_template('protocol_detail.html',
                          protocol=protocol,
                          decision_points=decision_points,
                          dp_form=dp_form,
                          title=f"Protocol: {protocol.name}")

@app.route('/protocols/<int:protocol_id>/add_decision_point', methods=['POST'])
def add_decision_point(protocol_id):
    """Route for adding a decision point to a protocol"""
    protocol = Protocol.query.get_or_404(protocol_id)
    form = DecisionPointForm()
    
    if form.validate_on_submit():
        try:
            # Create a new decision point
            decision_point = DecisionPoint(
                protocol_id=protocol_id,
                question=form.question.data,
                order=form.order.data
            )
            
            db.session.add(decision_point)
            db.session.commit()
            
            flash('Decision point added successfully!', 'success')
            
        except Exception as e:
            db.session.rollback()
            logger.error(f"Error adding decision point: {str(e)}")
            flash(f'Error adding decision point: {str(e)}', 'danger')
    
    return redirect(url_for('protocol_detail', protocol_id=protocol_id))

@app.route('/decision_points/<int:dp_id>', methods=['GET'])
def decision_point_detail(dp_id):
    """Route for viewing a decision point's details and options"""
    decision_point = DecisionPoint.query.get_or_404(dp_id)
    protocol = Protocol.query.get_or_404(decision_point.protocol_id)
    options = DecisionOption.query.filter_by(decision_point_id=dp_id).all()
    
    # Form for adding options
    form = DecisionOptionForm()
    
    # Get all other decision points in this protocol for the next_decision_id dropdown
    other_dps = DecisionPoint.query.filter_by(protocol_id=protocol.id).filter(DecisionPoint.id != dp_id).all()
    form.next_decision_id.choices = [(0, 'None')] + [(dp.id, f"{dp.order}. {dp.question[:30]}...") for dp in other_dps]
    
    return render_template('decision_point_detail.html',
                          decision_point=decision_point,
                          protocol=protocol,
                          options=options,
                          form=form,
                          title=f"Decision Point: {decision_point.question[:30]}...")

@app.route('/decision_points/<int:dp_id>/add_option', methods=['POST'])
def add_option(dp_id):
    """Route for adding an option to a decision point"""
    decision_point = DecisionPoint.query.get_or_404(dp_id)
    form = DecisionOptionForm()
    
    # Set up the choices for the form validation
    other_dps = DecisionPoint.query.filter_by(protocol_id=decision_point.protocol_id).filter(DecisionPoint.id != dp_id).all()
    form.next_decision_id.choices = [(0, 'None')] + [(dp.id, f"{dp.order}. {dp.question[:30]}...") for dp in other_dps]
    
    if form.validate_on_submit():
        try:
            # Create a new option
            option = DecisionOption(
                decision_point_id=dp_id,
                text=form.text.data,
                is_terminal=form.is_terminal.data,
                recommendation=form.recommendation.data if form.is_terminal.data else None
            )
            
            # Set next_decision_id if not terminal and a next decision was selected
            if not form.is_terminal.data and form.next_decision_id.data > 0:
                option.next_decision_id = form.next_decision_id.data
            
            db.session.add(option)
            db.session.commit()
            
            flash('Option added successfully!', 'success')
            
        except Exception as e:
            db.session.rollback()
            logger.error(f"Error adding option: {str(e)}")
            flash(f'Error adding option: {str(e)}', 'danger')
    
    return redirect(url_for('decision_point_detail', dp_id=dp_id))

@app.route('/analyze', methods=['GET', 'POST'])
def analyze():
    """Route for analyzing behavioral data and training models"""
    # Load data for display
    data = load_behavioral_data_to_dataframe()
    data_empty = data.empty
    
    # Create the training form
    form = ModelTrainingForm()
    
    # If data is available, add the columns as choices for target column
    if not data_empty:
        # Exclude certain columns that shouldn't be prediction targets
        exclude_cols = ['id', 'subject_id', 'created_at']
        valid_cols = [col for col in data.columns if col not in exclude_cols]
        form.target_column.choices = [(col, col) for col in valid_cols]
    
    # Process the form if submitted
    if form.validate_on_submit() and not data_empty:
        try:
            # Process the data for ML
            processed_data = preprocess_behavioral_data(data)
            
            # Train the model based on the selected type
            if form.model_type.data == 'decision_tree':
                success, model_info = behavioral_model.train_decision_tree(
                    data=processed_data,
                    target_column=form.target_column.data,
                    max_depth=form.max_depth.data
                )
            elif form.model_type.data == 'random_forest':
                success, model_info = behavioral_model.train_random_forest(
                    data=processed_data,
                    target_column=form.target_column.data,
                    n_estimators=form.n_estimators.data
                )
            
            if success:
                flash('Model trained successfully!', 'success')
                return redirect(url_for('analyze'))
            else:
                flash(f'Error training model: {model_info.get("error", "Unknown error")}', 'danger')
                
        except Exception as e:
            logger.error(f"Error in model training: {str(e)}")
            flash(f'Error: {str(e)}', 'danger')
    
    # Get existing models for display
    models = MLModel.query.all()
    
    # Basic data statistics if data is available
    data_stats = {}
    if not data_empty:
        data_stats = {
            'num_entries': len(data),
            'num_features': len(data.columns),
            'sample_columns': ', '.join(data.columns[:5]) + ('...' if len(data.columns) > 5 else '')
        }
    
    return render_template('analyze.html',
                          form=form,
                          data_empty=data_empty,
                          data_stats=data_stats,
                          models=models,
                          title="Analyze Data & Train Models")

@app.route('/decision_support', methods=['GET', 'POST'])
def decision_support():
    """Route for the decision support system"""
    form = DecisionSupportForm()
    
    # Populate protocol choices
    protocols = Protocol.query.all()
    form.protocol_id.choices = [(p.id, p.name) for p in protocols]
    
    if form.validate_on_submit():
        # Start a new decision support session with the selected protocol
        session['current_protocol_id'] = form.protocol_id.data
        session['current_dp_id'] = None  # Will be set to the first decision point
        
        # Get the protocol structure
        protocol_tree = get_protocol_tree(form.protocol_id.data)
        if not protocol_tree:
            flash('Error: Protocol structure could not be loaded', 'danger')
            return redirect(url_for('decision_support'))
        
        # Find the first decision point (lowest order)
        decision_points = DecisionPoint.query.filter_by(protocol_id=form.protocol_id.data).order_by(DecisionPoint.order).all()
        if not decision_points:
            flash('Error: Protocol has no decision points', 'danger')
            return redirect(url_for('decision_support'))
        
        # Set the first decision point as current
        session['current_dp_id'] = decision_points[0].id
        
        return redirect(url_for('decision_process'))
    
    return render_template('decision_support.html',
                          form=form,
                          title="Behavioral Decision Support")

@app.route('/decision_process', methods=['GET', 'POST'])
def decision_process():
    """Route for processing through a decision protocol"""
    # Check if a protocol is selected
    if 'current_protocol_id' not in session:
        flash('Please select a protocol first', 'warning')
        return redirect(url_for('decision_support'))
    
    protocol_id = session['current_protocol_id']
    current_dp_id = session.get('current_dp_id')
    
    # Get protocol info
    protocol = Protocol.query.get_or_404(protocol_id)
    
    # If no current decision point, get the first one
    if not current_dp_id:
        decision_point = DecisionPoint.query.filter_by(protocol_id=protocol_id).order_by(DecisionPoint.order).first()
        if not decision_point:
            flash('Error: Protocol has no decision points', 'danger')
            return redirect(url_for('decision_support'))
        current_dp_id = decision_point.id
        session['current_dp_id'] = current_dp_id
    else:
        decision_point = DecisionPoint.query.get_or_404(current_dp_id)
    
    # Get options for the current decision point
    options = DecisionOption.query.filter_by(decision_point_id=current_dp_id).all()
    
    # Handle option selection
    if request.method == 'POST':
        option_id = request.form.get('option_id')
        if not option_id:
            flash('Please select an option', 'warning')
        else:
            option = DecisionOption.query.get_or_404(option_id)
            
            # If this is a terminal option, show the result
            if option.is_terminal:
                session['recommendation'] = option.recommendation
                return redirect(url_for('decision_result'))
            
            # Otherwise, move to the next decision point
            if option.next_decision_id:
                session['current_dp_id'] = option.next_decision_id
                return redirect(url_for('decision_process'))
            else:
                # If no next decision point is specified but it's not terminal, 
                # that's an error in the protocol design
                flash('Error: Option leads nowhere and is not terminal', 'danger')
    
    return render_template('decision_process.html',
                          protocol=protocol,
                          decision_point=decision_point,
                          options=options,
                          title=f"Decision Support: {protocol.name}")

@app.route('/decision_result')
def decision_result():
    """Route for showing the decision support result"""
    # Check if a recommendation exists in the session
    if 'recommendation' not in session:
        flash('No recommendation available', 'warning')
        return redirect(url_for('decision_support'))
    
    recommendation = session['recommendation']
    protocol_id = session.get('current_protocol_id')
    protocol = Protocol.query.get_or_404(protocol_id) if protocol_id else None
    
    # Clear the session data
    session.pop('current_protocol_id', None)
    session.pop('current_dp_id', None)
    session.pop('recommendation', None)
    
    return render_template('results.html',
                          recommendation=recommendation,
                          protocol=protocol,
                          title="Decision Support Result")

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """Route for making predictions with trained models"""
    form = PredictionForm()
    
    # Populate model choices
    models = MLModel.query.all()
    form.model_id.choices = [(m.id, m.name) for m in models]
    
    if form.validate_on_submit():
        try:
            # Load the selected model
            model_id = form.model_id.data
            success = behavioral_model.load_model(model_id)
            
            if not success:
                flash('Error loading model', 'danger')
                return redirect(url_for('predict'))
            
            # Collect the input data from the form
            input_data = {}
            for field_name, field_value in request.form.items():
                # Skip the CSRF token and submit field
                if field_name not in ['csrf_token', 'submit', 'model_id']:
                    try:
                        # Try to convert to float for numeric fields
                        input_data[field_name] = float(field_value)
                    except ValueError:
                        # Keep as string if not numeric
                        input_data[field_name] = field_value
            
            # Make prediction
            prediction, confidence, explanation = behavioral_model.predict(input_data)
            
            if prediction is not None:
                # Store result in session
                session['prediction_result'] = {
                    'prediction': str(prediction),
                    'confidence': float(confidence),
                    'explanation': explanation
                }
                return redirect(url_for('prediction_result'))
            else:
                flash(f'Error making prediction: {explanation}', 'danger')
                
        except Exception as e:
            logger.error(f"Error in prediction: {str(e)}")
            flash(f'Error: {str(e)}', 'danger')
    
    return render_template('predict.html',
                          form=form,
                          title="Make Predictions")

@app.route('/prediction_result')
def prediction_result():
    """Route for showing prediction results"""
    # Check if a prediction result exists in the session
    if 'prediction_result' not in session:
        flash('No prediction available', 'warning')
        return redirect(url_for('predict'))
    
    result = session['prediction_result']
    
    # Clear the session data
    session.pop('prediction_result', None)
    
    return render_template('results.html',
                          prediction=result['prediction'],
                          confidence=result['confidence'],
                          explanation=result['explanation'],
                          title="Prediction Result")

@app.route('/api/model/<int:model_id>/fields')
def get_model_fields(model_id):
    """API endpoint to get the required fields for a model"""
    try:
        model_record = MLModel.query.get_or_404(model_id)
        features = json.loads(model_record.features)
        
        return jsonify({
            'success': True,
            'fields': features
        })
    except Exception as e:
        logger.error(f"Error getting model fields: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        })

# Teacher Input Routes

@app.route('/teacher-input', methods=['GET'])
def teacher_input():
    """Route for teacher input page with both text and voice options"""
    return render_template('teacher_input.html',
                          title="Behavioral Support for Teachers")

@app.route('/process-teacher-input', methods=['POST'])
def process_teacher_input():
    """Process teacher input from either text or voice form"""
    try:
        input_type = request.form.get('input_type')
        severity = request.form.get('severity')
        
        if input_type == 'text':
            # Process text input
            student_description = request.form.get('student_description', '')
            behavior_description = request.form.get('behavior_description', '')
            
            if not behavior_description:
                return jsonify({
                    "success": False,
                    "error": "Behavior description is required"
                })
                
            # Use the behavior description for analysis
            text_for_analysis = behavior_description
            
        elif input_type == 'voice':
            # Process voice input
            voice_transcript = request.form.get('voice_transcript', '')
            
            if not voice_transcript:
                return jsonify({
                    "success": False,
                    "error": "Voice transcript is empty. Please try recording again."
                })
                
            # Use the voice transcript for analysis
            text_for_analysis = voice_transcript
            
        else:
            return jsonify({
                "success": False,
                "error": "Invalid input type"
            })
            
        # Get protocol - use the first one for simplicity
        protocol = Protocol.query.first()
        if not protocol:
            return jsonify({
                "success": False,
                "error": "No protocols found in the system"
            })
            
        # Extract keywords and analyze the text
        keywords = extract_keywords_from_speech(text_for_analysis)
        
        # Adjust analysis based on severity
        is_emergency = False
        if severity == 'high':
            is_emergency = True
            
        # Get recommendation based on the text and severity
        # Use our existing analysis function
        analysis_result = analyze_speech_for_decision(text_for_analysis, protocol.id)
        
        if analysis_result['success']:
            recommendation = analysis_result['recommendation'] or "Follow standard classroom management protocols."
            
            return jsonify({
                "success": True,
                "recommendation_title": "Recommended Action",
                "recommendation": recommendation,
                "decision_point": analysis_result['decision_point'].question if 'decision_point' in analysis_result else None,
                "is_emergency": analysis_result.get('is_emergency', is_emergency),
                "keywords": keywords
            })
        else:
            # Provide a general recommendation based on severity if analysis fails
            general_recommendations = {
                'low': "Monitor the situation. Use calm redirection and positive reinforcement strategies. Document the behavior if it persists.",
                'medium': "Implement targeted behavior management strategies. Consider contacting parents/guardians if behavior continues. Document the incident.",
                'high': "Immediate intervention required. Ensure student and classroom safety first. Get additional support if needed. Document the incident thoroughly."
            }
            
            return jsonify({
                "success": True,
                "recommendation_title": "General Recommendation",
                "recommendation": general_recommendations.get(severity, general_recommendations['medium']),
                "is_emergency": severity == 'high',
                "keywords": keywords
            })
            
    except Exception as e:
        logger.error(f"Error processing teacher input: {str(e)}")
        return jsonify({
            "success": False,
            "error": f"An error occurred: {str(e)}"
        })

# Teacher rapid input interface
@app.route('/rapid_input', methods=['GET', 'POST'])
def rapid_input():
    """Route for teachers to quickly input behavioral data and get recommendations"""
    form = ProtocolForm()  # Reusing just for protocol selection, not creating
    
    # Get all protocols for selection
    protocols = Protocol.query.all()
    
    if request.method == 'POST':
        protocol_id = request.form.get('protocol_id')
        behavior_text = request.form.get('behavior_text')
        severity = request.form.get('severity')
        
        logger.info(f"Teacher input - Protocol: {protocol_id}, Behavior: {behavior_text}, Severity: {severity}")
        
        if not protocol_id or not behavior_text:
            flash('Please select a protocol and provide behavior description', 'warning')
            return render_template('teacher_input.html', 
                                  protocols=protocols,
                                  form=form,
                                  title="Rapid Input")
        
        # Process the input through our decision model
        analysis_result = analyze_speech_for_decision(behavior_text, int(protocol_id))
        
        if analysis_result['success']:
            # Store the recommendation in session
            recommendation = None
            next_step = None
            
            if analysis_result['is_terminal']:
                recommendation = analysis_result['recommendation']
                # Save the recommendation for display
                session['recommendation'] = recommendation
            else:
                # If not terminal, get the next decision point details
                next_dp_id = analysis_result['next_decision_id']
                next_dp = DecisionPoint.query.get(next_dp_id)
                if next_dp:
                    next_step = next_dp.question
                    session['current_dp_id'] = next_dp_id
            
            # Store input for display on result page
            session['behavior_text'] = behavior_text
            session['severity'] = severity
            session['keywords'] = analysis_result.get('keywords', [])
            session['is_emergency'] = analysis_result.get('is_emergency', False)
            session['current_protocol_id'] = int(protocol_id)
            
            # If there's a recommendation, go to results
            if recommendation:
                return redirect(url_for('rapid_result'))
            # If there's a next step, continue the decision tree
            elif next_step:
                return redirect(url_for('decision_process'))
            else:
                flash('Unable to determine next step in protocol', 'danger')
        else:
            # If analysis failed, show the error
            flash(f"Analysis error: {analysis_result.get('error', 'Unknown error')}", 'danger')
    
    return render_template('teacher_input.html', 
                          protocols=protocols,
                          form=form,
                          title="Rapid Input")

@app.route('/rapid_result')
def rapid_result():
    """Route for showing rapid recommendation results"""
    # Check if a recommendation exists in the session
    if 'recommendation' not in session:
        flash('No recommendation available', 'warning')
        return redirect(url_for('rapid_input'))
    
    # Get all stored session data
    recommendation = session.get('recommendation')
    behavior_text = session.get('behavior_text')
    severity = session.get('severity')
    keywords = session.get('keywords', [])
    is_emergency = session.get('is_emergency', False)
    protocol_id = session.get('current_protocol_id')
    
    protocol = Protocol.query.get_or_404(protocol_id) if protocol_id else None
    
    # Clear the session data
    session.pop('recommendation', None)
    session.pop('behavior_text', None)
    session.pop('severity', None)
    session.pop('keywords', None)
    session.pop('is_emergency', None)
    session.pop('current_protocol_id', None)
    session.pop('current_dp_id', None)
    
    return render_template('teacher_result.html',
                          recommendation=recommendation,
                          behavior_text=behavior_text,
                          severity=severity,
                          keywords=keywords,
                          is_emergency=is_emergency,
                          protocol=protocol,
                          title="Recommendation")

@app.route('/api/voice_input', methods=['POST'])
def voice_input():
    """API endpoint for processing voice input from teachers"""
    try:
        # Check if the request contains JSON data
        data = request.get_json() if request.is_json else {}
        speech_text = data.get('speech_text', '')
        protocol_id = data.get('protocol_id')
        
        if not speech_text or not protocol_id:
            return jsonify({
                "success": False,
                "error": "Missing speech text or protocol ID"
            })
        
        # Process through our decision model
        analysis_result = analyze_speech_for_decision(speech_text, int(protocol_id))
        
        # Add the original speech text to the result
        analysis_result['speech_text'] = speech_text
        
        return jsonify(analysis_result)
        
    except Exception as e:
        logger.error(f"Error processing voice input: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        })

# Voice recognition routes

@app.route('/voice_decision_support', methods=['GET', 'POST'])
def voice_decision_support():
    """Route for voice-based decision support"""
    form = DecisionSupportForm()
    
    # Populate protocol choices
    protocols = Protocol.query.all()
    form.protocol_id.choices = [(p.id, p.name) for p in protocols]
    
    if form.validate_on_submit():
        # Store the selected protocol ID in session
        session['current_protocol_id'] = form.protocol_id.data
        return redirect(url_for('voice_input_process'))
    
    return render_template('voice_support.html',
                          form=form,
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
                # Analyze the speech for decision mapping
                analysis_result = analyze_speech_for_decision(simulated_speech, protocol_id)
                
                if analysis_result['success']:
                    # If it's a terminal option, set the recommendation
                    if analysis_result['is_terminal']:
                        session['recommendation'] = analysis_result['recommendation']
                        return redirect(url_for('decision_result'))
                    else:
                        # If not terminal, proceed to the next decision point
                        session['current_dp_id'] = analysis_result['next_decision_id']
                        flash(f"Voice analyzed: '{simulated_speech}'. Proceeding with option: {analysis_result['selected_option'].text}", 'success')
                        return redirect(url_for('decision_process'))
                else:
                    # If analysis failed, show the error
                    flash(f"Could not process voice input: {analysis_result.get('error', 'Unknown error')}", 'danger')
                    
                    # If keywords were extracted but no mapping found, show them
                    if 'keywords' in analysis_result and analysis_result['keywords']:
                        flash(f"Detected keywords: {', '.join(analysis_result['keywords'])}", 'info')
    
    return render_template('voice_input.html',
                          protocol=protocol,
                          title="Voice Input Processing")

@app.route('/api/voice_capture', methods=['POST'])
def voice_capture():
    """API endpoint for capturing voice input"""
    try:
        # In real-world scenario, we would call the voice recognizer here
        # result = voice_recognizer.listen_once()
        
        # For demo purposes, simulate a successful voice recognition
        # Check if the request contains JSON data
        if request.is_json:
            simulated_text = request.json.get('simulated_text', 'The student is becoming agitated and disruptive in class')
        else:
            simulated_text = 'The student is becoming agitated and disruptive in class'
            
        result = {
            "success": True,
            "text": simulated_text
        }
        
        if result["success"]:
            protocol_id = session.get('current_protocol_id')
            if protocol_id:
                # Analyze the speech for decision support
                analysis = analyze_speech_for_decision(result["text"], protocol_id)
                return jsonify({
                    "success": True,
                    "text": result["text"],
                    "analysis": analysis
                })
            else:
                return jsonify({
                    "success": True,
                    "text": result["text"],
                    "analysis": {
                        "success": False,
                        "error": "No protocol selected"
                    }
                })
        else:
            return jsonify(result)
    
    except Exception as e:
        logger.error(f"Error in voice capture: {str(e)}")
        return jsonify({"success": False, "error": str(e)})
