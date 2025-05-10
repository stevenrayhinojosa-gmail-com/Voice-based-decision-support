from flask_wtf import FlaskForm
from wtforms import StringField, TextAreaField, IntegerField, SelectField, BooleanField, RadioField, SubmitField
from wtforms.validators import DataRequired, Optional, NumberRange, Length

class BehavioralDataForm(FlaskForm):
    """Form for entering behavioral data"""
    subject_id = StringField('Subject ID', validators=[DataRequired(), Length(min=2, max=50)])
    age = IntegerField('Age', validators=[Optional(), NumberRange(min=0, max=120)])
    gender = SelectField('Gender', choices=[('', 'Select...'), ('male', 'Male'), ('female', 'Female'), ('other', 'Other'), ('prefer_not_to_say', 'Prefer not to say')])
    context = StringField('Context', validators=[Optional(), Length(max=100)])
    behavior_description = TextAreaField('Behavior Description', validators=[DataRequired(), Length(min=10, max=1000)])
    intensity = IntegerField('Intensity (1-10)', validators=[Optional(), NumberRange(min=1, max=10)])
    frequency = IntegerField('Frequency (times per day/week)', validators=[Optional(), NumberRange(min=0)])
    duration = IntegerField('Duration (minutes)', validators=[Optional(), NumberRange(min=0)])
    triggers = TextAreaField('Triggers', validators=[Optional(), Length(max=500)])
    consequences = TextAreaField('Consequences', validators=[Optional(), Length(max=500)])
    outcome = TextAreaField('Outcome', validators=[Optional(), Length(max=500)])
    submit = SubmitField('Submit')

class ProtocolForm(FlaskForm):
    """Form for creating a new behavioral protocol"""
    name = StringField('Protocol Name', validators=[DataRequired(), Length(min=3, max=100)])
    description = TextAreaField('Description', validators=[Optional(), Length(max=1000)])
    category = StringField('Category', validators=[Optional(), Length(max=50)])
    submit = SubmitField('Create Protocol')

class DecisionPointForm(FlaskForm):
    """Form for adding a decision point to a protocol"""
    question = TextAreaField('Question/Decision Point', validators=[DataRequired(), Length(min=5, max=500)])
    order = IntegerField('Order', validators=[DataRequired(), NumberRange(min=1)])
    submit = SubmitField('Add Decision Point')

class DecisionOptionForm(FlaskForm):
    """Form for adding options to a decision point"""
    text = TextAreaField('Option Text', validators=[DataRequired(), Length(min=1, max=500)])
    is_terminal = BooleanField('Is this a terminal option?')
    recommendation = TextAreaField('Recommendation (if terminal)', validators=[Optional(), Length(max=1000)])
    next_decision_id = SelectField('Next Decision Point (if not terminal)', coerce=int, validators=[Optional()])
    submit = SubmitField('Add Option')

class ModelTrainingForm(FlaskForm):
    """Form for training a machine learning model"""
    model_type = SelectField('Model Type', choices=[
        ('decision_tree', 'Decision Tree'),
        ('random_forest', 'Random Forest')
    ])
    target_column = SelectField('Target Column (what to predict)', validators=[DataRequired()])
    max_depth = IntegerField('Max Depth (for Decision Tree)', validators=[Optional(), NumberRange(min=1, max=20)], default=5)
    n_estimators = IntegerField('Number of Trees (for Random Forest)', validators=[Optional(), NumberRange(min=10, max=1000)], default=100)
    submit = SubmitField('Train Model')

class DecisionSupportForm(FlaskForm):
    """Form for using the decision support model"""
    protocol_id = SelectField('Select Protocol', coerce=int, validators=[DataRequired()])
    submit = SubmitField('Start Decision Support')

class PredictionForm(FlaskForm):
    """Dynamic form for making predictions using a trained model"""
    model_id = SelectField('Select Model', coerce=int, validators=[DataRequired()])
    # The rest of the fields will be added dynamically based on the selected model
    submit = SubmitField('Make Prediction')
