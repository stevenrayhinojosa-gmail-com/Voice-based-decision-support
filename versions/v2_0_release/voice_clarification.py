"""
Voice Clarification System for SereniTeach
Handles ambiguous voice inputs by prompting targeted follow-up questions
"""

import logging
from localization import localization_manager, get_localized_phrase

logger = logging.getLogger(__name__)

# Confidence threshold for triggering clarification
NLP_CONFIDENCE_THRESHOLD = 0.6

class VoiceClarificationSystem:
    """Manages voice clarification prompts when NLP confidence is low"""
    
    def __init__(self):
        self.clarification_prompts = {
            # Behavior type clarification
            'behavior_type': {
                'aggression_physical': {
                    'en': "Did the student make physical contact with others?",
                    'es': "¿El estudiante hizo contacto físico con otros?"
                },
                'disruption_movement': {
                    'en': "Is the student attempting to leave the classroom?",
                    'es': "¿El estudiante está tratando de salir del aula?"
                },
                'defiance_compliance': {
                    'en': "Is the student refusing to follow direct instructions?",
                    'es': "¿El estudiante se niega a seguir instrucciones directas?"
                },
                'self_regulation': {
                    'en': "Is the student able to respond to their name when called?",
                    'es': "¿El estudiante puede responder a su nombre cuando se le llama?"
                },
                'safety_concern': {
                    'en': "Is there immediate risk of injury to the student or others?",
                    'es': "¿Hay riesgo inmediato de lesión para el estudiante u otros?"
                }
            },
            
            # Severity level clarification
            'severity': {
                'escalation_check': {
                    'en': "Has the behavior escalated in the last 5 minutes?",
                    'es': "¿El comportamiento ha escalado en los últimos 5 minutos?"
                },
                'intervention_response': {
                    'en': "Is the student responding to verbal interventions?",
                    'es': "¿El estudiante responde a las intervenciones verbales?"
                },
                'disruption_level': {
                    'en': "Is the behavior affecting other students' learning?",
                    'es': "¿El comportamiento está afectando el aprendizaje de otros estudiantes?"
                },
                'adult_attention': {
                    'en': "Does the situation require additional adult support?",
                    'es': "¿La situación requiere apoyo adicional de adultos?"
                }
            },
            
            # Emergency assessment
            'emergency': {
                'immediate_danger': {
                    'en': "Is anyone in immediate physical danger?",
                    'es': "¿Alguien está en peligro físico inmediato?"
                },
                'medical_concern': {
                    'en': "Does the student appear to have a medical emergency?",
                    'es': "¿El estudiante parece tener una emergencia médica?"
                },
                'property_damage': {
                    'en': "Is there significant damage to classroom property?",
                    'es': "¿Hay daños significativos a la propiedad del aula?"
                }
            }
        }
        
        self.clarification_mappings = {
            # Behavior type mappings based on responses
            'behavior_type': {
                'aggression_physical': {
                    'yes': {'behavior_type': 'aggression', 'severity': 'high'},
                    'no': {'behavior_type': 'disruption', 'severity': 'medium'}
                },
                'disruption_movement': {
                    'yes': {'behavior_type': 'off_task', 'severity': 'medium'},
                    'no': {'behavior_type': 'disruption', 'severity': 'low'}
                },
                'defiance_compliance': {
                    'yes': {'behavior_type': 'defiance', 'severity': 'high'},
                    'no': {'behavior_type': 'off_task', 'severity': 'medium'}
                },
                'self_regulation': {
                    'yes': {'behavior_type': 'disruption', 'severity': 'low'},
                    'no': {'behavior_type': 'off_task', 'severity': 'high'}
                },
                'safety_concern': {
                    'yes': {'behavior_type': 'aggression', 'severity': 'high', 'is_emergency': True},
                    'no': {'behavior_type': 'disruption', 'severity': 'medium'}
                }
            },
            
            # Severity mappings
            'severity': {
                'escalation_check': {
                    'yes': {'severity': 'high'},
                    'no': {'severity': 'medium'}
                },
                'intervention_response': {
                    'yes': {'severity': 'low'},
                    'no': {'severity': 'high'}
                },
                'disruption_level': {
                    'yes': {'severity': 'medium'},
                    'no': {'severity': 'low'}
                },
                'adult_attention': {
                    'yes': {'severity': 'high'},
                    'no': {'severity': 'medium'}
                }
            },
            
            # Emergency mappings
            'emergency': {
                'immediate_danger': {
                    'yes': {'is_emergency': True, 'severity': 'high'},
                    'no': {'is_emergency': False}
                },
                'medical_concern': {
                    'yes': {'is_emergency': True, 'severity': 'high', 'behavior_type': 'medical'},
                    'no': {'is_emergency': False}
                },
                'property_damage': {
                    'yes': {'severity': 'high', 'behavior_type': 'aggression'},
                    'no': {'severity': 'medium'}
                }
            }
        }
        
        logger.info("Voice clarification system initialized")
    
    def needs_clarification(self, analysis_result):
        """
        Determine if clarification is needed based on NLP confidence scores
        
        Parameters:
        - analysis_result: Dictionary containing NLP analysis with confidence scores
        
        Returns:
        - Dictionary with clarification needs or None if no clarification needed
        """
        try:
            # Check if confidence scores exist in the analysis
            if not analysis_result or 'confidence' not in analysis_result:
                return None
            
            confidence_data = analysis_result['confidence']
            clarification_needed = None
            
            # Check behavior type confidence
            behavior_confidence = confidence_data.get('behavior_type', 1.0)
            if behavior_confidence < NLP_CONFIDENCE_THRESHOLD:
                clarification_needed = {
                    'type': 'behavior_type',
                    'confidence': behavior_confidence,
                    'reason': 'Low confidence in behavior type classification'
                }
            
            # Check severity confidence (higher priority than behavior type)
            severity_confidence = confidence_data.get('severity', 1.0)
            if severity_confidence < NLP_CONFIDENCE_THRESHOLD:
                clarification_needed = {
                    'type': 'severity',
                    'confidence': severity_confidence,
                    'reason': 'Low confidence in severity assessment'
                }
            
            # Check emergency detection confidence (highest priority)
            emergency_confidence = confidence_data.get('emergency', 1.0)
            if emergency_confidence < NLP_CONFIDENCE_THRESHOLD:
                clarification_needed = {
                    'type': 'emergency',
                    'confidence': emergency_confidence,
                    'reason': 'Low confidence in emergency assessment'
                }
            
            return clarification_needed
            
        except Exception as e:
            logger.error(f"Error checking clarification needs: {e}")
            return None
    
    def get_clarification_prompt(self, clarification_type, context=None):
        """
        Get appropriate clarification prompt based on type and context
        
        Parameters:
        - clarification_type: Type of clarification needed
        - context: Additional context for selecting specific prompt
        
        Returns:
        - Dictionary with prompt information
        """
        try:
            current_language = localization_manager.current_language
            prompts = self.clarification_prompts.get(clarification_type, {})
            
            # Select specific prompt based on context or use default
            if clarification_type == 'behavior_type':
                # Choose prompt based on behavior keywords or context
                if context and 'keywords' in context:
                    keywords = context['keywords']
                    if any(kw in ['hit', 'push', 'throw', 'kick'] for kw in keywords):
                        prompt_key = 'aggression_physical'
                    elif any(kw in ['leave', 'run', 'escape', 'door'] for kw in keywords):
                        prompt_key = 'disruption_movement'
                    elif any(kw in ['refuse', 'no', 'won\'t', 'defiant'] for kw in keywords):
                        prompt_key = 'defiance_compliance'
                    elif any(kw in ['danger', 'hurt', 'unsafe', 'injury'] for kw in keywords):
                        prompt_key = 'safety_concern'
                    else:
                        prompt_key = 'self_regulation'
                else:
                    prompt_key = 'self_regulation'
            
            elif clarification_type == 'severity':
                # Choose severity prompt based on current assessment
                if context and context.get('severity') == 'high':
                    prompt_key = 'adult_attention'
                elif context and 'escalating' in str(context.get('description', '')).lower():
                    prompt_key = 'escalation_check'
                else:
                    prompt_key = 'intervention_response'
            
            elif clarification_type == 'emergency':
                # Choose emergency prompt based on keywords
                if context and 'keywords' in context:
                    keywords = context['keywords']
                    if any(kw in ['medical', 'sick', 'hurt', 'injury'] for kw in keywords):
                        prompt_key = 'medical_concern'
                    elif any(kw in ['damage', 'broken', 'destroy'] for kw in keywords):
                        prompt_key = 'property_damage'
                    else:
                        prompt_key = 'immediate_danger'
                else:
                    prompt_key = 'immediate_danger'
            
            else:
                logger.warning(f"Unknown clarification type: {clarification_type}")
                return None
            
            # Get localized prompt
            prompt_text = prompts.get(prompt_key, {}).get(current_language, 
                                                         prompts.get(prompt_key, {}).get('en', ''))
            
            if not prompt_text:
                logger.warning(f"No prompt found for {clarification_type}:{prompt_key}")
                return None
            
            return {
                'prompt_type': clarification_type,
                'prompt_key': prompt_key,
                'prompt_text': prompt_text,
                'language': current_language,
                'expected_responses': ['yes', 'no', 'sí', 'no']
            }
            
        except Exception as e:
            logger.error(f"Error getting clarification prompt: {e}")
            return None
    
    def process_clarification_response(self, clarification_context, response_text):
        """
        Process the teacher's response to a clarification prompt
        
        Parameters:
        - clarification_context: Original clarification prompt context
        - response_text: Teacher's voice response
        
        Returns:
        - Dictionary with updated analysis based on response
        """
        try:
            # Normalize response
            response_normalized = response_text.lower().strip()
            
            # Map response to yes/no
            yes_responses = ['yes', 'yeah', 'yep', 'correct', 'true', 'sí', 'si', 'claro', 'correcto']
            no_responses = ['no', 'nope', 'false', 'incorrect', 'wrong', 'no', 'negativo']
            
            if any(resp in response_normalized for resp in yes_responses):
                response_key = 'yes'
            elif any(resp in response_normalized for resp in no_responses):
                response_key = 'no'
            else:
                logger.warning(f"Unclear clarification response: {response_text}")
                return {
                    'success': False,
                    'error': 'Unclear response to clarification question',
                    'suggested_prompt': get_localized_phrase('error_processing')
                }
            
            # Get the mapping for this clarification
            prompt_type = clarification_context['prompt_type']
            prompt_key = clarification_context['prompt_key']
            
            mapping = self.clarification_mappings.get(prompt_type, {}).get(prompt_key, {})
            updates = mapping.get(response_key, {})
            
            if not updates:
                logger.warning(f"No mapping found for {prompt_type}:{prompt_key}:{response_key}")
                return {
                    'success': False,
                    'error': 'Unable to process clarification response'
                }
            
            # Log the clarification interaction
            logger.info(f"Clarification processed: {prompt_type}:{prompt_key} -> {response_key}")
            
            return {
                'success': True,
                'updates': updates,
                'clarification_log': {
                    'prompt_type': prompt_type,
                    'prompt_key': prompt_key,
                    'prompt_text': clarification_context['prompt_text'],
                    'response_text': response_text,
                    'response_interpretation': response_key,
                    'updates_applied': updates
                }
            }
            
        except Exception as e:
            logger.error(f"Error processing clarification response: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_clarification_status_message(self, clarification_type):
        """Get localized status message for clarification request"""
        messages = {
            'behavior_type': {
                'en': "I need to clarify the type of behavior. Please answer yes or no:",
                'es': "Necesito aclarar el tipo de comportamiento. Por favor responde sí o no:"
            },
            'severity': {
                'en': "I need to assess the severity level. Please answer yes or no:",
                'es': "Necesito evaluar el nivel de severidad. Por favor responde sí o no:"
            },
            'emergency': {
                'en': "I need to determine if this is an emergency. Please answer yes or no:",
                'es': "Necesito determinar si esto es una emergencia. Por favor responde sí o no:"
            }
        }
        
        current_language = localization_manager.current_language
        return messages.get(clarification_type, {}).get(current_language, 
                                                       messages.get(clarification_type, {}).get('en', ''))

# Create global instance
voice_clarification_system = VoiceClarificationSystem()