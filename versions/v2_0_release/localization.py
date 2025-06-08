"""
Multi-language localization system for SereniTeach
Supports English (en) and Spanish (es) for behavioral crisis responses
"""

import json
import logging

logger = logging.getLogger(__name__)

# Global language state
current_language = "en"

# Comprehensive phrase dictionary for behavioral crisis support
PHRASES = {
    # System responses
    "listening": {
        "en": "I'm listening. Please describe what's happening.",
        "es": "Te escucho. Por favor describe lo que está pasando."
    },
    "analyzing": {
        "en": "Analyzing the situation...",
        "es": "Analizando la situación..."
    },
    "recommendation_ready": {
        "en": "Here's my recommendation:",
        "es": "Aquí está mi recomendación:"
    },
    
    # Crisis management phrases
    "stay_calm": {
        "en": "Stay calm. Take a deep breath. You're handling this well.",
        "es": "Mantén la calma. Respira profundo. Lo estás manejando bien."
    },
    "safety_first": {
        "en": "Safety is the priority. Ensure everyone is safe.",
        "es": "La seguridad es la prioridad. Asegúrate de que todos estén seguros."
    },
    "de_escalate": {
        "en": "Use a calm, low voice. Give the student space to calm down.",
        "es": "Usa una voz calmada y baja. Dale al estudiante espacio para calmarse."
    },
    "redirect_attention": {
        "en": "Try redirecting their attention to a preferred activity.",
        "es": "Intenta redirigir su atención a una actividad preferida."
    },
    "crisis_detected": {
        "en": "Crisis situation detected. Front office has been notified.",
        "es": "Situación de crisis detectada. La oficina principal ha sido notificada."
    },
    "crisis_over": {
        "en": "Crisis has ended. Preparing incident report...",
        "es": "La crisis ha terminado. Preparando el informe del incidente..."
    },
    
    # Student identification
    "student_identified": {
        "en": "Student identified: {name}. {bip_status}",
        "es": "Estudiante identificado: {name}. {bip_status}"
    },
    "has_bip": {
        "en": "This student has a Behavior Intervention Plan.",
        "es": "Este estudiante tiene un Plan de Intervención de Comportamiento."
    },
    "no_bip": {
        "en": "No BIP on file for this student.",
        "es": "No hay PIC archivado para este estudiante."
    },
    
    # Behavior types and severity
    "behavior_types": {
        "disruption": {
            "en": "disruptive behavior",
            "es": "comportamiento disruptivo"
        },
        "defiance": {
            "en": "defiant behavior",
            "es": "comportamiento desafiante"
        },
        "aggression": {
            "en": "aggressive behavior",
            "es": "comportamiento agresivo"
        },
        "off_task": {
            "en": "off-task behavior",
            "es": "comportamiento fuera de tarea"
        },
        "social": {
            "en": "social behavior issue",
            "es": "problema de comportamiento social"
        }
    },
    
    "severity_levels": {
        "low": {
            "en": "low severity",
            "es": "severidad baja"
        },
        "medium": {
            "en": "medium severity",
            "es": "severidad media"
        },
        "high": {
            "en": "high severity",
            "es": "severidad alta"
        }
    },
    
    # Protocol recommendations
    "protocol_sama": {
        "en": "Using SAMA protocol: {description}",
        "es": "Usando protocolo SAMA: {description}"
    },
    "protocol_pfisd": {
        "en": "Using PfISD protocol: {description}",
        "es": "Usando protocolo PfISD: {description}"
    },
    
    # BIP-specific responses
    "bip_strategy": {
        "en": "BIP Strategy: {strategy}",
        "es": "Estrategia PIC: {strategy}"
    },
    "bip_antecedent": {
        "en": "Known trigger: {antecedent}",
        "es": "Desencadenante conocido: {antecedent}"
    },
    "bip_replacement": {
        "en": "Replacement behavior: {replacement}",
        "es": "Comportamiento de reemplazo: {replacement}"
    },
    
    # Environmental context
    "noise_level_high": {
        "en": "High noise level detected. Consider reducing environmental stimulation.",
        "es": "Nivel de ruido alto detectado. Considera reducir la estimulación ambiental."
    },
    "transition_period": {
        "en": "This is a transition period. Provide extra structure and support.",
        "es": "Este es un período de transición. Proporciona estructura y apoyo adicional."
    },
    
    # Language switching
    "language_switched": {
        "en": "Language switched to English.",
        "es": "Idioma cambiado a español."
    },
    "language_not_supported": {
        "en": "Language not supported. Available languages: English, Spanish.",
        "es": "Idioma no compatible. Idiomas disponibles: inglés, español."
    },
    
    # Feedback system
    "feedback_request": {
        "en": "Was this recommendation helpful? Say 'thumbs up' or 'thumbs down'.",
        "es": "¿Fue útil esta recomendación? Di 'pulgar arriba' o 'pulgar abajo'."
    },
    "feedback_received": {
        "en": "Thank you for your feedback.",
        "es": "Gracias por tu retroalimentación."
    },
    
    # Encouragement messages
    "encouragement": {
        "en": [
            "You're doing an amazing job managing this situation.",
            "Your calm presence is making a difference.",
            "Every student benefits from your dedication.",
            "You're creating a safe space for learning.",
            "Your patience and skill are truly valuable."
        ],
        "es": [
            "Estás haciendo un trabajo increíble manejando esta situación.",
            "Tu presencia calmada está marcando la diferencia.",
            "Cada estudiante se beneficia de tu dedicación.",
            "Estás creando un espacio seguro para el aprendizaje.",
            "Tu paciencia y habilidad son realmente valiosas."
        ]
    },
    
    # Error messages
    "error_processing": {
        "en": "I'm having trouble processing that. Please try again.",
        "es": "Tengo problemas para procesar eso. Por favor intenta de nuevo."
    },
    "no_recommendation": {
        "en": "I couldn't find a specific recommendation. Please provide more details.",
        "es": "No pude encontrar una recomendación específica. Por favor proporciona más detalles."
    },
    
    # Report generation
    "generating_report": {
        "en": "Generating incident report with behavior analytics...",
        "es": "Generando informe de incidente con análisis de comportamiento..."
    },
    "report_sent": {
        "en": "Incident report has been sent to administration with trend visualizations.",
        "es": "El informe de incidente ha sido enviado a la administración con visualizaciones de tendencias."
    },
    
    # Voice commands
    "voice_commands": {
        "en": {
            "switch_spanish": ["switch to spanish", "cambiar a español", "español"],
            "switch_english": ["switch to english", "cambiar a inglés", "english"],
            "crisis_over": ["crisis is over", "crisis over", "situation resolved"],
            "need_help": ["need help", "emergency", "crisis"],
            "thumbs_up": ["thumbs up", "good", "helpful", "yes"],
            "thumbs_down": ["thumbs down", "not helpful", "bad", "no"]
        },
        "es": {
            "switch_spanish": ["cambiar a español", "español", "switch to spanish"],
            "switch_english": ["cambiar a inglés", "english", "switch to english"],
            "crisis_over": ["la crisis terminó", "crisis terminada", "situación resuelta"],
            "need_help": ["necesito ayuda", "emergencia", "crisis"],
            "thumbs_up": ["pulgar arriba", "bueno", "útil", "sí"],
            "thumbs_down": ["pulgar abajo", "no útil", "malo", "no"]
        }
    }
}

class LocalizationManager:
    """Manages multi-language support for the behavioral crisis system"""
    
    def __init__(self):
        self.current_language = "en"
        self.supported_languages = ["en", "es"]
        logger.info(f"Localization manager initialized with default language: {self.current_language}")
    
    def set_language(self, language_code):
        """Set the active language"""
        if language_code in self.supported_languages:
            self.current_language = language_code
            logger.info(f"Language set to: {language_code}")
            return True
        else:
            logger.warning(f"Unsupported language: {language_code}")
            return False
    
    def get_phrase(self, key, **kwargs):
        """Get a localized phrase with optional formatting"""
        try:
            # Navigate nested keys (e.g., "behavior_types.disruption")
            if "." in key:
                keys = key.split(".")
                phrase_dict = PHRASES
                for k in keys:
                    phrase_dict = phrase_dict[k]
                phrase = phrase_dict.get(self.current_language, phrase_dict.get("en", key))
            else:
                phrase = PHRASES.get(key, {}).get(self.current_language, 
                                                 PHRASES.get(key, {}).get("en", key))
            
            # Format with provided kwargs
            if kwargs:
                phrase = phrase.format(**kwargs)
            
            return phrase
            
        except (KeyError, AttributeError, TypeError) as e:
            logger.warning(f"Error getting phrase '{key}': {e}")
            return key  # Return the key as fallback
    
    def detect_language_switch(self, text):
        """Detect language switch commands in voice input"""
        text_lower = text.lower().strip()
        
        # Check for Spanish switch commands
        spanish_triggers = ["switch to spanish", "cambiar a español", "español", "hablar español"]
        if any(trigger in text_lower for trigger in spanish_triggers):
            return "es"
        
        # Check for English switch commands
        english_triggers = ["switch to english", "cambiar a inglés", "english", "speak english"]
        if any(trigger in text_lower for trigger in english_triggers):
            return "en"
        
        return None
    
    def get_voice_commands(self):
        """Get voice commands for current language"""
        return PHRASES["voice_commands"].get(self.current_language, PHRASES["voice_commands"]["en"])
    
    def localize_behavior_type(self, behavior_type):
        """Get localized behavior type name"""
        return self.get_phrase(f"behavior_types.{behavior_type}")
    
    def localize_severity(self, severity):
        """Get localized severity level"""
        return self.get_phrase(f"severity_levels.{severity}")
    
    def get_encouragement(self):
        """Get a random encouragement message in current language"""
        import random
        encouragements = PHRASES["encouragement"].get(self.current_language, 
                                                     PHRASES["encouragement"]["en"])
        return random.choice(encouragements)

# Create global instance
localization_manager = LocalizationManager()

def get_localized_phrase(key, **kwargs):
    """Convenience function to get localized phrases"""
    return localization_manager.get_phrase(key, **kwargs)

def set_language(language_code):
    """Convenience function to set language"""
    return localization_manager.set_language(language_code)

def get_current_language():
    """Get current language code"""
    return localization_manager.current_language