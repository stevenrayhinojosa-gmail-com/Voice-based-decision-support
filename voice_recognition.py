import speech_recognition as sr
import logging
import threading
import time
from flask import session
from context_sensors import context_sensor  # Import the context sensor

# Set up logging
logger = logging.getLogger(__name__)

class VoiceRecognizer:
    """
    Class to handle voice recognition for behavioral decision support
    """
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.mic = None
        self.is_listening = False
        self.listen_thread = None
        self.callbacks = []
        
    def initialize_microphone(self):
        """Initialize the microphone for capture"""
        try:
            self.mic = sr.Microphone()
            # Adjust for ambient noise
            with self.mic as source:
                logger.info("Adjusting for ambient noise...")
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
                logger.info("Microphone initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Error initializing microphone: {str(e)}")
            return False
            
    def listen_once(self):
        """Capture a single voice input and convert to text"""
        if not self.mic:
            if not self.initialize_microphone():
                return {"success": False, "error": "Could not initialize microphone"}
        
        # Check that mic is properly initialized
        if self.mic is None:
            return {"success": False, "error": "Microphone not available"}
        
        try:
            with self.mic as source:
                logger.info("Listening for voice input...")
                audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=10)
                
            logger.info("Processing voice input...")
            # Google's speech recognition service
            text = self.recognizer.recognize_google(audio)
            logger.info(f"Recognized: {text}")
            return {"success": True, "text": text}
            
        except sr.WaitTimeoutError:
            logger.warning("No speech detected within timeout period")
            return {"success": False, "error": "No speech detected. Please try again."}
        except sr.UnknownValueError:
            logger.warning("Speech was unintelligible")
            return {"success": False, "error": "Could not understand audio. Please try again."}
        except sr.RequestError as e:
            logger.error(f"Error with speech recognition service: {str(e)}")
            return {"success": False, "error": f"Speech recognition service error: {str(e)}"}
        except Exception as e:
            logger.error(f"Unexpected error in speech recognition: {str(e)}")
            return {"success": False, "error": f"Error processing voice: {str(e)}"}
    
    def register_callback(self, callback_func):
        """Register a callback function to process recognized speech"""
        self.callbacks.append(callback_func)
    
    def _listen_loop(self):
        """Background listening loop"""
        while self.is_listening:
            result = self.listen_once()
            if result["success"]:
                for callback in self.callbacks:
                    callback(result["text"])
            time.sleep(0.5)  # Small delay to prevent CPU overuse
    
    def start_continuous_listening(self):
        """Start continuous listening in background thread"""
        if self.is_listening:
            logger.warning("Already listening")
            return False
        
        if not self.mic and not self.initialize_microphone():
            return False
            
        self.is_listening = True
        self.listen_thread = threading.Thread(target=self._listen_loop)
        self.listen_thread.daemon = True
        self.listen_thread.start()
        logger.info("Started continuous listening")
        return True
    
    def stop_continuous_listening(self):
        """Stop continuous listening"""
        self.is_listening = False
        if self.listen_thread and self.listen_thread.is_alive():
            self.listen_thread.join(timeout=2)
        logger.info("Stopped continuous listening")
        
    def process_voice_for_decision_support(self, protocol_id=None):
        """
        Process voice input specifically for decision support
        If protocol_id is provided, it will be used; otherwise,
        it will try to get it from the session
        
        Now includes context sensing for time-of-day and ambient noise
        """
        # Get time period before voice capture
        context_data = context_sensor.get_context_data()
        time_period = context_data['time_period']['name']
        
        # Capture voice input
        voice_result = self.listen_once()
        
        # If voice recognition failed, still include context data
        if not voice_result["success"]:
            voice_result.update({
                "time_period": time_period,
                "noise_level_db": context_data['noise_level_db']
            })
            return voice_result
        
        # Get noise level from the context sensor
        # Note: We already measured ambient noise during initialization,
        # but we take a fresh sample here for the current conditions
        noise_level = context_data['noise_level_db']
        
        # Add context data to result
        logger.info(f"Context data: Time period = {time_period}, Noise level = {noise_level} dB")
            
        # Return the recognized text along with the protocol context and environmental context
        return {
            "success": True,
            "text": voice_result["text"],
            "protocol_id": protocol_id or session.get('current_protocol_id'),
            "time_period": time_period,
            "noise_level_db": noise_level,
            "is_transition_period": context_data['time_period']['is_transition']
        }

# Create a global instance for use across the application
voice_recognizer = VoiceRecognizer()

def extract_keywords_from_speech(speech_text):
    """
    Extract keywords from speech text that might be relevant 
    for behavioral decision making
    """
    # This is a simple implementation that could be enhanced with NLP
    keywords = []
    
    # Categorized behavioral descriptors for more nuanced analysis
    behavioral_categories = {
        "safety_concerns": [
            "danger", "unsafe", "risk", "harm", "emergency", "crisis", 
            "injured", "violent", "threatening", "weapon", "hurt", "damage",
            "safety", "dangerous", "hazard", "threat", "self-harm", "suicide"
        ],
        "escalation_indicators": [
            "escalating", "increasing", "intensifying", "worsening", "building up",
            "getting worse", "growing", "heightened", "agitated", "explosive"
        ],
        "de_escalation_indicators": [
            "calming", "settling", "relaxing", "de-escalating", "decreasing",
            "subsiding", "improving", "quieter", "calmer", "less agitated"
        ],
        "aggression_descriptors": [
            "aggressive", "hitting", "kicking", "throwing", "pushing", "biting",
            "spitting", "punching", "attacking", "violent", "destructive",
            "breaking", "slamming", "physical", "hurting", "fighting"
        ],
        "emotional_states": [
            "angry", "sad", "frustrated", "anxious", "upset", "afraid", "scared",
            "worried", "distressed", "panicked", "overwhelmed", "agitated",
            "irritated", "annoyed", "furious", "enraged", "terrified"
        ],
        "behavioral_descriptors": [
            "disruptive", "withdrawn", "defiant", "impulsive", "inattentive", 
            "hyperactive", "crying", "shouting", "screaming", "yelling",
            "running", "hiding", "refused", "noncompliant", "oppositional",
            "resistant", "avoidant", "escape", "flight", "unresponsive"
        ],
        "redirection_responses": [
            "listening", "responding", "following", "complying", "cooperating",
            "accepting", "receiving", "acknowledging", "ignoring", "refusing",
            "rejecting", "resistant", "unresponsive", "redirect", "direction"
        ],
        "regulation_needs": [
            "space", "break", "time", "calm down", "cool off", "quiet", 
            "alone", "separation", "distance", "breathe", "regulate",
            "self-control", "coping", "strategies", "techniques", "skills"
        ]
    }
    
    # Process the speech text to handle word variations
    speech_lower = speech_text.lower()
    
    # Extract keywords by category
    for category, terms in behavioral_categories.items():
        for term in terms:
            # Check for whole word or phrase match
            if f" {term} " in f" {speech_lower} " or speech_lower.startswith(f"{term} ") or speech_lower.endswith(f" {term}") or speech_lower == term:
                keywords.append(term)
                continue
                
            # For single-word terms, also check for partial matches within words
            if len(term.split()) == 1:
                # Don't match small terms (3 letters or less) as parts of other words
                if len(term) <= 3:
                    continue
                
                # Check if the term appears as part of words
                words = speech_lower.split()
                if any(term in word and len(word) > len(term) for word in words):
                    keywords.append(term)
    
    return keywords

def analyze_speech_for_decision(speech_text, protocol_id=None, time_period=None, noise_level_db=None, is_transition_period=False):
    """
    Analyze speech text and map it to decision points in a protocol
    Returns the appropriate option based on the speech content and context data
    
    Parameters:
    - speech_text: The recognized speech text
    - protocol_id: The ID of the protocol to use
    - time_period: The current school time period (e.g., "morning-block-1", "lunch")
    - noise_level_db: The ambient noise level in decibels
    - is_transition_period: Whether the current time period is a transition period
    """
    from models import Protocol, DecisionPoint, DecisionOption
    
    # Get keywords from speech
    keywords = extract_keywords_from_speech(speech_text)
    logger.info(f"Extracted keywords: {keywords}")
    
    # If no protocol specified, cannot proceed with mapping
    if not protocol_id:
        return {
            "success": False,
            "error": "No protocol specified for decision mapping"
        }
    
    try:
        # Get the protocol and the current decision point (or first if none specified)
        protocol = Protocol.query.get(protocol_id)
        if not protocol:
            return {"success": False, "error": "Protocol not found"}
        
        # Use session to track the current decision point if available
        from flask import session
        current_dp_id = session.get('current_dp_id')
        
        if current_dp_id:
            current_dp = DecisionPoint.query.get(current_dp_id)
            if not current_dp or current_dp.protocol_id != protocol_id:
                # If the stored decision point doesn't exist or belongs to a different protocol,
                # fall back to the first decision point
                current_dp = DecisionPoint.query.filter_by(
                    protocol_id=protocol_id
                ).order_by(DecisionPoint.order).first()
        else:
            # Start with the first decision point in the protocol
            current_dp = DecisionPoint.query.filter_by(
                protocol_id=protocol_id
            ).order_by(DecisionPoint.order).first()
        
        if not current_dp:
            return {"success": False, "error": "Protocol has no decision points"}
        
        # Get options for the current decision point
        options = DecisionOption.query.filter_by(decision_point_id=current_dp.id).all()
        if not options:
            return {"success": False, "error": "Current decision point has no options"}
        
        # Initialize decision mapping analysis
        decision_mapping = {
            # Decision point 1: Is there an immediate safety concern?
            1: {
                "keywords": {
                    "yes": ["danger", "unsafe", "risk", "harm", "emergency", "crisis", 
                           "violent", "threatening", "weapon", "hurt", "injured", "damage",
                           "safety", "dangerous", "hazard", "threat", "self-harm", "suicide"],
                    "no": ["safe", "controlled", "manageable", "calm", "no danger", "settled"]
                },
                "default": "no"  # Default to safer option if unclear
            },
            
            # Decision point 2: Is the student escalating or de-escalating?
            2: {
                "keywords": {
                    "escalating": ["escalating", "increasing", "intensifying", "worsening", 
                                 "building up", "getting worse", "growing", "heightened", 
                                 "agitated", "explosive", "louder", "faster", "angrier"],
                    "de-escalating": ["calming", "settling", "relaxing", "de-escalating", 
                                     "decreasing", "subsiding", "improving", "quieter", 
                                     "calmer", "less agitated", "breathing", "listening"]
                },
                "default": "escalating"  # Default to the more cautious option
            },
            
            # Decision point 3: Has the student displayed physical aggression?
            3: {
                "keywords": {
                    "yes": ["aggressive", "hitting", "kicking", "throwing", "pushing", "biting",
                           "spitting", "punching", "attacking", "violent", "destructive",
                           "breaking", "slamming", "physical", "hurting", "fighting"],
                    "no": ["verbal", "talking", "shouting", "crying", "no contact", "no hitting",
                          "not physical", "not touching", "keeping distance"]
                },
                "default": "no"
            },
            
            # Decision point 4: Is the student responsive to verbal redirection?
            4: {
                "keywords": {
                    "yes": ["listening", "responding", "following", "complying", "cooperating",
                           "accepting", "receiving", "acknowledging", "nodding", "calming"],
                    "no": ["ignoring", "refusing", "rejecting", "resistant", "unresponsive", 
                          "not listening", "defiant", "oppositional", "escalating"]
                },
                "default": "no"
            },
            
            # Decision point 5: Does the student need space or time to calm down?
            5: {
                "keywords": {
                    "yes": ["space", "break", "time", "calm down", "cool off", "quiet", 
                           "alone", "separation", "distance", "breathe", "regulate",
                           "overwhelmed", "overstimulated", "sensory", "too much"],
                    "no": ["engage", "talk", "connect", "interact", "support", "present", 
                          "close", "near", "responding", "contact"]
                },
                "default": "yes"
            }
        }
        
        # Determine if this is an emergency situation
        safety_keywords = decision_mapping.get(1, {}).get("keywords", {}).get("yes", [])
        is_emergency = any(keyword in safety_keywords for keyword in keywords)
        
        # Map the speech to the current decision point
        if current_dp.id in decision_mapping:
            mapping = decision_mapping[current_dp.id]
            
            # Find the best matching option
            best_match = None
            max_matches = 0
            
            for option_text, option_keywords in mapping["keywords"].items():
                # Count how many keywords match this option
                matches = sum(1 for k in keywords if k in option_keywords)
                if matches > max_matches:
                    max_matches = matches
                    best_match = option_text
            
            # If no clear match, use the default
            if best_match is None or max_matches == 0:
                best_match = mapping["default"]
        
            # Find the corresponding option object
            selected_option = None
            for option in options:
                # Case-insensitive match that looks for the option text within the best match
                if best_match.lower() in option.text.lower():
                    selected_option = option
                    break
            
            if selected_option:
                # Adjust decision weight based on context
                adjusted_confidence = max_matches / (len(keywords) if keywords else 1)
                
                # Context-aware adjustments
                # Increase confidence if in a transition period and there are behavior keywords
                if is_transition_period and any(k in keywords for k in decision_mapping.get(3, {}).get("keywords", {}).get("yes", [])):
                    # During transitions, behavioral issues are more common
                    logger.info("Adjusting confidence due to transition period")
                    adjusted_confidence *= 1.2
                    
                # If noise level is high and keywords suggest a crisis, increase emergency confidence
                if noise_level_db and noise_level_db > -30:  # -30dB threshold for noisy environments
                    # In louder environments, safety concerns become more critical
                    logger.info(f"Considering high noise level ({noise_level_db} dB) in decision making")
                    if is_emergency:
                        adjusted_confidence *= 1.1
                
                return {
                    "success": True,
                    "decision_point": current_dp,
                    "selected_option": selected_option,
                    "is_terminal": selected_option.is_terminal,
                    "next_decision_id": selected_option.next_decision_id,
                    "recommendation": selected_option.recommendation if selected_option.is_terminal else None,
                    "keywords": keywords,
                    "is_emergency": is_emergency,
                    "matched_intent": best_match,
                    "confidence": adjusted_confidence,
                    # Include context data in the response
                    "time_period": time_period,
                    "noise_level_db": noise_level_db,
                    "is_transition_period": is_transition_period,
                    "context_adjusted": True
                }
            else:
                # Fallback if we couldn't map to a specific option
                return {
                    "success": False,
                    "error": f"Could not find option matching intent '{best_match}'",
                    "keywords": keywords,
                    "decision_point": current_dp,
                    "is_emergency": is_emergency
                }
        else:
            # For decision points without specific mappings, use simple yes/no mapping
            yes_option = next((opt for opt in options if "yes" in opt.text.lower()), None)
            no_option = next((opt for opt in options if "no" in opt.text.lower()), None)
            
            # Simple logic - count positive vs negative sentiment words
            positive_indicators = ["yes", "agree", "correct", "right", "affirmative", "true", "yeah"]
            negative_indicators = ["no", "disagree", "incorrect", "wrong", "negative", "false", "nope"]
            
            positive_count = sum(1 for word in speech_text.lower().split() if word in positive_indicators)
            negative_count = sum(1 for word in speech_text.lower().split() if word in negative_indicators)
            
            # Select the option based on the counts
            selected_option = yes_option if positive_count > negative_count else no_option
            
            if selected_option:
                return {
                    "success": True,
                    "decision_point": current_dp,
                    "selected_option": selected_option,
                    "is_terminal": selected_option.is_terminal,
                    "next_decision_id": selected_option.next_decision_id,
                    "recommendation": selected_option.recommendation if selected_option.is_terminal else None,
                    "keywords": keywords,
                    "is_emergency": is_emergency
                }
            else:
                return {
                    "success": False,
                    "error": "Could not map speech to decision options",
                    "keywords": keywords,
                    "decision_point": current_dp,
                    "is_emergency": is_emergency
                }
    
    except Exception as e:
        logger.error(f"Error analyzing speech for decision: {str(e)}")
        return {"success": False, "error": str(e)}