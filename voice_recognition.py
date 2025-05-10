import speech_recognition as sr
import logging
import threading
import time
from flask import session

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
        """
        voice_result = self.listen_once()
        
        if not voice_result["success"]:
            return voice_result
            
        # Return the recognized text along with the protocol context
        return {
            "success": True,
            "text": voice_result["text"],
            "protocol_id": protocol_id or session.get('current_protocol_id')
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
    
    # Common behavioral descriptors
    behavioral_terms = [
        "aggressive", "disruptive", "anxious", "withdrawn", "defiant",
        "impulsive", "inattentive", "hyperactive", "violent", "threatening",
        "crying", "shouting", "hitting", "throwing", "running", "hiding",
        "refused", "angry", "sad", "frustrated", "calm", "escalating",
        "danger", "unsafe", "risk", "harm", "emergency", "crisis"
    ]
    
    # Extract words that match our behavioral terms
    words = speech_text.lower().split()
    for term in behavioral_terms:
        if term in words or any(term in word for word in words):
            keywords.append(term)
    
    return keywords

def analyze_speech_for_decision(speech_text, protocol_id=None):
    """
    Analyze speech text and map it to decision points in a protocol
    Returns the appropriate option based on the speech content
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
        # Get the protocol and its first decision point
        protocol = Protocol.query.get(protocol_id)
        if not protocol:
            return {"success": False, "error": "Protocol not found"}
            
        first_dp = DecisionPoint.query.filter_by(
            protocol_id=protocol_id
        ).order_by(DecisionPoint.order).first()
        
        if not first_dp:
            return {"success": False, "error": "Protocol has no decision points"}
        
        # Map speech keywords to decision options
        # This is where more sophisticated NLP could be implemented
        
        # Emergency keywords that might indicate danger
        danger_keywords = ["danger", "unsafe", "risk", "harm", "emergency", "crisis", 
                          "violent", "threatening", "hitting", "throwing"]
        
        # Check for emergency keywords first as they might need immediate attention
        is_emergency = any(keyword in danger_keywords for keyword in keywords)
        
        # Get options for the first decision point
        options = DecisionOption.query.filter_by(decision_point_id=first_dp.id).all()
        
        # For simplicity, we'll map to "Yes" if emergency keywords are found,
        # otherwise to "No" for the first decision point in our sample protocol
        # (assuming it asks about danger)
        selected_option = None
        for option in options:
            # Very simple matching logic - could be much more sophisticated
            if is_emergency and "yes" in option.text.lower():
                selected_option = option
                break
            elif not is_emergency and "no" in option.text.lower():
                selected_option = option
                break
        
        if selected_option:
            return {
                "success": True,
                "decision_point": first_dp,
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
                "keywords": keywords
            }
    
    except Exception as e:
        logger.error(f"Error analyzing speech for decision: {str(e)}")
        return {"success": False, "error": str(e)}