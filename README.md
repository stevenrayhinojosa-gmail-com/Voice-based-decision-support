# Voice-Based Behavioral Decision Support System

## Overview

This application provides a specialized voice-only interface for teachers and school staff to receive immediate recommendations during behavioral incidents with students. The system uses context-aware decision making that incorporates environmental factors to provide tailored guidance.

## Features

- **Voice-Only Interface**: Simple, intuitive interface activated with a single button press
- **Context-Aware Decisions**: Takes into account time of day, noise levels, and physical location
- **Specialized Behavioral Protocols**: Incorporates educational best practices from SAMA and PfISD protocols
- **Real-time Recommendations**: Provides immediate guidance without follow-up questions
- **Environmental Context Detection**: Automatically senses ambient noise levels and time periods
- **Natural Language Processing**: Understands complex behavioral queries and descriptions
- **Button-Activated Speech Recognition**: Press-to-speak functionality for controlled input

## How to Use

1. Open the application in a web browser
2. Press the "Activate Voice Input" button
3. Describe the student behavior situation clearly
4. Review the immediate recommendation displayed on screen
5. Follow the suggested protocol steps based on the specific situation

## Technology Stack

- **Backend**: Python with Flask framework
- **Database**: PostgreSQL for data storage
- **Machine Learning**: scikit-learn for behavioral pattern recognition
- **Voice Processing**: Speech Recognition with PyAudio
- **Natural Language Processing**: NLTK for text analysis
- **Frontend**: Bootstrap 5 for responsive design

## System Requirements

- Python 3.11+
- PostgreSQL database
- Web browser with microphone access

## Contextual Factors

The system considers these contextual elements when providing recommendations:

- **Time Period**: Different times of day (e.g., morning classes, lunch, transition periods)
- **Noise Level**: Ambient noise detection to gauge classroom environment
- **Physical Setting**: Location context (classroom, hallway, playground, etc.)

## Behavioral Decision Process

1. Voice input captures the teacher's description of the situation
2. Natural language processing extracts key behavioral indicators
3. Context sensors provide environmental data
4. Decision engine matches situation to appropriate protocol
5. Recommendation is presented with actionable steps

## Privacy and Data

- All behavioral data is stored securely
- No personally identifiable student information is recorded
- Speech data is processed locally and not retained after analysis

---

Created for educational contexts to provide rapid, evidence-based responses to student behavioral situations.