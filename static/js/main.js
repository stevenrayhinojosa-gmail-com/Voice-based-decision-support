// Main JavaScript file for the application

// Voice Recognition Helper Functions
function initVoiceRecognition(statusElement, recordButton, resultElement, apiEndpoint, onResultCallback) {
    if (!statusElement || !recordButton || !resultElement) {
        console.error('Missing required elements for voice recognition');
        return;
    }
    
    // Check if the browser supports the Web Speech API
    if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
        statusElement.innerHTML = `
            <div class="alert alert-warning">
                <i class="fas fa-exclamation-triangle me-2"></i>
                Your browser doesn't support voice recognition. Please use Chrome or Edge.
            </div>
        `;
        recordButton.disabled = true;
        return;
    }
    
    // Use the SpeechRecognition interface
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    const recognition = new SpeechRecognition();
    
    // Configure recognition
    recognition.continuous = false;
    recognition.interimResults = true;
    recognition.lang = 'en-US';
    
    // Variables to track state
    let isRecording = false;
    let finalTranscript = '';
    let interimTranscript = '';
    
    // Event listeners for recognition
    recognition.onstart = function() {
        isRecording = true;
        statusElement.innerHTML = `
            <div class="alert alert-info">
                <div class="spinner-border spinner-border-sm me-2" role="status">
                    <span class="visually-hidden">Listening...</span>
                </div>
                Listening... Speak now
            </div>
        `;
        recordButton.innerHTML = '<i class="fas fa-stop me-2"></i> Stop Recording';
        recordButton.classList.remove('btn-primary');
        recordButton.classList.add('btn-danger');
    };
    
    recognition.onresult = function(event) {
        interimTranscript = '';
        
        for (let i = event.resultIndex; i < event.results.length; ++i) {
            if (event.results[i].isFinal) {
                finalTranscript += event.results[i][0].transcript;
            } else {
                interimTranscript += event.results[i][0].transcript;
            }
        }
        
        // Update the result element
        resultElement.value = finalTranscript + interimTranscript;
    };
    
    recognition.onerror = function(event) {
        console.error('Recognition error:', event.error);
        statusElement.innerHTML = `
            <div class="alert alert-danger">
                <i class="fas fa-exclamation-circle me-2"></i>
                Error: ${event.error}. Please try again.
            </div>
        `;
        isRecording = false;
        resetButton();
    };
    
    recognition.onend = function() {
        isRecording = false;
        resetButton();
        
        if (finalTranscript) {
            statusElement.innerHTML = `
                <div class="alert alert-success">
                    <i class="fas fa-check-circle me-2"></i>
                    Voice captured successfully!
                </div>
            `;
            
            // Call the API if an endpoint is provided
            if (apiEndpoint && onResultCallback) {
                sendVoiceToAPI(finalTranscript, apiEndpoint, onResultCallback);
            }
        } else {
            statusElement.innerHTML = `
                <div class="alert alert-warning">
                    <i class="fas fa-exclamation-triangle me-2"></i>
                    No speech detected. Please try again.
                </div>
            `;
        }
    };
    
    // Toggle recording when the button is clicked
    recordButton.addEventListener('click', function() {
        if (isRecording) {
            recognition.stop();
        } else {
            finalTranscript = '';
            recognition.start();
        }
    });
    
    function resetButton() {
        recordButton.innerHTML = '<i class="fas fa-microphone me-2"></i> Start Recording';
        recordButton.classList.remove('btn-danger');
        recordButton.classList.add('btn-primary');
    }
    
    function sendVoiceToAPI(text, endpoint, callback) {
        fetch(endpoint, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ simulated_text: text })
        })
        .then(response => response.json())
        .then(data => {
            if (callback && typeof callback === 'function') {
                callback(data);
            }
        })
        .catch(error => {
            console.error('Error:', error);
            statusElement.innerHTML = `
                <div class="alert alert-danger">
                    <i class="fas fa-exclamation-circle me-2"></i>
                    Error sending voice data: ${error.message}
                </div>
            `;
        });
    }
    
    // Return an object with functions that can be called externally
    return {
        start: function() {
            if (!isRecording) {
                finalTranscript = '';
                recognition.start();
            }
        },
        stop: function() {
            if (isRecording) {
                recognition.stop();
            }
        },
        isRecording: function() {
            return isRecording;
        }
    };
}

// Wait for the DOM to be fully loaded
document.addEventListener('DOMContentLoaded', function() {
    // Toggle terminal option fields in decision option form
    const terminalOptionCheckbox = document.getElementById('is_terminal');
    const recommendationField = document.getElementById('recommendation-field');
    const nextDecisionField = document.getElementById('next_decision-field');
    
    if (terminalOptionCheckbox && recommendationField && nextDecisionField) {
        // Set initial state
        updateOptionFields();
        
        // Add event listener for changes
        terminalOptionCheckbox.addEventListener('change', updateOptionFields);
        
        function updateOptionFields() {
            if (terminalOptionCheckbox.checked) {
                recommendationField.style.display = 'block';
                nextDecisionField.style.display = 'none';
            } else {
                recommendationField.style.display = 'none';
                nextDecisionField.style.display = 'block';
            }
        }
    }
    
    // Dynamic form fields for prediction based on selected model
    const modelSelectElement = document.getElementById('model_id');
    const predictionFormFields = document.getElementById('prediction-form-fields');
    
    if (modelSelectElement && predictionFormFields) {
        // Initial load
        if (modelSelectElement.value) {
            loadModelFields(modelSelectElement.value);
        }
        
        // Add event listener for changes
        modelSelectElement.addEventListener('change', function() {
            loadModelFields(this.value);
        });
        
        function loadModelFields(modelId) {
            // Clear existing fields
            predictionFormFields.innerHTML = '<div class="spinner-border text-primary" role="status"><span class="visually-hidden">Loading...</span></div>';
            
            // Fetch the fields for this model
            fetch(`/api/model/${modelId}/fields`)
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        // Clear loading spinner
                        predictionFormFields.innerHTML = '';
                        
                        // Create input fields for each feature
                        data.fields.forEach(field => {
                            const formGroup = document.createElement('div');
                            formGroup.className = 'mb-3';
                            
                            const label = document.createElement('label');
                            label.htmlFor = field;
                            label.className = 'form-label';
                            label.textContent = field.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
                            
                            const input = document.createElement('input');
                            input.type = 'text';
                            input.className = 'form-control';
                            input.id = field;
                            input.name = field;
                            input.required = true;
                            
                            formGroup.appendChild(label);
                            formGroup.appendChild(input);
                            predictionFormFields.appendChild(formGroup);
                        });
                    } else {
                        predictionFormFields.innerHTML = `<div class="alert alert-danger">Error: ${data.error}</div>`;
                    }
                })
                .catch(error => {
                    predictionFormFields.innerHTML = `<div class="alert alert-danger">Error: ${error.message}</div>`;
                });
        }
    }
    
    // Initialize any tooltips
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'))
    const tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl)
    });
    
    // Initialize any popovers
    const popoverTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="popover"]'))
    const popoverList = popoverTriggerList.map(function (popoverTriggerEl) {
        return new bootstrap.Popover(popoverTriggerEl)
    });
    
    // Add event listeners for protocol decision tree visualization
    const treeToggles = document.querySelectorAll('.tree-toggle');
    
    treeToggles.forEach(toggle => {
        toggle.addEventListener('click', function(e) {
            e.preventDefault();
            const targetId = this.getAttribute('data-target');
            const targetElement = document.getElementById(targetId);
            
            if (targetElement) {
                if (targetElement.style.display === 'none') {
                    targetElement.style.display = 'block';
                    this.querySelector('i').classList.remove('fa-plus');
                    this.querySelector('i').classList.add('fa-minus');
                } else {
                    targetElement.style.display = 'none';
                    this.querySelector('i').classList.remove('fa-minus');
                    this.querySelector('i').classList.add('fa-plus');
                }
            }
        });
    });
});
