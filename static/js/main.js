// Main JavaScript file for the application

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
