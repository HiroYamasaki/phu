$(document).ready(function () {
    let currentStep = 1;
    const totalSteps = 4;

    function updateProgressBar() {
        const progress = (currentStep / totalSteps) * 100;
        $('#progress-bar').css('width', progress + '%');
        $('#progress-bar').text('Step ' + currentStep + ' of ' + totalSteps);
    }

    $('.next-btn').click(function () {
        $('#step-' + currentStep).removeClass('active');
        currentStep++;
        $('#step-' + currentStep).addClass('active');
        updateProgressBar();
    });

    $('.prev-btn').click(function () {
        $('#step-' + currentStep).removeClass('active');
        currentStep--;
        $('#step-' + currentStep).addClass('active');
        updateProgressBar();
    });

    $('#multi-step-form').submit(function (event) {
        event.preventDefault();
        $.ajax({
            url: '/predict',
            method: 'POST',
            data: $(this).serialize(),
            success: function (response) {
                let predictionText = response.prediction === 1 ? "Depression Detected" : "No Depression Detected";

                if (predictionText === "Depression Detected") {
                    $('#prediction-result').html(`
                        <div id="advice-message">
                            <h4>Depression Detected (┬┬﹏┬┬)</h4>
                            <h4>We're here to support you through this challenging time</h4>
                            <p>Your responses indicate signs of depression. Remember that seeking help is a sign of strength, not weakness. There are many resources and professionals ready to support you on your journey to better mental health.</p>
                            <p>Here are some immediate resources available:</p>
                            <ul>
                                <li>24/7 Mental Health Hotline: 1-800-273-8255</li>
                                <li>Local Mental Health Centers</li>
                                <li>Online Counseling Services</li>
                                <li>Mental Health Support Apps</li>
                            </ul>
                            <p>You're not alone in this journey, and recovery is possible with the right support.</p>
                        </div>
                    `);
                } else {
                    $('#prediction-result').html(`
                        <div id="advice-message" class="no-depression">
                            <h4>No Depression Detected (●'◡'●)</h4>
                            <h4>Thank you for taking care of your mental health</h4>
                            <p>While your responses don't indicate clinical depression, mental health is an ongoing journey. Here are some ways to maintain your emotional well-being:</p>
                            <ul>
                                <li>Practice regular exercise and maintain healthy sleep habits</li>
                                <li>Stay connected with friends and family</li>
                                <li>Engage in activities you enjoy</li>
                                <li>Practice stress management techniques</li>
                            </ul>
                            <p>If you ever feel overwhelmed in the future, don't hesitate to reach out to mental health professionals.</p>
                        </div>
                    `);
                }
            }
        });
    });

    // Age Validation (Step 1)
    document.getElementById('age').addEventListener('input', function () {
        let ageInput = this.value;
        let errorElement = document.getElementById('age-error');

        if (ageInput === '') {
            errorElement.textContent = 'Age is required.';
            this.classList.add('is-invalid');
        } else if (ageInput < 6 || ageInput > 100) {
            errorElement.textContent = 'Age must be between 6 and 100.';
            this.classList.add('is-invalid');
        } else {
            errorElement.textContent = '';
            this.classList.remove('is-invalid');
            this.classList.add('is-valid');
        }
    });

    // Academic Pressure Validation (Step 2)
    document.getElementById('academic_pressure').addEventListener('input', function () {
        validateNumberInput(this, 0, 5, 'Academic pressure must be between 0 and 5.');
    });

    // CGPA Validation (Step 2)
    document.getElementById('cgpa').addEventListener('input', function () {
        validateNumberInput(this, 0, 10, 'CGPA must be between 0 và 10.');
    });

    // Study Satisfaction Validation (Step 2)
    document.getElementById('study_satisfaction').addEventListener('input', function () {
        validateNumberInput(this, 0, 5, 'Study satisfaction must be between 0 và 5.');
    });

    // Work/Study Hours Validation (Step 2)
    document.getElementById('work_study_hours').addEventListener('input', function () {
        validateNumberInput(this, 0, 12, 'Work/Study hours must be between 0 và 12.');
    });

    // Financial Stress Validation (Step 4)
    document.getElementById('financial_stress').addEventListener('input', function () {
        validateNumberInput(this, 0, 5, 'Financial stress must be between 0 và 5.');
    });

    // Generic Validation Function
    function validateNumberInput(element, min, max, errorMessage) {
        let value = parseFloat(element.value);
        if (isNaN(value) || value < min || value > max) {
            element.classList.add('is-invalid');
            element.classList.remove('is-valid');
            if (!element.nextElementSibling || !element.nextElementSibling.classList.contains('text-danger')) {
                let errorSpan = document.createElement('small');
                errorSpan.className = 'form-text text-danger';
                errorSpan.textContent = errorMessage;
                element.parentNode.appendChild(errorSpan);
            } else {
                element.nextElementSibling.textContent = errorMessage;
            }
        } else {
            element.classList.remove('is-invalid');
            element.classList.add('is-valid');
            if (element.nextElementSibling && element.nextElementSibling.classList.contains('text-danger')) {
                element.nextElementSibling.textContent = '';
            }
        }
    }

    const rangeInput = document.getElementById('work_study_hours');
    const rangeValue = document.getElementById('work_study_hours_value');

    rangeInput.addEventListener('input', function() {
        rangeValue.textContent = rangeInput.value;
    });
});