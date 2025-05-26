let timeRemaining; // Time in seconds
let progress = 0; // Initial progress value
let totalQuestions = 0; // Total number of questions
let questionsAnswered = 0; // Questions answered by the user

function startCountdown() {
    const countdownElement = document.getElementById('time');
    const interval = setInterval(() => {
        if (timeRemaining <= 0) {
            clearInterval(interval);
            alert("Time's up!");
            submitActivity();
        } else {
            timeRemaining--;
            const minutes = Math.floor(timeRemaining / 60);
            const seconds = timeRemaining % 60;
            countdownElement.textContent = `${minutes}:${seconds < 10 ? '0' : ''}${seconds}`;
        }
    }, 1000);
}

function updateProgress() {
    const progressElement = document.getElementById('progress');
    const progressTextElement = document.getElementById('progress-text');

    // Calculate progress based on answered questions
    progress = Math.floor((questionsAnswered / totalQuestions) * 100);
    progressElement.style.width = `${progress}%`;
    progressTextElement.textContent = `Progress: ${progress}%`;
}

function submitActivity() {
    // Gather answers
    const formData = new FormData(document.getElementById('mcqForm'));
    const answers = {};
    for (let [key, value] of formData.entries()) {
        answers[key] = value;
    }

    // Example: Sending answers to backend
    fetch('/api/submit_activity', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ answers }),
    })
    .then(response => {
        if (response.ok) {
            alert("Activity submitted successfully!");
            window.location.href = 'lecture.html'; // Redirect back to lecture page
        } else {
            alert("Error submitting activity.");
        }
    });
}

function fetchActivity() {
    fetch('/api/activity')
        .then(response => response.json())
        .then(data => {
            const activityContainer = document.getElementById('activity-container');
            document.getElementById('activity-title').textContent = data.activity_name;
            timeRemaining = data.time_limit * 60; // Convert minutes to seconds
            totalQuestions = data.questions.length; // Set total questions count
            startCountdown();

            data.questions.forEach((question, index) => {
                const questionDiv = document.createElement('div');
                questionDiv.innerHTML = `
                    <h2>Question ${index + 1}</h2>
                    <p>${question.question}</p>
                    ${generateOptions(question)}
                `;
                activityContainer.appendChild(questionDiv);
            });

            // Add event listener to track answered questions
            const inputs = document.querySelectorAll('input[type="radio"], input[type="text"], textarea');
            inputs.forEach(input => {
                input.addEventListener('change', () => {
                    // Update questions answered count
                    const answered = new Set();
                    document.querySelectorAll('input:checked, input[type="text"][value], textarea[value]').forEach(input => {
                        answered.add(input.name);
                    });
                    questionsAnswered = answered.size;

                    // Update progress bar
                    updateProgress();
                });
            });
        });
}

function generateOptions(question) {
    if (question.type === 'mcq') {
        return `
            <ul>
                ${question.options.map((option, index) => `
                    <li>
                        <input type="radio" name="q${question.question_id}" value="${option}"> ${option}
                    </li>
                `).join('')}
            </ul>
        `;
    } else if (question.type === 'fill_in_blanks') {
        return `<input type="text" name="q${question.question_id}" placeholder="Fill in the blank">`;
    } else if (question.type === 'short_answer') {
        return `<textarea name="q${question.question_id}" rows="4" cols="50" placeholder="Type your answer here"></textarea>`;
    }
    return '';
}

document.addEventListener('DOMContentLoaded', (event) => {
    fetchActivity();
    updateProgress(); // Initialize the progress bar
});