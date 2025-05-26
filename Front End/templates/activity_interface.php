<?php
session_start();
if (!isset($_SESSION['sessionid'])) {
    $_SESSION['sessionid'] = session_id(); 
}
?>

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Activity Interface</title>
    <link rel="stylesheet" href="../static/student.css">
</head>
<body>
    <header>
        <img src="images/logo.png" alt="Logo" id="logo">
        <div id="profile">
            <img src="images/profile.png" alt="Student Profile" id="profile-pic">
        </div>
        <div id="progress-container">
            <div id="progress-bar">
                <div id="progress"></div>
            </div>
            <span id="progress-text">Progress: 0%</span>
        </div>
    </header>
    <main>
        <div id="activity-wrapper">
            <div id="activity-header">
                <h1 id="activity-title">Lecturer's Activities</h1>
                <div id="countdown">Time left: <span id="time">2:00</span></div>
            </div>
            <div id="activity-container">
                
            </div>
            <button onclick="submitActivity()">Submit</button>
        </div>
    </main>
    <script src="../static/student.js"></script>
    <script>
        document.addEventListener('DOMContentLoaded', function() {
           
            fetch(`/get_score.php?sessionid=<?php echo $_SESSION['sessionid']; ?>`)
                .then(response => response.json())
                .then(data => {
                    if (data.status === 'success') {
                        const score = data.score;
                        const progress = Math.min(100, score); 
                        document.getElementById('progress').style.width = progress + '%';
                        document.getElementById('progress-text').innerText = `Progress: ${progress}%`;
                    } else {
                        console.error(data.message);
                    }
                })
                .catch(error => console.error('Error fetching score:', error));
        });

        function submitActivity() {
            const answers = {};
            document.querySelectorAll('.question input').forEach(input => {
                answers[input.name] = input.value;
            });
            
            fetch('/submit_answers.php', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ answers, sessionid: "<?php echo $_SESSION['sessionid']; ?>" })
            })
            .then(response => response.json())
            .then(data => {
                console.log('Activity submitted:', data);
            })
            .catch(error => console.error('Error submitting activity:', error));
        }
    </script>

<script>
        // Countdown Timer
        let timeLeft = 60;
        const timer = setInterval(function() {
            if (timeLeft <= 0) {
                clearInterval(timer);
                document.getElementById("activity-form").submit();
            }
            document.getElementById("countdown").textContent = timeLeft + "s";
            timeLeft -= 1;
        }, 1000);

        // Fetch Activity Content from PHP
        $(document).ready(function() {
            $.get("get_activity.php", function(data) {
                $("#activity-section").html(data);
            });

            // Update Progress Bar based on score
            $.get("get_progress.php", function(score) {
                $("#progress").css("width", score + "%");
            });
        });
    </script>
</body>
</html>
