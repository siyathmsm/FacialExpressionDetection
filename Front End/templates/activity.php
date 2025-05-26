<?php
// Database connection setup
$servername = "localhost";
$username = "root";
$password = "";
$dbname = "code_warriors";

$conn = new mysqli($servername, $username, $password, $dbname);
if ($conn->connect_error) {
    die("Connection failed: " . $conn->connect_error);
}

// Determine the table to fetch data from based on a GET parameter (default to 'mcq')
$table = isset($_GET['table']) ? $_GET['table'] : 'mcq';

/* // Sanitize table name to avoid SQL injection
$allowedTables = ['owa', 'mcq', 'yesno']; // Allowed table names
if (!in_array($table, $allowedTables)) {
    die("Invalid table selection");
} */

// Fetch questions from the selected table
$sql = "SELECT * FROM $table LIMIT 1"; // Safe as $table is whitelisted
$result = $conn->query($sql);

if ($result && $result->num_rows > 0) {
    $questions = $result->fetch_assoc();
} else {
    $questions = null; // No questions available
}

$conn->close();
?>

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Activity Interface</title>
    <link rel="stylesheet" href="../static/student.css">
    <link rel="stylesheet" href="../static/lec_login_2.css">
    <style>
        /* Navbar styles */
        .navbar {
            display: flex;
            justify-content: space-between;
            align-items: center;
            background-color: #77B0AA;
            padding: 10px 20px;
            position: fixed;
            top: 0;
            width: 100%;
            z-index: 1000;
        }

        /* Left section of navbar (logo + user info) */
        .nav-left {
            display: flex;
            align-items: center;
        }

        /* Logo styling */
        .navbar .logo img {
            height: 50px;
            width: 85px;
            margin-right: 20px; /* Space between logo and user info */
        }

        /* User info section */
        .user-info {
            display: flex;
            align-items: center;
            gap: 15px; /* Add spacing between items */
            font-size: 18px;
            line-height: 1.5;
        }

        .user-info span {
            position: relative;
            font-weight: bold;
            color: #333;
        }

        /* Add a dot after each span except the last one */
        .user-info span::after {
            content: "|";
            position: absolute;
            right: -10px; /* Adjust spacing between the text and the dot */
            color: #333;
            font-weight: normal;
        }

        /* Remove the dot from the last span */
        .user-info span:last-child::after {
            content: "";
        }

        .progress-bar-container {
            width: 150px;
            background-color: #ddd;
            border-radius: 10px;
            overflow: hidden;
        }

        .progress-bar {
            height: 20px;
            background-color: #4caf50;
            text-align: center;
            color: white;
            line-height: 20px;
            width: 0%;
        }

        /* Profile icon styling */
        .profile {
            position: relative;
            margin-right: 15px;
        }

        .profile img {
            width: 40px;
            height: 40px;
            cursor: pointer;
        }

        .profile-dropdown {
            display: none;
            position: absolute;
            right: 0;
            background-color: #f9f9f9;
            min-width: 164px;
            box-shadow: 0px 8px 16px 0px rgba(0, 0, 0, 0.2);
            z-index: 1;
        }

        .profile-dropdown a {
            color: black;
            padding: 12px 16px;
            text-decoration: none;
            display: block;
        }

        .profile-dropdown a:hover {
            background-color: #f1f1f1;
        }

        .profile:hover .profile-dropdown {
            display: block;
        }
    </style>
</head>
<body>
    <!-- Navbar -->
    <nav class="navbar">
        <div class="nav-left">
            <div class="logo">
                <img src="../static/images/logo.png" alt="Logo">
            </div>
            <div class="user-info">
                <span id="user-firstname">Name: </span>
                <span id="user-email">Email: </span>
                <span id="user-score">Score: </span>
                <div class="progress-bar-container">
                    <div id="progress-bar" class="progress-bar"></div>
                </div>
            </div>
        </div>
        <div class="nav-right">
            <div class="profile">
                <img id="profileIcon" src="../static/images/profile.png" alt="Profile">
                <div class="profile-dropdown">
                    <a href="edit_profilee.php">Edit Profile</a>
                    <a href="change_password.html">Change Password</a>
                    <a id="logoutBtn" href="lec_login.html">Logout</a>
                </div>
            </div>
        </div>
    </nav>

    <main>
        <div id="activity-wrapper">
            <div id="activity-header">
                <h1 id="activity-title">Lecturer's Activities</h1>
                <div id="countdown">Time left: <span id="time">2:00</span></div>
            </div>
            <div class="container">
                <div id="activity-container">
                    <h2>MCQ Form</h2>
                    <form id="mcqForm" action="submit_answer.php" method="post">
                    <input type="hidden" id="userEmail" name="userEmail">
                    <input type="hidden" id="userFirstName" name="userFirstName">
                    <input type="hidden" id="userScore" name="userScore">
                        <?php if ($questions): ?>
                            <label for="q1"><?= htmlspecialchars($questions['q1']) ?></label><br>
                            <input type="radio" name="q1" value="<?= htmlspecialchars($questions['q1a1']) ?>" onclick="updateProgress()"> <?= htmlspecialchars($questions['q1a1']) ?><br>
                            <input type="radio" name="q1" value="<?= htmlspecialchars($questions['q1a2']) ?>" onclick="updateProgress()"> <?= htmlspecialchars($questions['q1a2']) ?><br>
                            <input type="radio" name="q1" value="<?= htmlspecialchars($questions['q1a3']) ?>" onclick="updateProgress()"> <?= htmlspecialchars($questions['q1a3']) ?><br>
                            <input type="radio" name="q1" value="<?= htmlspecialchars($questions['q1a4']) ?>" onclick="updateProgress()"> <?= htmlspecialchars($questions['q1a4']) ?><br><br>

                            <label for="q2"><?= htmlspecialchars($questions['q2']) ?></label><br>
                            <input type="radio" name="q2" value="<?= htmlspecialchars($questions['q2a1']) ?>" onclick="updateProgress()"> <?= htmlspecialchars($questions['q2a1']) ?><br>
                            <input type="radio" name="q2" value="<?= htmlspecialchars($questions['q2a2']) ?>" onclick="updateProgress()"> <?= htmlspecialchars($questions['q2a2']) ?><br>
                            <input type="radio" name="q2" value="<?= htmlspecialchars($questions['q2a3']) ?>" onclick="updateProgress()"> <?= htmlspecialchars($questions['q2a3']) ?><br>
                            <input type="radio" name="q2" value="<?= htmlspecialchars($questions['q2a4']) ?>" onclick="updateProgress()"> <?= htmlspecialchars($questions['q2a4']) ?><br><br>

                            <label for="q3"><?= htmlspecialchars($questions['q3']) ?></label><br>
                            <input type="radio" name="q3" value="<?= htmlspecialchars($questions['q3a1']) ?>" onclick="updateProgress()"> <?= htmlspecialchars($questions['q3a1']) ?><br>
                            <input type="radio" name="q3" value="<?= htmlspecialchars($questions['q3a2']) ?>" onclick="updateProgress()"> <?= htmlspecialchars($questions['q3a2']) ?><br>
                            <input type="radio" name="q3" value="<?= htmlspecialchars($questions['q3a3']) ?>" onclick="updateProgress()"> <?= htmlspecialchars($questions['q3a3']) ?><br>
                            <input type="radio" name="q3" value="<?= htmlspecialchars($questions['q3a4']) ?>" onclick="updateProgress()"> <?= htmlspecialchars($questions['q3a4']) ?><br><br>

                            <!-- <label for="q4"><?= htmlspecialchars($questions['q4']) ?></label><br>
                            <input type="radio" name="q4" value="<?= htmlspecialchars($questions['q4a1']) ?>" onclick="updateProgress()"> <?= htmlspecialchars($questions['q4a1']) ?><br>
                            <input type="radio" name="q4" value="<?= htmlspecialchars($questions['q4a2']) ?>" onclick="updateProgress()"> <?= htmlspecialchars($questions['q4a2']) ?><br>
                            <input type="radio" name="q4" value="<?= htmlspecialchars($questions['q4a3']) ?>" onclick="updateProgress()"> <?= htmlspecialchars($questions['q4a3']) ?><br>
                            <input type="radio" name="q4" value="<?= htmlspecialchars($questions['q4a4']) ?>" onclick="updateProgress()"> <?= htmlspecialchars($questions['q4a4']) ?><br><br>

                            <label for="q5"><?= htmlspecialchars($questions['q5']) ?></label><br>
                            <input type="radio" name="q5" value="<?= htmlspecialchars($questions['q5a1']) ?>" onclick="updateProgress()"> <?= htmlspecialchars($questions['q5a1']) ?><br>
                            <input type="radio" name="q5" value="<?= htmlspecialchars($questions['q5a2']) ?>" onclick="updateProgress()"> <?= htmlspecialchars($questions['q5a2']) ?><br>
                            <input type="radio" name="q5" value="<?= htmlspecialchars($questions['q5a3']) ?>" onclick="updateProgress()"> <?= htmlspecialchars($questions['q5a3']) ?><br>
                            <input type="radio" name="q5" value="<?= htmlspecialchars($questions['q5a4']) ?>" onclick="updateProgress()"> <?= htmlspecialchars($questions['q5a4']) ?><br><br> -->
                            
                        <?php else: ?>
                            <p>No questions available.</p>
                        <?php endif; ?>
                        <button type="submit" id="submitButton">Submit</button>
                    </form>
                </div>
            </div>
        </div>
    </main>

    <script>
        // Countdown Timer (2:00)
        let timeLeft = 60; // 1 minutes in seconds
        const countdownElement = document.getElementById("time");

        function updateCountdown() {
            let minutes = Math.floor(timeLeft / 60);
            let seconds = timeLeft % 60;
            countdownElement.textContent = `${minutes}:${seconds < 10 ? '0' : ''}${seconds}`;
            timeLeft--;
            if (timeLeft < 0) {
                clearInterval(timerInterval);
                document.getElementById("mcqForm").submit(); // Submit the form
            }
        }

        const timerInterval = setInterval(updateCountdown, 1000);
    </script>

    <script>
        // Fetch data from session_data.php
        fetch('session_data_2.php', {
            credentials: 'include'
        })
            .then(response => {
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                return response.json(); // Assuming session_data.php returns JSON
            })
            .then(data => {
                // Update the DOM with the fetched data
                document.getElementById("user-firstname").textContent = `Name: ${data.firstname}`;
                document.getElementById("user-email").textContent = `Email: ${data.email}`;
                document.getElementById("user-score").textContent = `Score: ${data.score}`;

                document.getElementById("userEmail").value = data.email;
                document.getElementById("userFirstName").value = data.firstname;
                document.getElementById("userScore").value = data.score;
                
                // Update progress bar
                const score = data.score;
                const progressBar = document.getElementById("progress-bar");
                progressBar.style.width = `${score}%`;
                progressBar.textContent = `${score}%`;

                // Log to console for confirmation
                console.log("Data retrieved successfully:", data);
            })
            .catch(error => {
                console.error("Error fetching data:", error);
            });
    </script>
</body>
</html>