<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Session Login</title>
    <link rel="stylesheet" href="../static/score.css">
    <script>
        // JavaScript code to handle form submission and display alerts
        document.addEventListener('DOMContentLoaded', function() {
            document.getElementById('loginForm').addEventListener('submit', function(event) {
                event.preventDefault();

                var formData = new FormData(this);

                fetch('join_session.php', {
                    method: 'POST',
                    body: formData
                })
                .then(response => response.json())
                .then(data => {
                    if (data.status === 'success') {
                        alert('Login successful: ' + data.message);
                        window.location.href = 'http://127.0.0.1:5000/student_interface'; // Redirect to student interface on success
                    } else {
                        alert('Login failed: ' + data.message);
                    }
                })
                .catch(error => console.error('Error:', error));
            });
        });
    </script>
</head>
<body>
    <?php
    // Database connection
    $servername = "localhost";
    $username = "root";
    $password = "";
    $dbname = "code_warriors";

    $conn = new mysqli($servername, $username, $password, $dbname);

    if ($conn->connect_error) {
        die("Connection failed: " . $conn->connect_error);
    }

    // Fetch user details from the database
    $query = "SELECT firstname, score FROM score WHERE email = 'siyathmsm2000@gmail.com'";
    $result = mysqli_query($conn, $query);

    if ($result && mysqli_num_rows($result) > 0) {
        $user = mysqli_fetch_assoc($result);
        $fname = $user['firstname'];
        $score = $user['score'];
    } else {
        $fname = 'User';
        $score = 0;
    }
    ?>

    <div class="taskbar">
        <img src="images/logo.png" alt="Logo" class="logo">
        <div class="profile">
            <img src="images/profile.png" alt="Profile Image" class="profile-img">
            <div class="profile-info">
                <span class="profile-name"><?php echo $fname; ?></span>
                <span class="profile-score">Score: <?php echo $score; ?></span>
            </div>
        </div>
    </div>
    
    <div class="main">
        <h2>Session Login</h2>
        <form id="loginForm" action="join_session.php" method="post">
            <label for="sessionid">Session ID:</label>
            <input type="text" id="sessionid" name="sessionid" required>
            <label for="passcode">Passcode:</label>
            <input type="password" id="passcode" name="passcode" required>
            <button type="submit">Login</button>
        </form>
    </div>
</body>
</html>