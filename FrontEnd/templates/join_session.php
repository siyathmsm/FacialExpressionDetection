<?php
session_start();

// Database connection
$servername = "localhost";
$username = "root";
$password = "";
$dbname = "code_warriors";

$conn = new mysqli($servername, $username, $password, $dbname);

if ($conn->connect_error) {
    die("Connection failed: " . $conn->connect_error);
}

if ($_SERVER["REQUEST_METHOD"] == "POST") {
    $sessionId = $_POST['sessionid'];
    $passcode = $_POST['passcode'];
    $email = $_POST['email'];
    $firstname = $_POST['firstname'];
    $score = $_POST['score'];

    // Validate session ID and passcode
    $sql = "SELECT status FROM create_session WHERE sessionid = ? AND passcode = ?";
    $stmt = $conn->prepare($sql);
    $stmt->bind_param("ss", $sessionId, $passcode);
    $stmt->execute();
    $result = $stmt->get_result();

    if ($result->num_rows > 0) {
        $row = $result->fetch_assoc();
        $status = $row['status'];

        if ($status === "active") {
            // Store data in session variables
            $_SESSION['email'] = $email;
            $_SESSION['firstname'] = $firstname;
            $_SESSION['score'] = $score;

            // Success: Session is active
            echo json_encode([
                'email' => $email,
                'firstname' => $firstname,
                'score' => $score,
                'status' => 'success',
                'message' => 'Session is active'
            ]);
        } else {
            // Failure: Session is not active
            echo json_encode([
                'status' => 'inactive',
                'message' => 'The session is not active. Please wait for the session to start.'
            ]);
        }
    } else {
        // Failure: Invalid session ID or passcode
        echo json_encode([
            'status' => 'failure',
            'message' => 'Invalid session ID or passcode!'
        ]);
    }

    $stmt->close();
    $conn->close();
}
?>