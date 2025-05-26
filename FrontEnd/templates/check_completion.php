<?php
session_start();
header('Content-Type: application/json');

$servername = "localhost";
$username = "root";
$password = "";
$dbname = "code_warriors";

// Create connection
$conn = new mysqli($servername, $username, $password, $dbname);

// Check connection
if ($conn->connect_error) {
    die("Connection failed: " . $conn->connect_error);
}

// Retrieve student and session details
$studentId = $_SESSION['student_id'];
$sessionId = $_GET['session_id'] ?? ''; // Session ID passed as a GET parameter

// Function to check if all activities for the session are completed by the student
function checkAllActivitiesCompleted($conn, $studentId, $sessionId) {
    // Get the total number of activities for the session
    $query = "SELECT COUNT(*) AS total_activities FROM activities WHERE session_id = ?";
    $stmt = $conn->prepare($query);
    $stmt->bind_param("i", $sessionId);
    $stmt->execute();
    $result = $stmt->get_result();
    $totalActivities = $result->fetch_assoc()['total_activities'];
    $stmt->close();

    // Get the count of completed activities by the student in this session
    $query = "SELECT COUNT(*) AS completed_activities 
              FROM answers 
              WHERE student_id = ? AND session_id = ?";
    $stmt = $conn->prepare($query);
    $stmt->bind_param("si", $studentId, $sessionId);
    $stmt->execute();
    $result = $stmt->get_result();
    $completedActivities = $result->fetch_assoc()['completed_activities'];
    $stmt->close();

    // Return true if the student has completed all activities, otherwise false
    return $completedActivities == $totalActivities && $totalActivities > 0;
}

$allCompleted = checkAllActivitiesCompleted($conn, $studentId, $sessionId);

// If all activities are completed, mark the student as "attended"
if ($allCompleted) {
    $attendanceQuery = "INSERT INTO attendance (student_id, session_id, status)
                        VALUES (?, ?, 'attended')
                        ON DUPLICATE KEY UPDATE status = 'attended'";
    $stmt = $conn->prepare($attendanceQuery);
    $stmt->bind_param("si", $studentId, $sessionId);
    $stmt->execute();
    $stmt->close();
}

echo json_encode(['allCompleted' => $allCompleted]);

$conn->close();
?>
