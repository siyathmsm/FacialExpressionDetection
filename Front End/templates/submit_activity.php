<?php
// Database connection
$mysqli = new mysqli("localhost", "root", "", "code_warriors");

if ($mysqli->connect_error) {
    die("Connection failed: " . $mysqli->connect_error);
}

session_start();
$student_id = $_SESSION['student_id']; 
$session_id = $_POST['sessionId'];

// Retrieve activity data
$activity_key = $_POST['activityKey'];
$answer = $_POST['answer'];

// Insert answer for the specific activity
$sql = "INSERT INTO answers (student_id, session_id, activity_key, answer) 
        VALUES (?, ?, ?, ?) ON DUPLICATE KEY UPDATE answer = ?";
$stmt = $mysqli->prepare($sql);
$stmt->bind_param("sisss", $student_id, $session_id, $activity_key, $answer, $answer);
$stmt->execute();
$stmt->close();

// Check if all activities for the session are completed by the student
$sql = "SELECT COUNT(*) AS completed_activities 
        FROM answers WHERE student_id = ? AND session_id = ?";
$stmt = $mysqli->prepare($sql);
$stmt->bind_param("si", $student_id, $session_id);
$stmt->execute();
$result = $stmt->get_result();
$completed_activities = $result->fetch_assoc()['completed_activities'];
$stmt->close();

$sql = "SELECT COUNT(*) AS total_activities FROM activities WHERE session_id = ?";
$stmt = $mysqli->prepare($sql);
$stmt->bind_param("i", $session_id);
$stmt->execute();
$result = $stmt->get_result();
$total_activities = $result->fetch_assoc()['total_activities'];
$stmt->close();

// Mark attendance as "attended" if all activities are completed
if ($completed_activities == $total_activities) {
    $attendance_sql = "INSERT INTO attendance (student_id, session_id, status)
                       VALUES (?, ?, 'attended') ON DUPLICATE KEY UPDATE status = 'attended'";
    $stmt = $mysqli->prepare($attendance_sql);
    $stmt->bind_param("si", $student_id, $session_id);
    $stmt->execute();
    $stmt->close();

    echo json_encode(['success' => true, 'allCompleted' => true]);
} else {
    echo json_encode(['success' => true, 'allCompleted' => false]);
}

$mysqli->close();
?>
