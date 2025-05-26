<?php
header('Content-Type: application/json');

// Database connection details
$servername = "localhost";
$username = "root";
$password = "";
$dbname = "code_warriors";

// Create connection
$conn = new mysqli($servername, $username, $password, $dbname);

// Check connection
if ($conn->connect_error) {
    die(json_encode(["error" => "Connection failed: " . $conn->connect_error]));
}

// Get the question type from the URL
$question_type = isset($_GET['type']) ? $_GET['type'] : 'type1';

// Fetch questions for the specified type from the database
$sql = "SELECT q1, q2, q3, q4, q5 FROM owa WHERE questiontype = ?";
$stmt = $conn->prepare($sql);
if ($stmt) {
    $stmt->bind_param("s", $question_type);
    $stmt->execute();
    $result = $stmt->get_result();
    $questions = $result->fetch_assoc();
    $stmt->close();
} else {
    die(json_encode(["error" => "Database query failed: " . $conn->error]));
}

$conn->close();

echo json_encode($questions);
?>