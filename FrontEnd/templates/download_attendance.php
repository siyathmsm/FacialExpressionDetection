<?php
// download_attendance.php

// Database connection setup
$servername = "localhost";
$username = "root";
$password = "";
$dbname = "code_warriors";

$conn = new mysqli($servername, $username, $password, $dbname);

// Check connection
if ($conn->connect_error) {
    die("Connection failed: " . $conn->connect_error);
}

// Set headers to download as CSV
header('Content-Type: text/csv');
header('Content-Disposition: attachment; filename="attendance_sheet.csv"');

// Retrieve session ID from GET parameters
$sessionId = $_GET['session_id'] ?? null;

if (!$sessionId) {
    echo "Session ID not provided.";
    exit;
}

// Query to fetch attendance data for the specified session
$query = $conn->prepare("SELECT student_id, status, completion_details FROM attendance WHERE session_id = ?");
$query->bind_param("i", $sessionId);
$query->execute();
$result = $query->get_result();

// Open output stream for CSV download
$output = fopen('php://output', 'w');

// Write CSV headers
fputcsv($output, ['Session_ID', 'Student ID', 'Attendance Status', 'Completion Details']);

// Write attendance data
while ($row = $result->fetch_assoc()) {
    fputcsv($output, [
        $row['session_id'],
        $row['student_id'],
        $row['status'] === 'attended' ? 'Yes' : 'No', 
        $row['completion_details']
    ]);
}

// Close database connection and output stream
$query->close();
$conn->close();
fclose($output);
?>
