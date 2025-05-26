<?php
header('Content-Type: application/json');

// Database connection
$servername = "localhost";
$username = "root";
$password = "";
$dbname = "code_warriors";

$conn = new mysqli($servername, $username, $password, $dbname);
if ($conn->connect_error) {
    echo json_encode(['error' => 'Database connection failed: ' . $conn->connect_error]);
    exit;
}

$type = $_GET['type'] ?? 'default';
$sql = "SELECT q1, q1a1, q1a2, q1a3, q1a4 FROM mcq WHERE questiontype = ?";
$stmt = $conn->prepare($sql);
$stmt->bind_param("s", $type);
$stmt->execute();
$result = $stmt->get_result();

if ($result->num_rows === 0) {
    echo json_encode(['error' => 'No questions found for this type']);
    exit;
}

$questions = [];
while ($row = $result->fetch_assoc()) {
    $questions[] = [
        'text' => $row['q1'],
        'options' => [$row['q1a1'], $row['q1a2'], $row['q1a3'], $row['q1a4']]
    ];
}

echo json_encode(['questions' => $questions]);
$conn->close();
