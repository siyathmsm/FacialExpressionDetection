<?php
header('Content-Type: application/json');

// Include database connection
// Database connection
$servername = "localhost";
$username = "root";
$password = "";
$dbname = "code_warriors";

$conn = new mysqli($servername, $username, $password, $dbname);

if ($conn->connect_error) {
    die("Connection failed: " . $conn->connect_error);
}

if ($_SERVER['REQUEST_METHOD'] === 'POST') {
    $sessionId = $_POST['sessionId'] ?? null;
    $status = $_POST['status'] ?? null;

    if ($sessionId && $status) {
        // Prepare the SQL query
        $query = "UPDATE create_session SET status = ? WHERE sessionid = ?";
        $stmt = $conn->prepare($query);

        if ($stmt) {
            $stmt->bind_param("ss", $status, $sessionId); // Bind the status and sessionId
            if ($stmt->execute()) {
                if ($stmt->affected_rows > 0) {
                    echo json_encode(['success' => true]);
                } else {
                    echo json_encode(['success' => false, 'error' => 'No rows updated. Check sessionId']);
                }
            } else {
                echo json_encode(['success' => false, 'error' => $stmt->error]);
            }
            $stmt->close();
        } else {
            echo json_encode(['success' => false, 'error' => 'Failed to prepare query']);
        }
    } else {
        echo json_encode(['success' => false, 'error' => 'Invalid sessionId or status']);
    }
} else {
    echo json_encode(['success' => false, 'error' => 'Invalid request method']);
}

$conn->close();
?>
