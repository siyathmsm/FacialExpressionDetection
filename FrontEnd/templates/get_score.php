<?php
header('Content-Type: application/json');


$servername = "localhost";
$username = "root";
$password = "";
$dbname = "code_warriors";


$conn = new mysqli($servername, $username, $password, $dbname);


if ($conn->connect_error) {
    echo json_encode(["status" => "error", "message" => "Connection failed: " . $conn->connect_error]);
    exit;
}


$session_id = isset($_GET['sessionid']) ? $_GET['sessionid'] : '';

if (empty($session_id)) {
    echo json_encode(["status" => "error", "message" => "Session ID is missing."]);
    $conn->close();
    exit;
}


$stmt = $conn->prepare("SELECT score FROM score WHERE sessionid = ?");
$stmt->bind_param("s", $session_id);
$stmt->execute();
$result = $stmt->get_result();

if ($result->num_rows > 0) {
    $row = $result->fetch_assoc();
    echo json_encode(["status" => "success", "score" => $row['score']]);
} else {
    echo json_encode(["status" => "error", "message" => "Score not found for the given session ID."]);
}

$stmt->close();
$conn->close();
?>
