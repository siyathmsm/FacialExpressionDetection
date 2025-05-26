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
    $user = $_POST['email'];
    $pass = $_POST['password'];

    // SQL query to check username and password
    $sql = "SELECT * FROM register_lecturer WHERE email = ? AND p = ?";
    $stmt = $conn->prepare($sql);
    $stmt->bind_param("ss", $user, $pass);
    $stmt->execute();
    $result = $stmt->get_result();

    if ($result->num_rows > 0) {
        // Success: Credentials are valid
        echo json_encode(['status' => 'success', 'message' => 'Login successfully!']);
    } else {
        // Failure: Invalid credentials
        echo json_encode(['status' => 'failure', 'message' => 'Invalid email or password!']);
    }

    $stmt->close();
    $conn->close();
}
?>