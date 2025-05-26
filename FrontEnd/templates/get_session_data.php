<?php
session_start(); // Start the session
header("Content-Type: application/json");
// Allow requests from your specific origin
header("Access-Control-Allow-Origin: http://127.0.0.1:5000");
header("Access-Control-Allow-Methods: GET, POST, OPTIONS");
header("Access-Control-Allow-Headers: Content-Type");
header("Access-Control-Allow-Credentials: true"); 

// Check if session variables exist
if (isset($_SESSION['email'], $_SESSION['firstname'], $_SESSION['score'])) {
    $data = [
        "firstname" => $_SESSION['firstname'],
        "email" => $_SESSION['email'],
        "score" => $_SESSION['score']
    ];
    
    echo json_encode($data);
} else {
    echo json_encode(["error" => "No session data found."]);
}
?>