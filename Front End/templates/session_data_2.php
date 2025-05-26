<?php
session_start();

// Check if session variables exist
if (isset($_SESSION['email'], $_SESSION['firstname'], $_SESSION['score'])) {
    echo json_encode([
        "email" => $_SESSION['email'],
        "firstname" => $_SESSION['firstname'],
        "score" => $_SESSION['score']
    ]);
} else {
    echo json_encode(["error" => "No session data found."]);
}
?>