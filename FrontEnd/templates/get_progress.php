<?php
include 'db_connection.php';

session_start();
$user_id = $_SESSION['user_id'];

$score_query = "SELECT SUM(score) as total_score FROM score WHERE user_id = $user_id";
$score_result = $conn->query($score_query);
$total_score = $score_result->fetch_assoc()['total_score'] ?? 0;

// Assuming a max score of 100 for simplicity
$progress_percentage = min(100, $total_score);
echo $progress_percentage;
?>
