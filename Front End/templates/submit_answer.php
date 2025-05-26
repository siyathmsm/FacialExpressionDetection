<?php
// Database connection setup
$servername = "localhost";
$username = "root";
$password = "";
$dbname = "code_warriors";

$conn = new mysqli($servername, $username, $password, $dbname);
if ($conn->connect_error) {
    die("Connection failed: " . $conn->connect_error);
}

// Fetch submitted data
$sessionId = $_POST['sessionId'] ?? 1; // Dynamic session ID
$userEmail = $conn->real_escape_string($_POST['userEmail'] ?? '');
$submittedAnswers = [
    'q1' => $_POST['q1'] ?? '',
    'q2' => $_POST['q2'] ?? '',
    'q3' => $_POST['q3'] ?? ''
];

// Check if user exists
$userQuery = "SELECT firstname, score FROM register_student WHERE email = ?";
$stmt = $conn->prepare($userQuery);
$stmt->bind_param("s", $userEmail);
$stmt->execute();
$userResult = $stmt->get_result();

if ($userResult->num_rows === 0) {
    die("User not found.");
}
$userRow = $userResult->fetch_assoc();
$userName = $userRow['firstname'];
$existingScore = intval($userRow['score']);

// Fetch correct answers
$correctAnswersQuery = "SELECT q1correctanswer, q2correctanswer, q3correctanswer FROM mcq";
$stmt = $conn->prepare($correctAnswersQuery);
//$stmt->bind_param("i", $sessionId);
$stmt->execute();
$correctAnswersResult = $stmt->get_result();

if ($correctAnswersResult->num_rows === 0) {
    die("No correct answers found for this session.");
}
$correctAnswers = $correctAnswersResult->fetch_assoc();

// Compare submitted answers with correct answers
$correctCount = 0;
foreach ($submittedAnswers as $key => $answer) {
    $correctKey = $key . "correctanswer"; // Dynamically match correct answer key (e.g., q1correctanswer)
    if (!empty($answer) && isset($correctAnswers[$correctKey]) && $answer === $correctAnswers[$correctKey]) {
        $correctCount++;
    }
}

// Calculate new score and attendance
$newScore = $existingScore + $correctCount;
$attendance = $correctCount > 0 ? 'Yes' : 'No';

// Save submitted answers as JSON
$submittedAnswersJSON = $conn->real_escape_string(json_encode($submittedAnswers));

// Insert data into `score` table
$insertScoreQuery = "
    INSERT INTO score (name, email, previous_score, new_score, submitted_answers, sessionid, correct_answers_count, attendance)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
";
$stmt = $conn->prepare($insertScoreQuery);
$stmt->bind_param("ssiissis", $userName, $userEmail, $existingScore, $newScore, $submittedAnswersJSON, $sessionId, $correctCount, $attendance);
if (!$stmt->execute()) {
    die("Error inserting data into score table: " . $stmt->error);
}

// Update score in `register_student`
$updateScoreQuery = "UPDATE register_student SET score = ? WHERE email = ?";
$stmt = $conn->prepare($updateScoreQuery);
$stmt->bind_param("is", $newScore, $userEmail);
if (!$stmt->execute()) {
    die("Error updating register_student table: " . $stmt->error);
}

// Close connection and output details
$stmt->close();
$conn->close();

header("Location: success_answer.html");
exit;
?>