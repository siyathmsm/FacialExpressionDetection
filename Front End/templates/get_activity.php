<?php
include 'db_connection.php';

session_start();
$activity_type = isset($_GET['type']) ? $_GET['type'] : 'mcq';
$activity = null;

if ($activity_type === 'owa') {
    $query = "SELECT * FROM owa ORDER BY RAND() LIMIT 1";
} elseif ($activity_type === 'mcq') {
    $query = "SELECT * FROM mcq ORDER BY RAND() LIMIT 1";
} else {
    $query = "SELECT * FROM yesno ORDER BY RAND() LIMIT 1";
}

$result = $conn->query($query);
if ($result->num_rows > 0) {
    $activity = $result->fetch_assoc();
}

if ($activity_type === 'mcq') {
    echo "<h2>" . htmlspecialchars($activity['question']) . "</h2>";
    echo "<form id='activity-form' action='submit_answer.php' method='POST'>";
    echo "<input type='hidden' name='activity_id' value='" . $activity['id'] . "'>";
    echo "<input type='hidden' name='activity_type' value='mcq'>";
    echo "<label><input type='radio' name='answer' value='A'> " . htmlspecialchars($activity['option_a']) . "</label><br>";
    echo "<label><input type='radio' name='answer' value='B'> " . htmlspecialchars($activity['option_b']) . "</label><br>";
    echo "<label><input type='radio' name='answer' value='C'> " . htmlspecialchars($activity['option_c']) . "</label><br>";
    echo "<label><input type='radio' name='answer' value='D'> " . htmlspecialchars($activity['option_d']) . "</label><br>";
    echo "<button type='submit'>Submit</button>";
    echo "</form>";
} elseif ($activity_type === 'owa') {
    echo "<h2>" . htmlspecialchars($activity['question']) . "</h2>";
    echo "<form id='activity-form' action='submit_answer.php' method='POST'>";
    echo "<input type='hidden' name='activity_id' value='" . $activity['id'] . "'>";
    echo "<input type='hidden' name='activity_type' value='owa'>";
    echo "<textarea name='answer' required></textarea>";
    echo "<button type='submit'>Submit</button>";
    echo "</form>";
} elseif ($activity_type === 'yesno') {
    echo "<h2>" . htmlspecialchars($activity['question']) . "</h2>";
    echo "<form id='activity-form' action='submit_answer.php' method='POST'>";
    echo "<input type='hidden' name='activity_id' value='" . $activity['id'] . "'>";
    echo "<input type='hidden' name='activity_type' value='yesno'>";
    echo "<label><input type='radio' name='answer' value='yes'> Yes</label>";
    echo "<label><input type='radio' name='answer' value='no'> No</label>";
    echo "<button type='submit'>Submit</button>";
    echo "</form>";
}
?>
