<?php
$servername = "localhost";
$username = "root";
$password = "";
$dbname = "code_warriors";

// Create connection
$conn = new mysqli($servername, $username, $password, $dbname);

// Check connection
if ($conn->connect_error) {
    die("Connection failed: " . $conn->connect_error);
}

// Check if form is submitted
if ($_SERVER["REQUEST_METHOD"] == "POST") {
    // Get form data
    $sessionid = $conn->real_escape_string($_POST['sessionid']);
    $passcode = $conn->real_escape_string($_POST['passcode']);
    $coursecode = $conn->real_escape_string($_POST['coursecode']);
    $topic = $conn->real_escape_string($_POST['topic']);
    $starttime = $conn->real_escape_string($_POST['starttime']);
    $endtime = $conn->real_escape_string($_POST['endtime']);
    $duration = $conn->real_escape_string($_POST['duration']);
    $date = $conn->real_escape_string($_POST['date']);

    // Insert data into database
    $sql = "INSERT INTO create_session (sessionid,passcode, coursecode, topic,starttime, endtime, duration, date) VALUES ('$sessionid', '$passcode', '$coursecode', '$topic', '$starttime' , '$endtime', '$duration', '$date')";

    if ($conn->query($sql) === TRUE) {
        // Success: Data inserted successfully
        echo json_encode(['status' => 'success', 'message' => 'Session created successfully!']);
    } else {
        // Failure: Insertion failed
        echo json_encode(['status' => 'failure', 'message' => 'Error: ' . $conn->error]);
    }

    // Close connection
    $conn->close();
}
?>

