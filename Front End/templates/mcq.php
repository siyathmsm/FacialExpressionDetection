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
    $q1 = $conn->real_escape_string($_POST['q1']);
    $q1a1 = $conn->real_escape_string($_POST['q1a1']);
    $q1a2 = $conn->real_escape_string($_POST['q1a2']);
    $q1a3 = $conn->real_escape_string($_POST['q1a3']);
    $q1a4 = $conn->real_escape_string($_POST['q1a4']);
    $q1correctanswer = $conn->real_escape_string($_POST['q1correctanswer']);

    $q2 = $conn->real_escape_string($_POST['q2']);
    $q2a1 = $conn->real_escape_string($_POST['q2a1']);
    $q2a2 = $conn->real_escape_string($_POST['q2a2']);
    $q2a3 = $conn->real_escape_string($_POST['q2a3']);
    $q2a4 = $conn->real_escape_string($_POST['q2a4']);
    $q2correctanswer = $conn->real_escape_string($_POST['q2correctanswer']);

    $q3 = $conn->real_escape_string($_POST['q3']);
    $q3a1 = $conn->real_escape_string($_POST['q3a1']);
    $q3a2 = $conn->real_escape_string($_POST['q3a2']);
    $q3a3 = $conn->real_escape_string($_POST['q3a3']);
    $q3a4 = $conn->real_escape_string($_POST['q3a4']);
    $q3correctanswer = $conn->real_escape_string($_POST['q3correctanswer']);

    /* $q4 = $conn->real_escape_string($_POST['q4']);
    $q4a1 = $conn->real_escape_string($_POST['q4a1']);
    $q4a2 = $conn->real_escape_string($_POST['q4a2']);
    $q4a3 = $conn->real_escape_string($_POST['q4a3']);
    $q4a4 = $conn->real_escape_string($_POST['q4a4']);
    $q4correctanswer = $conn->real_escape_string($_POST['q4correctanswer']);

    $q5 = $conn->real_escape_string($_POST['q5']);
    $q5a1 = $conn->real_escape_string($_POST['q5a1']);
    $q5a2 = $conn->real_escape_string($_POST['q5a2']);
    $q5a3 = $conn->real_escape_string($_POST['q5a3']);
    $q5a4 = $conn->real_escape_string($_POST['q5a4']);
    $q5correctanswer = $conn->real_escape_string($_POST['q5correctanswer']); */

    // Insert data into database
    $sql = "INSERT INTO mcq (q1, q1a1, q1a2, q1a3, q1a4, q1correctanswer, q2, q2a1, q2a2, q2a3, q2a4, q2correctanswer, q3, q3a1, q3a2, q3a3, q3a4, q3correctanswer) VALUES ('$q1', '$q1a1', '$q1a2', '$q1a3', '$q1a4', '$q1correctanswer', '$q2', '$q2a1', '$q2a2', '$q2a3', '$q2a4', '$q2correctanswer', '$q3', '$q3a1', '$q3a2', '$q3a3', '$q3a4', '$q3correctanswer')";

    if ($conn->query($sql) === TRUE) {
        // Success: Data inserted successfully
        echo json_encode(['status' => 'success', 'message' => 'MCQ questions added successfully!']);
        // Redirect to success page
        header('Location: success_page_activities_update.html');
        exit();
    } else {
        // Failure: Insertion failed
        echo json_encode(['status' => 'failure', 'message' => 'Error: ' . $conn->error]);
    }

    // Close connection
    $conn->close();
}
?>