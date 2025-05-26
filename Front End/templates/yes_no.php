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
    $questiontype = $conn->real_escape_string($_POST['questiontype']);

    $q1 = $conn->real_escape_string($_POST['q1']);
    $a1 = $conn->real_escape_string($_POST['a1']);

    $q2 = $conn->real_escape_string($_POST['q2']);
    $a2 = $conn->real_escape_string($_POST['a2']);

    $q3 = $conn->real_escape_string($_POST['q3']);
    $a3 = $conn->real_escape_string($_POST['a3']);

    $q4 = $conn->real_escape_string($_POST['q4']);
    $a4 = $conn->real_escape_string($_POST['a4']);

    $q5 = $conn->real_escape_string($_POST['q5']);
    $a5 = $conn->real_escape_string($_POST['a5']);
    
    // Insert data into database
    $sql = "INSERT INTO yesno (questiontype, q1, a1, q2, a2, q3, a3, q4, a4, q5, a5) VALUES ('$questiontype', '$q1', '$a1', '$q2' , '$a2', '$q3', '$a3', '$q4', '$a4', '$q5' , '$a5')";

    if ($conn->query($sql) === TRUE) {
        // Success: Data inserted successfully
        echo json_encode(['status' => 'success', 'message' => 'Yes-No questions added successfully!']);
    } else {
        // Failure: Insertion failed
        echo json_encode(['status' => 'failure', 'message' => 'Error: ' . $conn->error]);
    }

    // Close connection
    $conn->close();
}
?>