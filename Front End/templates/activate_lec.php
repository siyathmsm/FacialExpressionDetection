<?php
$conn = new mysqli('localhost', 'root', '', 'code_warriors');

if ($conn->connect_error) {
    die("Connection failed: " . $conn->connect_error);
}

if (isset($_GET['token'])) {
    $token = $conn->real_escape_string($_GET['token']);
    
    // Debug: Output the token received
    //echo "Token received: $token <br>";

    $sql = "SELECT * FROM register_lecturer WHERE activation_token='$token'";
    $result = $conn->query($sql);

    if ($result->num_rows > 0) {
        $row = $result->fetch_assoc();
        
        // Debug: Output the fetched row for verification
        //echo "Fetched row: " . print_r($row, true) . " <br>";

        $sql_update = "UPDATE register_lecturer SET is_activated=1, activation_token=NULL WHERE activation_token='$token'";
        if ($conn->query($sql_update) === TRUE) {
            // Redirect to a success page with JavaScript animation
            header('Location: success_page.html');
            exit;
        } else {
            echo "Error updating record: " . $conn->error;
        }
    } else {
        echo "Invalid activation link.";
    }
} else {
    echo "No activation token provided.";
}

$conn->close();
?>