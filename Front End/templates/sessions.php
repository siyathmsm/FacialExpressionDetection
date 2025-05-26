<?php
// Database connection details
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

$sql = "SELECT sessionid,passcode, coursecode, topic, starttime, endtime, duration, date, status FROM create_session";
$result = $conn->query($sql);

$sessions = array();
if ($result->num_rows > 0) {
  // Output data of each row
  while($row = $result->fetch_assoc()) {
    $sessions[] = $row;
  }
} else {
  echo "0 results";
}
$conn->close();

echo json_encode($sessions);
?>
