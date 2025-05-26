<?php
// Database connection setup
$servername = "localhost";
$username = "root";
$password = "";
$dbname = "code_warriors";

$conn = new mysqli($servername, $username, $password, $dbname);

// Check connection
if ($conn->connect_error) {
    die("Connection failed: " . $conn->connect_error);
}

// Set headers to download as CSV
header('Content-Type: text/csv');
header('Content-Disposition: attachment; filename="attendance_sheet.csv"');

// Query to fetch attendance data where score is 10
$query = "SELECT * FROM score WHERE attendance = 'Yes'";
$result = $conn->query($query);

// Check if any records were found
if ($result->num_rows > 0) {
    // Open output buffer for writing Excel data
    $output = fopen('php://output', 'w');

    // Write Excel column headers
    fputcsv($output, ['Name', 'Email', 'Session ID', 'Correct Answers Count' , 'Previous Score' , 'New Score']);

    // Write each row of data to the Excel file
    while ($row = $result->fetch_assoc()) {
        fputcsv($output, [
            $row['name'],
            $row['email'],
            $row['sessionid'],
            $row['correct_answers_count'],
            $row['previous_score'],
            $row['new_score']
        ]);
    }

    // Close output buffer
    fclose($output);
} else {
    echo "No records found.";
}

// Close database connection
$conn->close();
?>