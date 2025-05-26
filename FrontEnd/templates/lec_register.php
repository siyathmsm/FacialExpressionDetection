<?php
require 'vendor/autoload.php'; // PHPMailer
require 'vendor/class.phpmailer.php';
require 'vendor/class.smtp.php';

// Database connection parameters
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
    $firstname = $conn->real_escape_string($_POST['firstname']);
    $lastname = $conn->real_escape_string($_POST['lastname']);
    $username = $conn->real_escape_string($_POST['username']);
    $email = $conn->real_escape_string($_POST['email']);
    $phoneno = $conn->real_escape_string($_POST['phoneno']);
    $password = $conn->real_escape_string($_POST['password']);
    $p = $conn->real_escape_string($_POST['password']);

    // Generate activation token
    $token = bin2hex(random_bytes(16));

    // Insert data into database
    $hashedPassword = password_hash($password,PASSWORD_BCRYPT);
    $sql = "INSERT INTO register_lecturer (firstname, lastname, username, email, phoneno, password, activation_token, p) VALUES ('$firstname', '$lastname', '$username', '$email', '$phoneno', '$hashedPassword', '$token', '$p')";

    if ($conn->query($sql) === TRUE) {
        // Send activation email using PHPMailer
        $mail = new PHPMailer(true);
        try {
            // SMTP configuration
            $mail->isSMTP();
            $mail->Host = 'smtp.gmail.com'; // Set your SMTP server
            $mail->SMTPAuth = true;
            $mail->Username = 'siyathmsm2000@gmail.com'; // Your Gmail address
            $mail->Password = 'oacs vyhy cius ndyj'; // Your Gmail password
            $mail->SMTPSecure = 'tls';
            $mail->Port = 587;

            // Sender and recipient
            $mail->setFrom('siyathmsm2000@gmail.com', 'Code Warriors');
            $mail->addAddress($email, $firstname); // Lecturer's email and name

            // Email content
            $mail->isHTML(true);
            $mail->Subject = 'Lecturer Account Activation';
            $activation_link = "http://127.0.0.1/CodeWarriors/templates/activate_lec.php?token=$token"; // Replace with your activation script URL
            $mail->Body = "Dear $firstname, click the link to activate your account: <a href='$activation_link'>Activate Account</a>";

            // Send email
            $mail->send();

            // Success message
            echo json_encode(['status' => 'success', 'message' => 'Lecturer account registered successfully! Activation email sent.']);
        } catch (Exception $e) {
            // Error message
            echo json_encode(['status' => 'failure', 'message' => "Message could not be sent. Mailer Error: {$mail->ErrorInfo}"]);
        }
    } else {
        // Error message if insertion fails
        echo json_encode(['status' => 'failure', 'message' => 'Error: ' . $conn->error]);
    }

    // Close connection
    $conn->close();
}
?>