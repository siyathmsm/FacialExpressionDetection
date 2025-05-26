<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Password Reset Request</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background-color: #f0f0f0;
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
            margin: 0;
        }

        .container {
            background-color: #fff;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
            text-align: center;
        }

        form {
            width: 300px;
            margin: 0 auto;
        }

        input[type="email"], button {
            width: 100%;
            padding: 10px;
            margin-bottom: 10px;
            box-sizing: border-box;
            border: 1px solid #ccc;
            border-radius: 4px;
            font-size: 16px;
        }

        button {
            background-color: #003C43;
            color: white;
            border: none;
            cursor: pointer;
        }

        button:hover {
            background-color: transparent;
            border: 1px solid #003C43;
            color: #003C43;
            transition: 0.3s ease;
        }

        label {
            font-weight: bold;
        }
    </style>
</head>

<body>
    <div class="container">
        <h2>Password Reset Request</h2>
        <form action="request_reset_student.php" method="POST">
            <label for="email">Enter your email:</label><br>
            <input type="email" id="email" name="email" placeholder="Your email address" required><br><br>
            <button type="submit">Reset Password</button>
        </form>
    </div>
</body>
</html>

<?php
require 'vendor/autoload.php'; // PHPMailer
require 'vendor/class.phpmailer.php';
require 'vendor/class.smtp.php';

// Database connection
$conn = new mysqli('localhost', 'root', '', 'code_warriors');

if ($_SERVER["REQUEST_METHOD"] == "POST") {
    $email = $conn->real_escape_string($_POST['email']);

    // Generate unique reset token
    $token = bin2hex(random_bytes(16));

    // Update user's reset token in database
    $sql = "UPDATE register_student SET reset_token = '$token' WHERE email = '$email'";
    if ($conn->query($sql) === TRUE) {
        // Send password reset email
        $mail = new PHPMailer(true);
        try {
            $mail->isSMTP();
            $mail->Host = 'smtp.gmail.com'; // Set your SMTP server
            $mail->SMTPAuth = true;
            $mail->Username = 'siyathmsm2000@gmail.com';
            $mail->Password = 'oacs vyhy cius ndyj';
            $mail->SMTPSecure = 'tls';
            $mail->Port = 587;

            $mail->setFrom('siyathmsm2000@gmail.com', 'Code Warriors');
            $mail->addAddress($email);

            $mail->isHTML(true);
            $mail->Subject = 'Password Reset';
            
            // Replace localhost with your actual domain and path to reset password form
            $reset_link = "http://127.0.0.1/CodeWarriors/templates/reset_password_student.php?token=$token";
            
            $mail->Body = "Click the link to reset your password: <a href='$reset_link'>Reset Password</a>";

            $mail->send();
            echo '<script>alert("Password reset link sent! Please check your email.");</script>';
        } catch (Exception $e) {
            echo '<script>alert("Message could not be sent. Mailer Error: ' . $mail->ErrorInfo . '");</script>';
        }
    } else {
        echo '<script>alert("Error: ' . $conn->error . '");</script>';
    }
}
?>