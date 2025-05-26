<?php
$conn = new mysqli('localhost', 'root', '', 'code_warriors');

if ($conn->connect_error) {
    die("Connection failed: " . $conn->connect_error);
}

if ($_SERVER["REQUEST_METHOD"] == "GET") {
    if (isset($_GET['token'])) {
        $token = $conn->real_escape_string($_GET['token']);
        $sql = "SELECT * FROM register_student WHERE reset_token='$token'";
        $result = $conn->query($sql);

        if ($result && $result->num_rows > 0) {
            // Display the reset password form
            echo '
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Reset Password</title>
                <style>
                    body {
                        font-family: Arial, sans-serif;
                        background-color: #f0f2f5;
                        display: flex;
                        justify-content: center;
                        align-items: center;
                        height: 100vh;
                        margin: 0;
                    }
                    .reset-container {
                        background-color: #fff;
                        padding: 20px;
                        border-radius: 8px;
                        box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
                        width: 100%;
                        max-width: 400px;
                    }
                    .reset-container h2 {
                        margin-bottom: 20px;
                        font-size: 24px;
                        text-align: center;
                    }
                    .reset-container input[type="password"] {
                        width: 100%;
                        padding: 10px;
                        margin: 10px 0;
                        border: 1px solid #ccc;
                        border-radius: 4px;
                    }
                    .reset-container button {
                        width: 100%;
                        padding: 10px;
                        background-color: #003C43;
                        color: #fff;
                        border: none;
                        border-radius: 4px;
                        cursor: pointer;
                        font-size: 16px;
                    }
                    .reset-container button:hover {
                        background-color: transparent;
                        border: 1px solid #003C43;
                        color: #003C43;
                        transition: 0.3s ease;
                    }
                </style>
            </head>
            <body>
                <div class="reset-container">
                    <h2>Reset Password</h2>
                    <form action="reset_password_student.php" method="POST">
                        <input type="hidden" name="token" value="' . htmlspecialchars($token) . '">
                        <input type="password" name="password" placeholder="New Password" required>
                        <button type="submit">Reset Password</button>
                    </form>
                </div>
            </body>
            </html>';
        } else {
            echo "Invalid reset link.";
        }
    } else {
        echo "Token is missing.";
    }
} elseif ($_SERVER["REQUEST_METHOD"] == "POST") {
    if (isset($_POST['token']) && isset($_POST['password'])) {
        $token = $conn->real_escape_string($_POST['token']);
        $new_password = $conn->real_escape_string($_POST['password']);

        $sql = "UPDATE register_student SET p='$new_password', reset_token=NULL WHERE reset_token='$token'";
        if ($conn->query($sql) === TRUE) {
            header('Location: success_password_reset.html');
            exit;
        } else {
            echo "Error: " . $conn->error;
        }
    } else {
        echo "Token or password is missing.";
        if (!isset($_POST['token'])) {
            echo " Missing token.";
        }
        if (!isset($_POST['password'])) {
            echo " Missing password.";
        }
    }
} else {
    echo "Invalid request.";
}
?>