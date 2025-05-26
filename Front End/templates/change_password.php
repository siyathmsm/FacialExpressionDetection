<?php
session_start();

if ($_SERVER["REQUEST_METHOD"] == "POST") {
    $email = $_POST['email'];
    $current_password = $_POST['current_password'];
    $new_password = $_POST['new_password'];

    $host = 'localhost';
    $db = 'code_warriors';
    $user = 'root';
    $pass = '';

    try {
        $pdo = new PDO("mysql:host=$host;dbname=$db", $user, $pass);
        $pdo->setAttribute(PDO::ATTR_ERRMODE, PDO::ERRMODE_EXCEPTION);

        // Select the current password from the database
        $stmt = $pdo->prepare("SELECT p FROM register_lecturer WHERE email = ?");
        $stmt->execute([$email]);
        $row = $stmt->fetch(PDO::FETCH_ASSOC);

        // Verify the current password and update to the new password if valid
        if ($row && $current_password === $row['password']) {
            $stmt_update = $pdo->prepare("UPDATE register_lecturer SET p = ? WHERE email = ?");
            $stmt_update->execute([$new_password, $email]);

            echo "<script>alert('Password updated successfully'); window.location.href = 'lec_login.html';</script>";
        } else {
            echo "<script>alert('Current password is incorrect or email not found'); window.location.href = 'change_password.html';</script>";
        }
    } catch (PDOException $e) {
        die("Error: " . $e->getMessage());
    }
}
?>