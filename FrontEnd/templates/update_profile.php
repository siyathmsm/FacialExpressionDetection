<?php  //this is for student
session_start();

// Database connection
$host = 'localhost';
$db = 'code_warriors';
$user = 'root';
$pass = '';
$conn = new mysqli($host, $user, $pass, $db);

if ($conn->connect_error) {
    die("Connection failed: " . $conn->connect_error);
}

// Check if user is logged in
if (!isset($_SESSION['email'])) {
    echo "<script>alert('Please log in first.'); window.location.href='student_login.html';</script>";
    exit();
}

// Get the logged-in user's email
$email = $_SESSION['email'];

// Get form data
$firstname = $_POST['firstname'];
$lastname = $_POST['lastname'];

// Prepare to update user data
$query = $conn->prepare("UPDATE register_student SET firstname = ?, lastname = ? WHERE email = ?");
$query->bind_param("sss", $firstname, $lastname, $email);

// Check if a new profile picture has been uploaded
if (isset($_FILES['profile_picture']) && $_FILES['profile_picture']['error'] === UPLOAD_ERR_OK) {
    $fileTmpPath = $_FILES['profile_picture']['tmp_name'];
    $fileName = $_FILES['profile_picture']['name'];
    $fileSize = $_FILES['profile_picture']['size'];
    $fileType = $_FILES['profile_picture']['type'];
    $fileNameCmps = explode(".", $fileName);
    $fileExtension = strtolower(end($fileNameCmps));

    // Validate file extension
    $allowedfileExtensions = array('jpg', 'png', 'jpeg');
    if (in_array($fileExtension, $allowedfileExtensions)) {
        // Define the upload path
        $uploadFileDir = './uploads/';
        $dest_path = $uploadFileDir . 'profile_' . $email . '.' . $fileExtension;

        // Move the uploaded file to the server
        if (move_uploaded_file($fileTmpPath, $dest_path)) {
            // Update the profile picture path in the database
            $queryUpdatePic = $conn->prepare("UPDATE register_student SET profile_picture = ? WHERE email = ?");
            $queryUpdatePic->bind_param("ss", $dest_path, $email);
            $queryUpdatePic->execute();
        }
    } else {
        echo "<script>alert('Upload failed. Invalid file type.'); window.location.href='edit_profile.php';</script>";
        exit();
    }
}

// Execute update for other details
if ($query->execute()) {
    echo "<script>alert('Profile updated successfully!'); window.location.href='join_session.html';</script>";
} else {
    echo "<script>alert('Profile update failed. Please try again.'); window.location.href='edit_profile.php';</script>";
}

$conn->close();
?>
