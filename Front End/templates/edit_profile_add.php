<?php //this is for lecturer
// Start the session if it's not already started
session_start();

// Database connection
$host = 'localhost';
$db = 'code_warriors';
$user = 'root';
$pass = '';
$conn = new mysqli($host, $user, $pass, $db);

// Check connection
if ($conn->connect_error) {
    die("Connection failed: " . $conn->connect_error);
}

// Get form data
$firstname = $_POST['firstname'];
$lastname = $_POST['lastname'];
$phoneno = $_POST['phone_no'];
$email = $_POST['email']; // Use the email from the form

// Prepare to update user data
$query = $conn->prepare("UPDATE register_lecturer SET firstname = ?, lastname = ?, phoneno = ? WHERE email = ?");
$query->bind_param("ssss", $firstname, $lastname, $phoneno, $email);

// Initialize variable for profile picture path
$dest_path = ""; // Initialize dest_path to avoid undefined variable issues

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
        // Upload path
        $uploadFileDir = './uploads/';
        $dest_path = $uploadFileDir . 'profile_' . $email . '.' . $fileExtension;

        // Move the uploaded file
        if (move_uploaded_file($fileTmpPath, $dest_path)) {
            // Update the profile picture path in the database
            $queryUpdatePic = $conn->prepare("UPDATE register_lecturer SET profile_picture = ? WHERE email = ?");
            $queryUpdatePic->bind_param("ss", $dest_path, $email);
            if ($queryUpdatePic->execute()) {
                echo "Profile picture updated in database: " . $dest_path; // Debug log
            } else {
                echo "Error updating profile picture in database."; // Debug log
            }
        } else {
            echo "Error moving uploaded file."; // Debug log
        }
    }
}

// Execute update for other details
if ($query->execute()) {
    // Update the session with the new profile picture path if a new one was uploaded
    if (!empty($dest_path)) {
        $_SESSION['profile_picture'] = $dest_path; // Update session with new profile picture path
    }
    $_SESSION['firstname'] = $firstname; // Store first name in session
    $_SESSION['lastname'] = $lastname;   // Store last name in session
    
    echo "<script>alert('Profile updated successfully!'); window.location.href='join_session.html';</script>";
} else {
    echo "<script>alert('Profile update failed. Please try again.'); window.location.href='edit_profile.php';</script>";
}

// Close the database connection
$conn->close();
?>
