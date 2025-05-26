<?php //this is for lecturer
session_start(); // Start the session
//$profilePicture = isset($_SESSION['profile_picture']) ? $_SESSION['profile_picture'] : 'path/to/default/image.jpg';

// Check if user is logged in
/*if (!isset($_SESSION['logged_in']) || $_SESSION['logged_in'] !== true) {
    // Redirect to the login page if not logged in
    header("Location: lec_login.php");
    exit();
}*/


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

// Prepare SQL statement to retrieve user data for the first lecturer in the table
$query = $conn->prepare("SELECT firstname, lastname, email, phoneno, profile_picture FROM register_student LIMIT 1");
$query->execute();
$result = $query->get_result();

// Check if any lecturer exists
if ($result->num_rows == 0) {
    echo "<script>alert('No lecturers found, please register.'); window.location.href='register.html';</script>";
    exit();
}

// Fetch user data
$userData = $result->fetch_assoc();
?>

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Edit Profile</title>
    <link rel="stylesheet" href="../static/edit_profile.css">
    <script>

        function previewImage(event) {
    const reader = new FileReader();
    reader.onload = function() {
        const output = document.getElementById('profilePicturePreview');
        output.src = reader.result;
        // Trigger the custom event to update the profile icon
        window.dispatchEvent(new CustomEvent('updateProfilePicture', {
            detail: { newProfilePicture: reader.result }
        }));
        console.log("Preview image loaded: ", reader.result); // Debug log
    };
    reader.readAsDataURL(event.target.files[0]);
}

    </script>
</head>
<body>

<div class="container profile-edit-container">
    <h1>Edit Profile</h1>
    
    
    <img id="profilePicturePreview" src="<?php echo htmlspecialchars($userData['profile_picture']) . '?v=' . time(); ?>" alt="Profile Picture Preview" style="width: 150px; height: 150px; border-radius: 50%;"/>


    <form action="edit_profile_add.php" method="POST" enctype="multipart/form-data">
        <label for="firstname">First Name:</label>
        <input type="text" id="firstname" name="firstname" value="<?php echo htmlspecialchars($userData['firstname']); ?>" required>

        <label for="lastname">Last Name:</label>
        <input type="text" id="lastname" name="lastname" value="<?php echo htmlspecialchars($userData['lastname']); ?>" required>

        <label for="email">Email:</label>
        <input type="email" id="email" name="email" value="<?php echo htmlspecialchars($userData['email']); ?>" required readonly>

        <label for="phone_no">Phone Number:</label>
        <input type="text" id="phone_no" name="phone_no" value="<?php echo htmlspecialchars($userData['phoneno']); ?>" required>

        <label for="profile_picture">Profile Picture:</label>
        <input type="file" id="profile_picture" name="profile_picture" accept="image/*" onchange="previewImage(event)">

        <input type="submit" value="Update Profile">
    </form>
</div>

<script>
    window.addEventListener('updateProfilePicture', function(event) {
    var newProfilePicture = event.detail.newProfilePicture;
    console.log("New Profile Picture URL: ", newProfilePicture); // Debug log
    document.getElementById('profileIcon').src = newProfilePicture + '?v=' + new Date().getTime(); // Cache busting
});

</script>

</body>
</html>

