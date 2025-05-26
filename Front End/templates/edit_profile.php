<?php   //this is for student
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

// Retrieve user data
$query = $conn->prepare("SELECT firstname, lastname, email, profile_picture FROM register_student WHERE email = ?");
$query->bind_param("s", $email);
$query->execute();
$result = $query->get_result();

// Check if the user exists
if ($result->num_rows == 0) {
    echo "<script>alert('User not found, please register.'); window.location.href='register_student.html';</script>";
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
    <link rel="stylesheet" href="../static/edit_profile.css"> <!-- Your CSS file for styling -->
    <script>
        function previewImage(event) {
            const reader = new FileReader();
            reader.onload = function(){
                const output = document.getElementById('profilePicturePreview');
                output.src = reader.result;
            };
            reader.readAsDataURL(event.target.files[0]);
        }
    </script>
</head>
<body>
    <div class="container profile-edit-container">
        <h1>Edit Profile</h1>
        <img id="profilePicturePreview" src="<?php echo htmlspecialchars($userData['profile_picture']); ?>" alt="Profile Picture" style="width: 150px; height: 150px; border-radius: 50%;"/>

        <form action="update_profile.php" method="POST" enctype="multipart/form-data">
            <label for="firstname">First Name:</label>
            <input type="text" id="firstname" name="firstname" value="<?php echo htmlspecialchars($userData['firstname']); ?>" required>

            <label for="lastname">Last Name:</label>
            <input type="text" id="lastname" name="lastname" value="<?php echo htmlspecialchars($userData['lastname']); ?>" required>

            <label for="email">Email:</label>
            <input type="email" id="email" name="email" value="<?php echo htmlspecialchars($userData['email']); ?>" readonly>

            <label for="profile_picture">Profile Picture:</label>
            <input type="file" id="profile_picture" name="profile_picture" accept="image/*" onchange="previewImage(event)">

            <input type="submit" value="Update Profile">
        </form>
    </div>
</body>
</html>
