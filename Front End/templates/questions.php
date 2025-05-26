<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Conduct Sessions</title>
    <link rel="stylesheet" href="../static/conduct_sessions.css">
</head>
<body>
    <nav class="navbar">
        <div class="notification">
            <img src="../static/images/logo.png" alt="Logo" height="50px" width="85px">
        </div>
        <ul class="nav-links">
            <li class="active"><a href="#" class="nav-link">Activity Area</a></li>
        </ul>
        <div class="nav-right">
            <div class="profile">
                <img src="../static/images/profile.png" alt="Profile">
            </div>
            <div id="logout">
                <img src="../static/images/logout.png" alt="Logout">
            </div>
        </div>
    </nav>

    <!-- PHP code to include questions.php content -->
    <?php include('question.php'); ?>

</body>
</html>