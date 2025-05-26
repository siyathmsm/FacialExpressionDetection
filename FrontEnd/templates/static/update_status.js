function showSessionDetails(sessionId, status) {
    if (status === "active") {
        alert("This session is already active.");
        window.location.href = `http://127.0.0.1:5000/analytics?sessionId=${sessionId}`;
        return;
    }

    // Send a request to update the session status to active
    fetch('update_status.php', {
        method: 'POST',
        headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
        body: new URLSearchParams({ sessionId }) // Send session ID
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            alert("Session started successfully.");
            window.location.href = `http://127.0.0.1:5000/analytics?sessionId=${sessionId}`;
        } else {
            alert("Failed to start the session. Please try again.");
        }
    })
    .catch(error => {
        console.error('Error updating session status:', error);
        alert("An error occurred while starting the session.");
    });
}