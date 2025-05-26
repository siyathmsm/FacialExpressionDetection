document.getElementById('loginForm').addEventListener('submit', function(event) {
    event.preventDefault();

    var formData = new FormData(this);

    fetch('lec_login.php', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        showMessage(data.message, data.status === 'success');
    })
    .catch(error => console.error('Error:', error));
});

function showMessage(message, isSuccess) {
    var messageTab = document.getElementById('messageTab');
    var messageImage = document.getElementById('messageImage');
    var messageHeading = document.getElementById('messageHeading');
    var messageText = document.getElementById('messageText');
    var messageButton = document.getElementById('messageButton');

    messageText.textContent = message;
    messageTab.classList.remove('hidden');

    if (isSuccess) {
        messageHeading.textContent = "Success";
        messageImage.src = "images/success.png"; // Add a path to your success image
        messageButton.className = "success";
    } else {
        messageHeading.textContent = "Error";
        messageImage.src = "images/error.png"; // Add a path to your error image
        messageButton.className = "error";
    }

    messageButton.onclick = function() {
        messageTab.classList.add('hidden');
        if (isSuccess) {
            window.location.href = 'create_session.html'; // Redirect to another page on success
        }
    };
}
