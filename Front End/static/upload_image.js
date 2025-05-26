document.addEventListener('DOMContentLoaded', () => {
    const video = document.getElementById('localVideo');
    const canvas = document.getElementById('canvas');
    const context = canvas.getContext('2d');

    // Access the webcam
    navigator.mediaDevices.getUserMedia({ video: true })
      .then(stream => {
        video.srcObject = stream;
        captureImage();
      })
      .catch(err => console.error('Error accessing webcam:', err));

    function captureImage() {
      setInterval(() => {
        context.drawImage(video, 0, 0, canvas.width, canvas.height);
        const imageData = canvas.toDataURL('image/png');
        sendImageToServer(imageData);
      }, 60000); // Capture image every 5 minutes (300000 ms)
    }

    function sendImageToServer(imageData) {
      fetch('upload_image.php', {
        method: 'POST',
        body: JSON.stringify({ image: imageData }),
        headers: {
          'Content-Type': 'application/json'
        }
      })
      .then(response => response.json())
      .then(data => {
        console.log('Server response:', data);
      })
      .catch(error => console.error('Error sending image to server:', error));
    }
  });