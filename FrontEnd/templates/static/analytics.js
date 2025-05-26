document.addEventListener("DOMContentLoaded", () => {
    const localVideo = document.getElementById('localVideo');
    const muteButton = document.getElementById('muteButton');
    const videoButton = document.getElementById('videoButton');
    const shareScreenButton = document.getElementById('shareScreenButton');
    const participantsButton = document.getElementById('participantsButton');
    const chatButton = document.getElementById('chatButton');
    const participantsList = document.getElementById('participantsList');
    const chatArea = document.getElementById('chatArea');
    const chatInput = document.getElementById('chatInput');
    const sendMessageButton = document.getElementById('sendMessageButton');
    const messages = document.getElementById('messages');
    const participants = document.getElementById('participants');
    const leaveButton = document.getElementById('leaveButton');
    const activityButton = document.getElementById("activityButton");
    const mcqOption = document.getElementById("mcqOption");
    const fillOption = document.getElementById("fillOption");
    const shortAnswerOption = document.getElementById("shortAnswerOption");
    const randomOption = document.getElementById("randomOption");
    
    let localStream;
    let remoteStream;
    let isMuted = false;
    let isVideoEnabled = true;
    let isScreenSharing = false;
    let originalStream;
    
    const videoConstraints = {
        video: {
            width: { ideal: 1280 },
            height: { ideal: 720 },
            frameRate: { ideal: 30 }
        },
        audio: true
    };
    // Get local media stream
    navigator.mediaDevices.getUserMedia(videoConstraints)
    .then(stream => {
        localStream = stream;
        originalStream = stream;
        localVideo.srcObject = stream;
    })
    .catch(error => {
        console.error('Error accessing media devices.', error);
    });
    
    // Mute/Unmute Audio
    muteButton.addEventListener('click', () => {
    isMuted = !isMuted;
    toggleAudioTracks(localStream, !isMuted);
    muteButton.textContent = isMuted ? 'Unmute' : 'Mute';
    });
    
    // Enable/Disable Video
    videoButton.addEventListener('click', () => {
    isVideoEnabled = !isVideoEnabled;
    localStream.getVideoTracks()[0].enabled = !isVideoEnabled;
    videoButton.textContent = isVideoEnabled ? 'Disable Video' : 'Enable Video';
    });
     
    // Share Screen
    shareScreenButton.addEventListener('click', () => {
        if (isScreenSharing) {
            stopScreenShare();
        } else {
            startScreenShare();
        }
    });
    
    function startScreenShare() {
        navigator.mediaDevices.getDisplayMedia({ video: true })
            .then(stream => {
                isScreenSharing = true;
                const audioTrack = originalStream.getAudioTracks()[0];
                stream.addTrack(audioTrack);
                localStream = stream;
                localVideo.srcObject = stream;
                shareScreenButton.textContent = 'Stop Sharing';
                stream.getVideoTracks()[0].addEventListener('ended', () => {
                    stopScreenShare();
                });
            })
            .catch(error => {
                console.error('Error accessing display media.', error);
            });
    }
    
    function stopScreenShare() {
        isScreenSharing = false;
        localStream.getTracks().forEach(track => track.stop());
        localStream = originalStream;
        localVideo.srcObject = originalStream;
        shareScreenButton.textContent = 'Share Screen';
        toggleAudioTracks(originalStream, !isMuted); // Ensure audio tracks are in the correct state
    }
    
    function toggleAudioTracks(stream, enabled) {
        stream.getAudioTracks().forEach(track => {
            track.enabled = enabled;
        });
    }
    
    // Toggle Participants List
    participantsButton.addEventListener('click', () => {
        participantsList.classList.toggle('active');
    });
    
    // Toggle Chat Area
    chatButton.addEventListener('click', () => {
        chatArea.classList.toggle('active');
    });
    
    // Send Chat Message
    sendMessageButton.addEventListener('click', () => {
        const message = chatInput.value;
        if (message.trim() !== '') {
            const messageElement = document.createElement('div');
            messageElement.textContent = message;
            messages.appendChild(messageElement);
            chatInput.value = '';
        }
    });
    
    // Fetch participants from the backend
    function fetchParticipants() {
        fetch('http://127.0.0.1:3000/api/participants')
            .then(response => response.json())
            .then(data => {
                participants.innerHTML = '';
                data.forEach(participant => {
                    const participantElement = document.createElement('li');
                    participantElement.textContent = participant.name;
                    participants.appendChild(participantElement);
                });
            })
            .catch(error => {
                console.error('Error fetching participants:', error);
            });
    }
    
    // Initial fetch of participants
    fetchParticipants();
    
    /*// Add participant (for demo purposes)
    function addParticipant(name) {
        const participantElement = document.createElement('li');
        participantElement.textContent = name;
        participants.appendChild(participantElement);
    }
    
    // Demo participants
    addParticipant('User 1');
    addParticipant('User 2');*/
    
    
    // Leave Meeting
    leaveButton.addEventListener('click', () => {
        if (localStream) {
            localStream.getTracks().forEach(track => track.stop());
        }
        if (remoteStream) {
            remoteStream.getTracks().forEach(track => track.stop());
        }
        window.location.href = 'https://example.com/goodbye';  
    });
    
    // Activity Drop-up
    mcqOption.addEventListener("click", () => {
        saveActivitySelection("MCQ");
    });
    
    fillOption.addEventListener("click", () => {
        saveActivitySelection("Fill in blank");
    });
    
    shortAnswerOption.addEventListener("click", () => {
        saveActivitySelection("Short Answers");
    });
    
    randomOption.addEventListener("click", () => {
        saveActivitySelection("Random Quection");
    });
    
    function saveActivitySelection(activityType) {
        // Replace with your actual API endpoint and payload structure
        const apiEndpoint = "https://example.com/api/saveActivity";
        const payload = { activityType: activityType };
    
        fetch(apiEndpoint, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify(payload),
        })
        .then(response => response.json())
        .then(data => {
            console.log("Activity saved:", data);
            // Redirect to the corresponding activity page
            switch (activityType) {
                case "MCQ":
                    window.location.href = "https://example.com/mcq";
                    break;
                case "Fill in blank":
                    window.location.href = "https://example.com/fill";
                    break;
                case "Short Answers":
                    window.location.href = "https://example.com/short-answers";
                    break;
                case "Random Quection":
                    window.location.href = "https://example.com/random-quection";
                    break;
            }
        })
        .catch(error => {
            console.error("Error saving activity:", error);
        });
    }
    
    
    
    /*const activityOptions = document.querySelectorAll('.activityOption');
                activityOptions.forEach(option => {
                    option.addEventListener('click', () => {
                        const selectedActivity = option.getAttribute('data-activity');
                        fetch('/api/activity', {
                            method: 'POST',
                            headers: {
                                'Content-Type': 'application/json'
                            },
                            body: JSON.stringify({ activity: selectedActivity })
                        })
                        .then(response => response.json())
                        .then(data => {
                            console.log('Activity selected:', data);
                        })
                        .catch(error => {
                            console.error('Error:', error);
                        });
                    });
                });
           */
    
    // Sidebar Resizer
    const resizer = document.querySelector('.resizer');
    const sidebar = document.querySelector('.sidebar');
    const container = document.querySelector('.container');
    
    let isResizing = false;
    
    resizer.addEventListener('mousedown', (e) => {
        isResizing = true;
        document.addEventListener('mousemove', resizeSidebar);
        document.addEventListener('mouseup', stopResize);
    });
    
    function resizeSidebar(e) {
        if (isResizing) {
            const newWidth = container.offsetWidth - e.clientX;
            sidebar.style.width = newWidth + 'px';
        }
    }
    
    function stopResize() {
        isResizing = false;
        document.removeEventListener('mousemove', resizeSidebar);
        document.removeEventListener('mouseup', stopResize);
    };
    
    
    });