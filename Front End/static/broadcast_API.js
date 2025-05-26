const channel = new BroadcastChannel('activity_channel');
        channel.addEventListener('message', function(event) {
            if (event.data.type === 'navigate') {
                document.getElementById('contentFrame').src = event.data.url;
            }
        });