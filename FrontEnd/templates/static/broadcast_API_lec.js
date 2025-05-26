const channel = new BroadcastChannel('activity_channel');
        document.querySelectorAll('.link').forEach(link => {
            link.addEventListener('click', function(event) {
                event.preventDefault();
                const url = this.getAttribute('data-url');
                channel.postMessage({ type: 'navigate', url: url });
            });
        });