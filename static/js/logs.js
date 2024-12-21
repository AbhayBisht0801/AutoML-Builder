

// static/js/logs.js
document.addEventListener('DOMContentLoaded', function() {
    const logsDiv = document.getElementById('logs');
    const eventSource = new EventSource('/stream-logs');
    
    eventSource.onmessage = function(event) {
        logsDiv.innerHTML += event.data;
        logsDiv.scrollTop = logsDiv.scrollHeight;
    };
    
    eventSource.onerror = function() {
        eventSource.close();
        document.querySelector('.status').innerHTML = 'Processing complete';
    };
});