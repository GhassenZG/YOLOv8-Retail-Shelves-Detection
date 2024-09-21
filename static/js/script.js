document.getElementById('webcamButton').addEventListener('click', function() {
    fetch('/start_webcam')
        .then(response => {
            if (response.ok) {
                document.getElementById('video').src = '/video_feed';
                document.getElementById('video').style.display = 'block';
                alert("Webcam stream started successfully!");
            } else {
                alert("Failed to start webcam stream. Please try again later.");
            }
        })
        .catch(err => {
            console.error("Error starting webcam stream: ", err);
            alert("An error occurred while starting webcam stream. Please try again later.");
        });
});
