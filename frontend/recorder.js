// Audio Recorder Module using Web Audio API
(function() {
    let mediaRecorder = null;
    let audioChunks = [];
    let isRecording = false;

    const micButton = document.getElementById('micButton');
    const micStatus = document.getElementById('micStatus');

    // Audio Recorder object
    window.AudioRecorder = {
        toggleRecording: async function() {
            if (!isRecording) {
                await this.startRecording();
            } else {
                this.stopRecording();
            }
        },

        startRecording: async function() {
            try {
                    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
                        alert(
                        "Microphone recording is not supported on this browser or requires HTTPS.\n\n" +
                        "Please use:\n" +
                        "• Chrome / Edge\n" +
                        "• Safari on iOS 14+\n" +
                        "• HTTPS connection"
                        );
                        return;
                }

                const stream = await navigator.mediaDevices.getUserMedia({ 
                    audio: {
                        echoCancellation: true,
                        noiseSuppression: true,
                        autoGainControl: true
                    } 
                });

                // Check if MediaRecorder is supported
                if (!MediaRecorder.isTypeSupported('audio/webm')) {
                    console.warn('audio/webm not supported, using default');
                }

                mediaRecorder = new MediaRecorder(stream, {
                    mimeType: MediaRecorder.isTypeSupported('audio/webm') 
                        ? 'audio/webm' 
                        : 'audio/ogg'
                });

                audioChunks = [];

                mediaRecorder.addEventListener('dataavailable', (e) => {
                    if (e.data.size > 0) {
                        audioChunks.push(e.data);
                    }
                });

                mediaRecorder.addEventListener('stop', () => {
                    const mimeType = mediaRecorder.mimeType;
                    const audioBlob = new Blob(audioChunks, { type: mimeType });
                    
                    // Call the callback function if it exists
                    if (typeof window.onRecordingComplete === 'function') {
                        window.onRecordingComplete(audioBlob);
                    }

                    // Stop all tracks
                    stream.getTracks().forEach(track => track.stop());
                });

                mediaRecorder.start();
                isRecording = true;

                // Update UI
                micButton.classList.add('recording');
                micButton.textContent = '⏹️';
                micStatus.textContent = 'Recording... Click to stop';

            } catch (error) {
                console.error('Error accessing microphone:', error);
                alert('Error accessing microphone: ' + error.message + 
                      '\n\nPlease ensure you have granted microphone permissions.');
                isRecording = false;
            }
        },

        stopRecording: function() {
            if (mediaRecorder && mediaRecorder.state !== 'inactive') {
                mediaRecorder.stop();
                isRecording = false;

                // Update UI
                micButton.classList.remove('recording');
                micButton.textContent = '🎤';
                micStatus.textContent = 'Click to start recording';
            }
        },

        isRecording: function() {
            return isRecording;
        }
    };

    // Optional: Visualize audio input
    window.AudioRecorder.visualize = async function(stream) {
        const audioContext = new (window.AudioContext || window.webkitAudioContext)();
        const analyser = audioContext.createAnalyser();
        const microphone = audioContext.createMediaStreamSource(stream);
        
        microphone.connect(analyser);
        analyser.fftSize = 256;
        
        const bufferLength = analyser.frequencyBinCount;
        const dataArray = new Uint8Array(bufferLength);
        
        // You can use dataArray to create visualizations
        // This is a basic implementation - expand as needed
        
        return {
            analyser,
            dataArray,
            bufferLength,
            audioContext
        };
    };

})();