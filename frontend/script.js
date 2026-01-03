document.addEventListener("DOMContentLoaded", () => {

    // =====================
    // API Configuration
    // =====================
    const API_URL = 'http://localhost:8000';  // TODO: Change to deployed URL in production

    // =====================
    // State management
    // =====================
    let audioFileData = null;
    let recordedAudio = null;

    // =====================
    // DOM elements
    // =====================
    const uploadZone = document.getElementById('uploadZone');
    const audioFileInput = document.getElementById('audioFile');
    const fileName = document.getElementById('fileName');
    const micButton = document.getElementById('micButton');
    const micStatus = document.getElementById('micStatus');
    const processButton = document.getElementById('processButton');
    const loading = document.getElementById('loading');
    const resultsSection = document.getElementById('resultsSection');
    const transcriptionResult = document.getElementById('transcriptionResult');
    const audioResult = document.getElementById('audioResult');

    // =====================
    // Navigation
    // =====================
    const navLinks = document.querySelectorAll('.nav-link');
    const sections = document.querySelectorAll('.page-section');

    navLinks.forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();

            const targetId = link.getAttribute('href').substring(1);

            // Update nav link state
            navLinks.forEach(l => l.classList.remove('active'));
            link.classList.add('active');

            // Toggle sections
            sections.forEach(section => {
                section.classList.toggle('active', section.id === targetId);
            });

            window.scrollTo({ top: 0, behavior: 'smooth' });
        });
    });

    // =====================
    // Upload handling
    // =====================
    uploadZone.addEventListener('click', () => audioFileInput.click());

    audioFileInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) {
            audioFileData = file;
            recordedAudio = null;
            fileName.textContent = `✓ ${file.name}`;
            processButton.disabled = false;
        }
    });

    uploadZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadZone.classList.add('drag-over');
    });

    uploadZone.addEventListener('dragleave', () => {
        uploadZone.classList.remove('drag-over');
    });

    uploadZone.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadZone.classList.remove('drag-over');

        const file = e.dataTransfer.files[0];
        if (file && file.type.startsWith('audio/')) {
            audioFileData = file;
            recordedAudio = null;
            fileName.textContent = `✓ ${file.name}`;
            processButton.disabled = false;
        }
    });

    // =====================
    // Mic recording
    // =====================
    micButton.addEventListener('click', () => {
        if (window.AudioRecorder) {
            window.AudioRecorder.toggleRecording();
        }
    });

    window.onRecordingComplete = async (audioBlob) => {
        try {
            // Convert webm/ogg blob to WAV using Web Audio API
            const arrayBuffer = await audioBlob.arrayBuffer();
            const audioContext = new (window.AudioContext || window.webkitAudioContext)();
            const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
            
            // Convert to WAV
            const wavBlob = await audioBufferToWav(audioBuffer);
            
            recordedAudio = new File([wavBlob], 'recording.wav', { type: 'audio/wav' });
            audioFileData = null;
            fileName.textContent = '✓ Audio recorded successfully';
            processButton.disabled = false;
        } catch (error) {
            console.error('Error processing recorded audio:', error);
            alert('Error processing recording. Please try again.');
        }
    };

    // =====================
    // Processing
    // =====================
    processButton.addEventListener('click', async () => {
        const audioData = audioFileData || recordedAudio;
        if (!audioData) return;

        const formData = new FormData();
        formData.append('file', audioData);
        formData.append('language', 'en');

        loading.classList.add('active');
        resultsSection.classList.remove('active');
        processButton.disabled = true;

        try {
            // Call your ClearSpeech backend API
            const response = await fetch(`${API_URL}/process`, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || 'Processing failed');
            }

            const data = await response.json();

            console.log('Backend response:', data);

            // Display transcript
            transcriptionResult.textContent = data.transcript;
            transcriptionResult.parentElement.style.display = 'block';

            // Set up enhanced audio player
            const audioElement = audioResult.querySelector('audio');
            audioElement.src = `${API_URL}${data.enhanced_audio_url}`;
            audioResult.style.display = 'block';

            // Setup download button
            document.getElementById('downloadButton').onclick = () => {
                const a = document.createElement('a');
                a.href = `${API_URL}${data.enhanced_audio_url}`;
                a.download = 'enhanced_audio.wav';
                a.click();
            };

            resultsSection.classList.add('active');

        } catch (err) {
            console.error('Processing error:', err);
            alert('Error processing audio: ' + err.message + '\n\nPlease make sure the backend server is running.');
        } finally {
            loading.classList.remove('active');
            processButton.disabled = false;
        }
    });

    // =====================
    // Helper Functions for WAV Conversion
    // =====================
    
    function audioBufferToWav(audioBuffer) {
        const numberOfChannels = audioBuffer.numberOfChannels;
        const sampleRate = audioBuffer.sampleRate;
        const format = 1; // PCM
        const bitDepth = 16;
        
        const bytesPerSample = bitDepth / 8;
        const blockAlign = numberOfChannels * bytesPerSample;
        
        const data = [];
        for (let i = 0; i < audioBuffer.numberOfChannels; i++) {
            data.push(audioBuffer.getChannelData(i));
        }
        
        const interleaved = interleave(data);
        const dataLength = interleaved.length * bytesPerSample;
        const buffer = new ArrayBuffer(44 + dataLength);
        const view = new DataView(buffer);
        
        // Write WAV header
        writeString(view, 0, 'RIFF');
        view.setUint32(4, 36 + dataLength, true);
        writeString(view, 8, 'WAVE');
        writeString(view, 12, 'fmt ');
        view.setUint32(16, 16, true);
        view.setUint16(20, format, true);
        view.setUint16(22, numberOfChannels, true);
        view.setUint32(24, sampleRate, true);
        view.setUint32(28, sampleRate * blockAlign, true);
        view.setUint16(32, blockAlign, true);
        view.setUint16(34, bitDepth, true);
        writeString(view, 36, 'data');
        view.setUint32(40, dataLength, true);
        
        // Write audio data
        floatTo16BitPCM(view, 44, interleaved);
        
        return new Blob([buffer], { type: 'audio/wav' });
    }

    function interleave(channelData) {
        const length = channelData[0].length;
        const numberOfChannels = channelData.length;
        const result = new Float32Array(length * numberOfChannels);
        
        let offset = 0;
        for (let i = 0; i < length; i++) {
            for (let channel = 0; channel < numberOfChannels; channel++) {
                result[offset++] = channelData[channel][i];
            }
        }
        return result;
    }

    function writeString(view, offset, string) {
        for (let i = 0; i < string.length; i++) {
            view.setUint8(offset + i, string.charCodeAt(i));
        }
    }

    function floatTo16BitPCM(view, offset, input) {
        for (let i = 0; i < input.length; i++, offset += 2) {
            const s = Math.max(-1, Math.min(1, input[i]));
            view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
        }
    }
});