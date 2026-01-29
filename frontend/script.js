document.addEventListener("DOMContentLoaded", () => {

    // =====================
    // API Configuration
    // =====================
   const API_URL =
  window.location.hostname === "localhost"
    ? "http://localhost:8000"
    : "https://thecodeworm-clearspeechapi.hf.space";


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
    const ttsToggle = document.getElementById('ttsToggle'); // NEW
    const audioPlayerContainer = document.getElementById('audioPlayerContainer'); // NEW

    // =====================
    // Navigation
    // =====================
    const navLinks = document.querySelectorAll('.nav-link');
    const sections = document.querySelectorAll('.page-section');

    navLinks.forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();

            const targetId = link.getAttribute('href').substring(1);

            navLinks.forEach(l => l.classList.remove('active'));
            link.classList.add('active');

            sections.forEach(section => {
                section.classList.toggle('active', section.id === targetId);
            });

            navMenu.classList.remove('active');
            window.scrollTo({ top: 0, behavior: 'smooth' });  
        });
    });

    // =====================
    // Mobile Hamburger Nav
    // =====================
    const hamburger = document.getElementById('hamburger');
    const navMenu = document.getElementById('navMenu');

    hamburger.addEventListener('click', () => {
        navMenu.classList.toggle('active');
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
            const arrayBuffer = await audioBlob.arrayBuffer();
            const audioContext = new (window.AudioContext || window.webkitAudioContext)();
            const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
            
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
        formData.append('generate_tts', ttsToggle.checked); // NEW: Send TTS preference

        loading.classList.add('active');
        resultsSection.classList.remove('active');
        processButton.disabled = true;

        try {
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

            // Clear previous audio players
            audioPlayerContainer.innerHTML = '';

            // Create enhanced audio player
            const enhancedContainer = document.createElement('div');
            enhancedContainer.className = 'audio-option';
            enhancedContainer.innerHTML = `
                <h3>Enhanced Original Audio</h3>
                <audio controls style="width: 100%; margin-bottom: 8px;">
                    <source src="${API_URL}${data.enhanced_audio_url}" type="audio/wav">
                </audio>
                <button class="download-button" onclick="downloadAudio('${API_URL}${data.enhanced_audio_url}', 'enhanced_audio.wav')">
                    Download Enhanced Audio
                </button>
            `;
            audioPlayerContainer.appendChild(enhancedContainer);

            // Create TTS audio player if available
            if (data.tts_audio_url) {
                const ttsContainer = document.createElement('div');
                ttsContainer.className = 'audio-option';
                ttsContainer.innerHTML = `
                    <h3>Text-to-Speech Version</h3>
                    <audio controls style="width: 100%; margin-bottom: 8px;">
                        <source src="${API_URL}${data.tts_audio_url}" type="audio/wav">
                    </audio>
                    <button class="download-button" onclick="downloadAudio('${API_URL}${data.tts_audio_url}', 'tts_audio.wav')">
                        Download TTS Audio
                    </button>
                `;
                audioPlayerContainer.appendChild(ttsContainer);
            }

            audioResult.style.display = 'block';
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
    // Download Helper
    // =====================
    window.downloadAudio = function(url, filename) {
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        a.click();
    };

    // =====================
    // Helper Functions for WAV Conversion
    // =====================
    
    function audioBufferToWav(audioBuffer) {
        const numberOfChannels = audioBuffer.numberOfChannels;
        const sampleRate = audioBuffer.sampleRate;
        const format = 1;
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