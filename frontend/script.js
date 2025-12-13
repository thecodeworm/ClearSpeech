// State management
let audioFileData = null;
let recordedAudio = null;

// DOM elements
const uploadZone = document.getElementById('uploadZone');
const audioFileInput = document.getElementById('audioFile');
const fileName = document.getElementById('fileName');
const micButton = document.getElementById('micButton');
const micStatus = document.getElementById('micStatus');
const outputType = document.getElementById('outputType');
const processButton = document.getElementById('processButton');
const loading = document.getElementById('loading');
const resultsSection = document.getElementById('resultsSection');
const transcriptionResult = document.getElementById('transcriptionResult');
const audioResult = document.getElementById('audioResult');

// Navigation
const navLinks = document.querySelectorAll('.nav-link');
const sections = document.querySelectorAll('.page-section');

navLinks.forEach(link => {
    link.addEventListener('click', (e) => {
        e.preventDefault();
        const targetId = link.getAttribute('href').substring(1);
        
        // Update active nav link
        navLinks.forEach(l => l.classList.remove('active'));
        link.classList.add('active');
        
        // Show target section and hide others
        sections.forEach(section => {
            section.classList.toggle('active', section.id === targetId);
        });

        // Smooth scroll to the top of the page
        window.scrollTo({ top: 0, behavior: 'smooth' });
    });
});

// Upload zone click
uploadZone.addEventListener('click', () => {
    audioFileInput.click();
});

// File selection
audioFileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) {
        audioFileData = file;
        fileName.textContent = `✓ ${file.name}`;
        processButton.disabled = false;
        recordedAudio = null;
    }
});

// Drag and drop
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
        fileName.textContent = `✓ ${file.name}`;
        processButton.disabled = false;
        recordedAudio = null;
    }
});

// Microphone recording - handled by recorder.js
micButton.addEventListener('click', () => {
    if (window.AudioRecorder) {
        window.AudioRecorder.toggleRecording();
    }
});

// Callback for when recording completes
window.onRecordingComplete = (audioBlob) => {
    recordedAudio = audioBlob;
    fileName.textContent = '✓ Audio recorded successfully';
    processButton.disabled = false;
    audioFileData = null;
};

// Process audio
processButton.addEventListener('click', async () => {
    const audioData = audioFileData || recordedAudio;
    if (!audioData) return;

    const formData = new FormData();
    formData.append('audio', audioData);
    formData.append('output_type', outputType.value);

    loading.classList.add('active');
    resultsSection.classList.remove('active');
    processButton.disabled = true;

    try {
        // Replace with your actual API endpoint
        const response = await fetch('/api/process', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) throw new Error('Processing failed');

        if (outputType.value === 'text') {
            const data = await response.json();
            transcriptionResult.textContent = data.transcription;
            transcriptionResult.parentElement.style.display = 'block';
            audioResult.style.display = 'none';
        } else {
            const blob = await response.blob();
            const url = URL.createObjectURL(blob);
            const audio = audioResult.querySelector('audio');
            audio.src = url;
            transcriptionResult.parentElement.style.display = 'none';
            audioResult.style.display = 'block';
            
            document.getElementById('downloadButton').onclick = () => {
                const a = document.createElement('a');
                a.href = url;
                a.download = 'processed_audio.wav';
                a.click();
            };
        }

        resultsSection.classList.add('active');
    } catch (error) {
        alert('Error processing audio: ' + error.message);
    } finally {
        loading.classList.remove('active');
        processButton.disabled = false;
    }
});