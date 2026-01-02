document.addEventListener("DOMContentLoaded", () => {

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
    const outputType = document.getElementById('outputType');
    const processButton = document.getElementById('processButton');
    const loading = document.getElementById('loading');
    const resultsSection = document.getElementById('resultsSection');
    const transcriptionResult = document.getElementById('transcriptionResult');
    const audioResult = document.getElementById('audioResult');

    // =====================
    // Navigation (FIXED)
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

    window.onRecordingComplete = (audioBlob) => {
        recordedAudio = audioBlob;
        audioFileData = null;
        fileName.textContent = '✓ Audio recorded successfully';
        processButton.disabled = false;
    };

    // =====================
    // Processing
    // =====================
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
        } catch (err) {
            alert('Error processing audio: ' + err.message);
        } finally {
            loading.classList.remove('active');
            processButton.disabled = false;
        }
    });
});