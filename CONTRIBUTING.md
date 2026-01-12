# Contributing to ClearSpeech

Thank you for your interest in contributing to ClearSpeech! This document provides guidelines and information for contributors.

## 🎯 Project Overview

ClearSpeech is an AI-powered speech enhancement system that uses deep learning to remove background noise from audio and provide accurate transcriptions. This is a science fair project created by middle school students.

## 📋 Table of Contents

- Code of Conduct
- How Can I Contribute?
- Development Setup
- Pull Request Process
- Coding Standards
- Testing
- Documentation

## 📜 Code of Conduct

Please read and follow our [Code of Conduct](CODE_OF_CONDUCT.md) to keep our community welcoming and respectful.

## 🤝 How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check existing issues to avoid duplicates. When creating a bug report, include:

- **Clear title and description**
- **Steps to reproduce** the issue
- **Expected behavior** vs. actual behavior
- **Screenshots** if applicable
- **Environment details** (OS, Python version, etc.)

### Suggesting Enhancements

Enhancement suggestions are welcome! Please include:

- **Clear title and description**
- **Use case** - why is this enhancement useful?
- **Possible implementation** - if you have ideas
- **Alternatives considered**

### Code Contributions

We welcome code contributions! Here are some areas where you can help:

#### 🎓 Beginner-Friendly
- Documentation improvements
- Adding code comments
- Fixing typos
- Adding examples

#### 🔧 Intermediate
- Bug fixes
- Performance improvements
- Adding tests
- UI/UX improvements

#### 🚀 Advanced
- Model improvements
- New features (batch processing, real-time audio)
- Infrastructure upgrades
- Optimizations

## 💻 Development Setup

### Prerequisites

- Python 3.10+
- Git
- 8GB+ RAM recommended

### Local Setup

```bash
# 1. Fork the repository on GitHub

# 2. Clone your fork
git clone https://github.com/YOUR_USERNAME/ClearSpeech.git
cd ClearSpeech

# 3. Add upstream remote
git remote add upstream https://github.com/ORIGINAL_OWNER/ClearSpeech.git

# 4. Create virtual environment
python3.10 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 5. Install dependencies
pip install -r requirements.txt

# 6. Test the setup
python backend/app.py
```

## 🔄 Pull Request Process

### Before Submitting

1. **Create a branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following our coding standards

3. **Test your changes**

4. **Commit with clear messages**:
   ```bash
   git commit -m "feat: add batch processing support"
   ```

   Use conventional commits:
   - `feat:` - New feature
   - `fix:` - Bug fix
   - `docs:` - Documentation
   - `style:` - Formatting
   - `refactor:` - Code refactoring
   - `test:` - Adding tests

5. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

### Submitting the PR

1. Go to the original repository
2. Click "New Pull Request"
3. Select your fork and branch
4. Fill out the PR template
5. Request review

## 📝 Coding Standards

### Python

```python
# Use clear function names and docstrings
def process_audio(audio_file: str) -> dict:
    """
    Process audio file through enhancement pipeline.
    
    Args:
        audio_file: Path to input audio file
    
    Returns:
        Dictionary with enhanced audio and transcript
    """
    pass

# Type hints
def enhance_audio(audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
    pass
```

### JavaScript

```javascript
// Use modern ES6+ syntax
const processAudio = async (file) => {
    const result = await fetch(API_URL, { ... });
    return result.json();
};

// Clear variable names
const enhancedAudio = await processFile(audioFile);
```

## 🧪 Testing

Write tests for new features:

```python
def test_audio_loading():
    """Test that audio files load correctly."""
    processor = AudioProcessor()
    audio = processor.load_audio("tests/data/sample.wav")
    assert audio is not None
```

## 📚 Documentation

- Add docstrings to functions
- Update README if needed
- Add examples for new features

## 🏆 Recognition

Contributors will be:

- Listed in CONTRIBUTORS.md
- Mentioned in release notes
- Acknowledged in README

## 💬 Questions?

- Open a Discussion on GitHub
- Email: [info@clearspeech.app](mailto:info@clearspeech.app)

## 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

## 🎓 For Students

This is a student project! We especially welcome contributions from students interested in ML, audio processing, and web development.

Don't be shy - everyone starts somewhere! 🌟

---

Thank you for contributing to ClearSpeech! 🎙️✨