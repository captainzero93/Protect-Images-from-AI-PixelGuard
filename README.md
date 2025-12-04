# PixelGuard AI (AI IMAGE PROTECT)
[![ko-fi](https://ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/captainzero)
## Introduction
AI scraping involves the automated collection of images from the internet for training AI models. This practice can lead to unauthorized use of personal or copyrighted images. PixelGuard AI aims to protect your images from such scraping and other AI training (like 'deepfakes') by applying various invisible techniques that interfere with AI processing while preserving the visual quality for human viewers (as much as possible).

## Example Results

Here's an example of an earlier version of PixelGuard AI:

| Unprotected Image | Protected Image (light preset) |
|:-----------------:|:---------------:|
| ![Unprotected](https://github.com/captainzero93/Protect-Images-from-AI-PixelGuard/raw/main/efcsdzecvzdscz.png) | ![Protected](https://github.com/captainzero93/Protect-Images-from-AI-PixelGuard/raw/main/protected_efcsdzecvzdscz.png) |

# Changelog

## [2.0.0] - 2025-01-15

### 🎉 Major Release - Complete Rewrite

Version 2.0 represents a ground-up rewrite with **50+ improvements** over v1.0. This is essentially a new product while maintaining backward compatibility where possible.

### Added - Protection Techniques

#### Multi-Model Adversarial Protection ⭐
- Uses multiple neural networks (ResNet50 + VGG16) instead of single model
- Gradient accumulation from both models for robust perturbations
- Better generalization across different AI architectures
- Transferable protection against various AI systems

#### Enhanced Frequency Domain Protection ⭐⭐⭐
- **Multi-channel DCT**: Now operates on all RGB channels (was blue only)
- **Multi-level Wavelets**: 3-level decomposition (was 1 level)
- **Fourier Ring Masking**: Targets mid-frequencies specifically with ring masks
- Better preservation of visual quality while increasing robustness

#### New Spatial Techniques ⭐⭐
- **Texture-Aware Perturbations**: Gradient-based protection using Sobel operators
- **Strategic Multi-Type Noise**: Combines Gaussian + Salt-and-Pepper + Perlin-like noise
- **Perceptual Color Shifts**: LAB color space manipulation (was simple HSV)
- **High-Frequency Masking**: Targets CNN feature extraction specifically

#### New Multi-Scale Protection ⭐⭐
- Pyramid-based protection at 3 scales (100%, 50%, 25%)
- Ensures survival of downsampling common in AI pipelines
- Blended protection across resolutions

### Added - Features & Infrastructure

#### Professional CLI Interface ⭐⭐⭐
- Complete command-line interface with argparse
- Subcommands: `protect`, `verify`, `export-config`
- Comprehensive help text and examples
- Progress tracking with tqdm
- Colored output support

#### Configuration System ⭐⭐⭐
- JSON-based configuration files
- Three presets included: subtle, balanced, maximum
- Load/save/export configurations
- Override system for command-line arguments
- Reusable settings across runs

#### Batch Processing & Parallelization ⭐⭐⭐
- Multi-threaded processing with ProcessPoolExecutor
- Configurable worker count (auto-detects CPU cores)
- Progress bars with real-time updates
- Error handling per image
- Summary statistics

#### Key Management System ⭐⭐⭐
- Persistent RSA-4096 key storage (upgraded from 2048)
- Auto-generation on first run
- Secure PEM format serialization
- Keys stored in dedicated `keys/` directory
- Proper public/private key separation

#### Enhanced Verification ⭐⭐⭐
- **4 hash types**: SHA-256, perceptual hash, average hash, difference hash
- **Graduated status levels**: VERIFIED, LIKELY_AUTHENTIC, MODIFIED, TAMPERED
- Hash difference calculation
- Age tracking with days since protection
- JSON output option
- Detailed reporting

#### Documentation Suite ⭐⭐⭐
- **PROJECT_SUMMARY.md**: Complete package overview
- **QUICKSTART.md**: 5-minute setup guide
- **README.md**: Comprehensive manual (updated to match original style)
- **TECHNICAL.md**: Deep algorithm documentation
- **STRUCTURE.md**: File organization guide
- **IMPROVEMENTS.md**: All 50+ enhancements listed
- **This CHANGELOG**: Version history

#### Testing & Development ⭐⭐
- **test_protector.py**: Complete test suite
- Creates test images automatically
- Validates all features
- Dependency checking
- Installation verification

#### Helper Scripts ⭐⭐
- **batch_protect.sh**: Bash script for easy batch processing
- Auto CPU detection
- Pretty colored output
- Error handling
- Progress tracking

#### Additional Features
- **Tracking System**: UUID v4 for unique tracking IDs
- **Visible Watermark**: Optional visible watermark with custom text
- **Format Conversion**: Protect and convert between formats
- **Statistics Tracking**: Processed/failed/skipped counters
- **Logging System**: File and console logging with levels
- **Python API**: Clean import and use as module
- **Examples**: 10 complete usage examples in examples.py
- **Git Integration**: Proper .gitignore with security

### Changed - Algorithm Improvements

#### DCT Watermarking
- **Before**: Single channel (blue only)
- **After**: All three RGB channels
- **Impact**: 3x redundancy, more robust

#### Wavelet Decomposition  
- **Before**: Single level
- **After**: 3-level multi-resolution
- **Impact**: Survives downsampling better

#### Fourier Masking
- **Before**: Random frequency modification
- **After**: Ring-based mid-frequency targeting
- **Impact**: More effective AI confusion

#### Adversarial Perturbations
- **Before**: Single model (ResNet50)
- **After**: Dual model (ResNet50 + VGG16)
- **Impact**: Better generalization

#### Noise Injection
- **Before**: Simple Gaussian noise
- **After**: Three-type strategic combination
- **Impact**: Harder to denoise

#### Color Manipulation
- **Before**: HSV color jittering
- **After**: Perceptual LAB color shifts
- **Impact**: More imperceptible

#### Steganography
- **Before**: Basic LSB embedding
- **After**: Enhanced with end markers and tracking
- **Impact**: Better detection and verification

#### Cryptographic Security
- **Before**: RSA-2048, keys per session
- **After**: RSA-4096, persistent key storage
- **Impact**: Stronger security, consistent verification

### Changed - Code Architecture

#### Code Organization
- **Before**: ~300 lines, single file, minimal structure
- **After**: ~1,100 lines, modular design, clean separation
- **Impact**: Maintainable, extensible, professional

#### Error Handling
- **Before**: Minimal try-catch
- **After**: Comprehensive error management
- **Impact**: Graceful degradation, detailed errors

#### Memory Management
- **Before**: Potential leaks
- **After**: Proper cleanup, pooling
- **Impact**: Handles large batches

#### Image Processing
- **Before**: Basic handling
- **After**: Robust conversion, shape preservation
- **Impact**: Handles all formats reliably

#### Quality Preservation
- **Before**: Basic clipping
- **After**: Advanced normalization, better algorithms
- **Impact**: Better visual quality

### Changed - User Experience

#### Output Messages
- **Before**: Simple prints
- **After**: ✓/✗ indicators, colored output, clear status
- **Impact**: Professional, easy to understand

#### Progress Tracking
- **Before**: None
- **After**: Real-time progress bars, percentages
- **Impact**: Know exactly what's happening

#### Configuration
- **Before**: Hardcoded parameters
- **After**: JSON files, presets, CLI overrides
- **Impact**: Flexible, reusable

#### Documentation
- **Before**: Inline comments only
- **After**: 7 markdown files, 50+ pages
- **Impact**: Easy to learn and use

### Fixed

#### Critical Fixes
- Image shape preservation during adversarial perturbation
- Color space conversion RGB/BGR issues
- Metadata embedding failures for PNG
- Memory leaks in batch processing
- EXIF data loss for JPEG files
- Key generation randomness
- Hash verification edge cases

#### Quality Improvements
- Better clipping to prevent artifacts
- Improved normalization for all channels
- Fixed gradient computation errors
- Corrected wavelet reconstruction
- Fixed QR code opacity calculation

#### Compatibility Fixes
- Cross-platform path handling
- Font fallbacks for systems without fonts
- Python 3.8-3.12 compatibility
- Better error messages for missing dependencies

### Deprecated
- Old single-file script approach (use new CLI)
- Hardcoded protection parameters (use configs)
- Per-session keys (use persistent keys)

### Removed
- GUI dependencies (CLI only for v2.0)
- Redundant duplicate code
- Unused imports and functions

### Security
- Upgraded to RSA-4096 (from 2048)
- Added PSS padding for signatures
- Persistent secure key storage
- Never commits keys to git (.gitignore)
- Multiple verification methods

### Performance
- 2-3x faster with GPU acceleration
- 4-8x faster with parallel processing
- Better memory usage for large batches
- Optimized image operations
- Cached model loading

### Compatibility
- **Python**: 3.8+ (tested up to 3.12)
- **OS**: Windows, macOS, Linux
- **Formats**: JPG, PNG, BMP, TIFF, WebP, GIF
- **GPU**: Optional CUDA support

---

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2024-12-01

### Initial Release

#### Features
- Basic DCT watermarking (blue channel only)
- Single-level wavelet watermarking (green channel)
- Fourier watermarking (red channel)
- Adversarial perturbations (ResNet50, FGSM)
- Simple color jittering (HSV)
- Invisible QR code embedding
- Basic LSB steganography
- RSA-2048 digital signatures
- SHA-256 hash verification
- Perceptual hash (pHash)

#### Interface
- GUI with Tkinter
- Single image protection
- Batch processing (sequential)
- Basic verification

#### Protection
- 7 protection techniques
- Adjustable sliders
- 3 preset levels
- JPEG and PNG support

#### Limitations
- GUI only (no CLI)
- Sequential processing (slow)
- Keys generated per session
- Single model adversarial
- Basic error handling
- Minimal documentation

---

## Version Comparison Summary

| Feature | v1.0 | v2.0 |
|---------|------|------|
| **Protection Techniques** | 7 | 12 |
| **Adversarial Models** | 1 | 2 |
| **Hash Types** | 2 | 4 |
| **RSA Key Size** | 2048 | 4096 |
| **DCT Channels** | 1 | 3 |
| **Wavelet Levels** | 1 | 3 |
| **Interface** | GUI only | CLI + API + Scripts |
| **Configuration** | Hardcoded | JSON + Presets |
| **Batch Processing** | Sequential | Parallel |
| **Documentation** | Minimal | Comprehensive |
| **Test Suite** | None | Complete |
| **Lines of Code** | ~300 | ~1,100 |
| **Files** | 1 | 15+ |

---

## Migration Guide (v1.0 → v2.0)

### If you used v1.0:

**Before (v1.0):**
```bash
python imgprotect.py  # Opens GUI
```

**After (v2.0):**
```bash
# CLI approach
python advanced_image_protector.py protect image.jpg

# With similar settings to v1.0 "Recommended" preset
python advanced_image_protector.py protect image.jpg --config config_balanced.json
```

### Key Changes:
- No GUI in v2.0 (CLI is faster and more powerful)
- Keys now persist across runs
- Configuration via JSON files
- Much faster batch processing
- More protection techniques
- Better verification

---

## Future Roadmap

### Planned for v2.1
- Adaptive protection based on image content
- Additional model architectures (EfficientNet, ViT)
- GUI option (separate tool)
- API server mode
- Docker containerization

### Planned for v3.0
- Video protection (frame-by-frame)
- Neural watermarking
- GAN-based perturbations
- Blockchain integration
- Zero-knowledge proofs
- Real-time streaming protection

---

## Contributors

Special thanks to:
- Original PixelGuard concept and v1.0
- Community feedback and testing
- Open-source libraries: OpenCV, PyTorch, Pillow, SciPy
- Adversarial examples research community
- Digital watermarking researchers

---

**For detailed technical information, see [TECHNICAL.md](TECHNICAL.md)**

**For quick start, see [QUICKSTART.md](QUICKSTART.md)**

**For complete documentation, see [README.md](README.md)**
## License

This project is available under a dual license:

1. **Non-Commercial Use**: For non-commercial purposes, this project is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0). This allows for sharing and adaptation of the code for non-commercial purposes, with appropriate attribution.

2. **Commercial Use**: Any commercial use, including but not limited to selling the code, using it in commercial products or services, or any revenue-generating activities, requires a separate commercial license. You must contact the project owner to discuss terms and potential payment.

Please see the [LICENSE](LICENSE) file for full details on both licenses.

For commercial licensing github repo owner

By using this project, you agree to abide by the terms of the appropriate license based on your intended use.



