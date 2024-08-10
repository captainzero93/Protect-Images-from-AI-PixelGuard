# PixelGuard AI

## Overview
PixelGuard AI is an advanced Python-based tool designed to protect images from AI scraping and unauthorized use in AI training, such as facial recognition models or style transfer algorithms. It employs multiple invisible protection techniques that are imperceptible to the human eye but can significantly interfere with AI processing.

## Features
- **Multiple Invisible Protection Techniques**:
  - DCT (Discrete Cosine Transform) Watermarking
  - Wavelet-based Watermarking
  - Fourier Transform Watermarking
  - Adversarial Perturbation
  - Color Jittering
  - Invisible QR Code Embedding
  - Steganography
- **Digital Signature and Hash Verification** for tamper detection
- **Perceptual Hash** for content change detection
- **Timestamp Verification** to check the age of protection
- **Support for Multiple Image Formats**: JPEG, PNG, BMP, TIFF, WebP
- **Batch Processing** capability with progress tracking
- **User-friendly GUI** for easy interaction
- **Verification Tool** to check if an image has been protected and/or tampered with

## Installation
1. Clone this repository:
   ```
   git clone https://github.com/yourusername/PixelGuard-AI.git
   cd PixelGuard-AI
   ```

2. Set up a virtual environment:
   - For Windows:
     ```
     python -m venv venv
     venv\Scripts\activate
     ```
   - For macOS and Linux:
     ```
     python3 -m venv venv
     source venv/bin/activate
     ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

4. When you're done working on the project, you can deactivate the virtual environment:
   ```
   deactivate
   ```

## Usage
Activate your virtual environment (if not already activated), then run the script:
```
python pixelguard_ai.py
```
This will open a GUI with three main options:
1. **Protect Single Image**: Select a single image to protect.
2. **Batch Protect Images**: Select multiple images to protect in batch.
3. **Verify Image**: Check if an image has been protected and if it has been tampered with.

### Protecting Images
1. Click on "Protect Single Image" or "Batch Protect Images".
2. Select the image(s) you want to protect.
3. Choose an output directory for the protected images.
4. Wait for the process to complete. A success message will appear when done.

### Verifying Images
1. Click on "Verify Image".
2. Select the image you want to verify.
3. The tool will check if the image contains protection information, if it has been tampered with, and how long ago it was protected.

## How It Works
PixelGuard AI uses several techniques to protect images:
1. **DCT Watermarking**: Embeds a watermark in the frequency domain of the blue channel.
2. **Wavelet-based Watermarking**: Embeds a watermark in the wavelet domain of the green channel.
3. **Fourier Transform Watermarking**: Applies a watermark in the frequency domain of the red channel.
4. **Adversarial Perturbation**: Adds minor perturbations to the image that are designed to confuse AI models.
5. **Color Jittering**: Randomly adjusts brightness, contrast, and saturation to add another layer of protection.
6. **Invisible QR Code**: Embeds an invisible QR code containing image information.
7. **Steganography**: Hides additional protection data within the image itself.
8. **Digital Signature**: Signs the entire image to detect any tampering.
9. **Hash Verification**: Uses both a cryptographic hash and a perceptual hash to check if the image has been altered.
10. **Timestamp Verification**: Checks when the image was protected and suggests re-protection if it's too old.

These techniques work together to create multiple layers of protection that are extremely difficult for AI training algorithms to remove or ignore, while remaining imperceptible to human viewers.

## Caution
While these protection methods significantly increase the difficulty of using the images for AI training, no protection method is perfect. Always be cautious about sharing sensitive images online.

## Contributing
Contributions to improve the protection techniques or the user interface are welcome. Please open an issue or submit a pull request.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
