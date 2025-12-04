#!/usr/bin/env python3
"""
Advanced Image Protector - PixelGuard Enhanced
Protects images from AI scraping and training using multiple techniques
"""

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from PIL.PngImagePlugin import PngInfo
import os
import logging
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import json
import hashlib
import base64
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from tqdm import tqdm

# Cryptography
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.backends import default_backend

# Image processing
import piexif
from scipy.fftpack import dct, idct
from scipy import ndimage
import pywt
import qrcode
import imagehash

# Deep learning
try:
    import torch
    import torch.nn as nn
    import torchvision.models as models
    import torchvision.transforms as transforms
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logging.warning("PyTorch not available. Adversarial perturbations will be disabled.")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('image_protector.log'),
        logging.StreamHandler()
    ]
)

class AdvancedImageProtector:
    """
    Advanced image protection system using multiple techniques to prevent AI training
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the image protector with optional configuration"""
        self.config = config or self._default_config()
        
        # Generate or load RSA keys
        self.private_key, self.public_key = self._setup_keys()
        
        self.supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp', '.gif'}
        
        # Setup deep learning models if available
        if TORCH_AVAILABLE and self.config['use_adversarial']:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logging.info(f"Using device: {self.device}")
            self._setup_models()
        
        # Statistics tracking
        self.stats = {
            'processed': 0,
            'failed': 0,
            'skipped': 0
        }

    def _default_config(self) -> Dict:
        """Return default configuration"""
        return {
            'dct_strength': 0.08,
            'wavelet_strength': 0.06,
            'fourier_strength': 0.07,
            'adversarial_strength': 0.02,
            'qr_opacity': 0.03,
            'noise_strength': 0.015,
            'color_shift_strength': 0.05,
            'use_adversarial': True,
            'use_frequency_masking': True,
            'use_texture_synthesis': True,
            'use_multi_scale': True,
            'jpeg_quality': 95,
            'png_compression': 6,
            'preserve_quality': True,
            'add_visible_watermark': False,
            'watermark_text': 'Protected',
            'batch_size': 4,
            'output_format': 'same'  # 'same', 'png', 'jpg'
        }

    def _setup_keys(self) -> Tuple:
        """Setup or load RSA key pair"""
        key_dir = Path('keys')
        key_dir.mkdir(exist_ok=True)
        
        private_key_path = key_dir / 'private_key.pem'
        public_key_path = key_dir / 'public_key.pem'
        
        if private_key_path.exists() and public_key_path.exists():
            # Load existing keys
            with open(private_key_path, 'rb') as f:
                private_key = serialization.load_pem_private_key(
                    f.read(),
                    password=None,
                    backend=default_backend()
                )
            with open(public_key_path, 'rb') as f:
                public_key = serialization.load_pem_public_key(
                    f.read(),
                    backend=default_backend()
                )
            logging.info("Loaded existing RSA keys")
        else:
            # Generate new keys
            private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=4096,
                backend=default_backend()
            )
            public_key = private_key.public_key()
            
            # Save keys
            with open(private_key_path, 'wb') as f:
                f.write(private_key.private_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PrivateFormat.PKCS8,
                    encryption_algorithm=serialization.NoEncryption()
                ))
            with open(public_key_path, 'wb') as f:
                f.write(public_key.public_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PublicFormat.SubjectPublicKeyInfo
                ))
            logging.info("Generated new RSA keys")
        
        return private_key, public_key

    def _setup_models(self):
        """Setup deep learning models for adversarial perturbations"""
        try:
            # Load multiple models for diverse perturbations
            self.models = []
            
            # ResNet50
            resnet = models.resnet50(pretrained=True).to(self.device)
            resnet.eval()
            self.models.append(('resnet50', resnet))
            
            # VGG16
            vgg = models.vgg16(pretrained=True).to(self.device)
            vgg.eval()
            self.models.append(('vgg16', vgg))
            
            logging.info(f"Loaded {len(self.models)} models for adversarial perturbations")
            
            # Define preprocessing
            self.preprocess = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            
        except Exception as e:
            logging.error(f"Error setting up models: {e}")
            self.models = []

    def protect_image(self, image_path: str, output_dir: str = 'protected_images', 
                     custom_config: Optional[Dict] = None) -> str:
        """
        Protect a single image with multiple techniques
        
        Args:
            image_path: Path to input image
            output_dir: Directory for output
            custom_config: Optional custom configuration overrides
            
        Returns:
            Status message
        """
        try:
            # Merge custom config with defaults
            config = {**self.config, **(custom_config or {})}
            
            os.makedirs(output_dir, exist_ok=True)
            logging.info(f"Processing: {image_path}")

            # Validate file format
            file_extension = Path(image_path).suffix.lower()
            if file_extension not in self.supported_formats:
                self.stats['skipped'] += 1
                return f"Unsupported format: {file_extension}"

            # Load image
            image, original_mode, original_info = self._load_image(image_path)
            if image is None:
                self.stats['failed'] += 1
                return "Failed to load image"
            
            original_shape = image.shape
            
            # Apply protection layers
            protected_image = image.copy()
            
            # Layer 1: Frequency domain watermarking (invisible but robust)
            protected_image = self._apply_multi_channel_dct(protected_image, config['dct_strength'])
            protected_image = self._apply_wavelet_protection(protected_image, config['wavelet_strength'])
            protected_image = self._apply_fourier_masking(protected_image, config['fourier_strength'])
            
            # Layer 2: Spatial domain protection
            if config['use_texture_synthesis']:
                protected_image = self._apply_texture_perturbation(protected_image)
            
            protected_image = self._apply_strategic_noise(protected_image, config['noise_strength'])
            protected_image = self._apply_color_shift(protected_image, config['color_shift_strength'])
            
            # Layer 3: High-frequency manipulation (confuses AI models)
            if config['use_frequency_masking']:
                protected_image = self._apply_high_freq_masking(protected_image)
            
            # Layer 4: Adversarial perturbations (anti-AI training)
            if TORCH_AVAILABLE and config['use_adversarial'] and self.models:
                protected_image = self._apply_multi_model_adversarial(
                    protected_image, 
                    epsilon=config['adversarial_strength']
                )
            
            # Layer 5: Multi-scale protection
            if config['use_multi_scale']:
                protected_image = self._apply_multi_scale_protection(protected_image)
            
            # Layer 6: Invisible QR code tracking
            protected_image = self._apply_invisible_qr(
                protected_image, 
                opacity=config['qr_opacity'],
                data=self._generate_tracking_id()
            )
            
            # Layer 7: Steganographic signature
            protected_image = self._apply_robust_steganography(protected_image)
            
            # Layer 8: Visible watermark (optional)
            if config['add_visible_watermark']:
                protected_image = self._apply_visible_watermark(
                    protected_image,
                    config['watermark_text']
                )
            
            # Ensure shape is preserved
            if protected_image.shape != original_shape:
                protected_image = cv2.resize(protected_image, (original_shape[1], original_shape[0]))
            
            # Generate cryptographic signature
            image_bytes = cv2.imencode('.png', protected_image)[1].tobytes()
            signature_data = self._generate_signature(image_bytes)
            
            # Create protection metadata
            protection_info = self._create_protection_metadata(
                protected_image, 
                signature_data,
                config
            )
            
            # Save protected image
            output_path = self._save_protected_image(
                protected_image,
                image_path,
                output_dir,
                protection_info,
                original_info,
                config
            )
            
            self.stats['processed'] += 1
            return f"✓ Protected: {output_path}"
            
        except Exception as e:
            self.stats['failed'] += 1
            logging.error(f"Error processing {image_path}: {str(e)}", exc_info=True)
            return f"✗ Error: {image_path} - {str(e)}"

    def _load_image(self, image_path: str) -> Tuple[Optional[np.ndarray], str, Dict]:
        """Load image and preserve metadata"""
        try:
            with Image.open(image_path) as img:
                original_mode = img.mode
                original_info = img.info.copy()
                
                # Convert to RGB
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                image = np.array(img)
                
                # Convert to BGR for OpenCV
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                
                return image, original_mode, original_info
                
        except Exception as e:
            logging.error(f"Failed to load {image_path}: {e}")
            return None, None, {}

    def _apply_multi_channel_dct(self, image: np.ndarray, strength: float) -> np.ndarray:
        """Apply DCT watermarking to all color channels"""
        logging.debug("Applying multi-channel DCT watermark")
        protected = image.copy().astype(float)
        
        for channel in range(3):
            channel_data = protected[:, :, channel]
            
            # Apply DCT
            dct_data = dct(dct(channel_data.T, norm='ortho').T, norm='ortho')
            
            # Create deterministic watermark
            np.random.seed(42 + channel)
            watermark = np.random.normal(0, 3, dct_data.shape)
            
            # Focus on mid-frequency components
            rows, cols = dct_data.shape
            mid_start_r, mid_end_r = rows // 4, 3 * rows // 4
            mid_start_c, mid_end_c = cols // 4, 3 * cols // 4
            
            dct_data[mid_start_r:mid_end_r, mid_start_c:mid_end_c] += strength * watermark[mid_start_r:mid_end_r, mid_start_c:mid_end_c]
            
            # Inverse DCT
            protected[:, :, channel] = idct(idct(dct_data.T, norm='ortho').T, norm='ortho')
        
        return np.clip(protected, 0, 255).astype(np.uint8)

    def _apply_wavelet_protection(self, image: np.ndarray, strength: float) -> np.ndarray:
        """Apply wavelet-based protection"""
        logging.debug("Applying wavelet protection")
        protected = image.copy().astype(float)
        
        for channel in range(3):
            channel_data = protected[:, :, channel]
            
            # Multi-level wavelet decomposition
            coeffs = pywt.wavedec2(channel_data, 'db4', level=3)
            
            # Modify approximation coefficients
            cA = coeffs[0]
            np.random.seed(24 + channel)
            watermark = np.random.normal(0, 2, cA.shape)
            coeffs[0] = cA + strength * watermark
            
            # Modify detail coefficients slightly
            for i in range(1, len(coeffs)):
                cH, cV, cD = coeffs[i]
                np.random.seed(24 + channel + i * 10)
                coeffs[i] = (
                    cH + strength * 0.5 * np.random.normal(0, 1, cH.shape),
                    cV + strength * 0.5 * np.random.normal(0, 1, cV.shape),
                    cD + strength * 0.5 * np.random.normal(0, 1, cD.shape)
                )
            
            # Reconstruct
            protected[:, :, channel] = pywt.waverec2(coeffs, 'db4')
        
        return np.clip(protected, 0, 255).astype(np.uint8)

    def _apply_fourier_masking(self, image: np.ndarray, strength: float) -> np.ndarray:
        """Apply Fourier domain masking"""
        logging.debug("Applying Fourier masking")
        protected = image.copy().astype(float)
        
        for channel in range(3):
            channel_data = protected[:, :, channel]
            
            # FFT
            f_transform = np.fft.fft2(channel_data)
            f_shifted = np.fft.fftshift(f_transform)
            
            # Create protection pattern in frequency domain
            rows, cols = f_shifted.shape
            crow, ccol = rows // 2, cols // 2
            
            # Create ring mask (mid-frequencies)
            y, x = np.ogrid[:rows, :cols]
            mask = np.sqrt((x - ccol)**2 + (y - crow)**2)
            ring_mask = ((mask > rows // 8) & (mask < rows // 4)).astype(float)
            
            # Add watermark to mid-frequencies
            np.random.seed(36 + channel)
            watermark = np.random.normal(0, 100, f_shifted.shape) * ring_mask
            f_shifted += strength * watermark
            
            # Inverse FFT
            f_ishift = np.fft.ifftshift(f_shifted)
            protected[:, :, channel] = np.fft.ifft2(f_ishift).real
        
        return np.clip(protected, 0, 255).astype(np.uint8)

    def _apply_texture_perturbation(self, image: np.ndarray) -> np.ndarray:
        """Apply subtle texture perturbations that confuse AI models"""
        logging.debug("Applying texture perturbation")
        
        # Create texture noise based on image gradients
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Compute gradients
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
        
        # Normalize
        gradient_magnitude = (gradient_magnitude / gradient_magnitude.max() * 255).astype(np.uint8)
        
        # Create perturbation based on texture
        np.random.seed(99)
        perturbation = np.random.normal(0, 1, image.shape)
        
        # Scale perturbation by gradient magnitude
        for channel in range(3):
            perturbation[:, :, channel] *= (gradient_magnitude / 255.0)
        
        protected = image.astype(float) + 2.0 * perturbation
        return np.clip(protected, 0, 255).astype(np.uint8)

    def _apply_strategic_noise(self, image: np.ndarray, strength: float) -> np.ndarray:
        """Apply strategic noise that targets AI model weaknesses"""
        logging.debug("Applying strategic noise")
        
        # Mix of different noise types
        np.random.seed(77)
        
        # Gaussian noise
        gaussian_noise = np.random.normal(0, strength * 255, image.shape)
        
        # Salt and pepper noise
        salt_pepper = np.random.choice([-1, 0, 1], size=image.shape, p=[0.01, 0.98, 0.01])
        salt_pepper_noise = salt_pepper * strength * 255
        
        # Perlin-like noise (smooth)
        smooth_noise = ndimage.gaussian_filter(np.random.randn(*image.shape), sigma=2)
        smooth_noise = smooth_noise * strength * 50
        
        # Combine noises
        combined_noise = 0.6 * gaussian_noise + 0.2 * salt_pepper_noise + 0.2 * smooth_noise
        
        protected = image.astype(float) + combined_noise
        return np.clip(protected, 0, 255).astype(np.uint8)

    def _apply_color_shift(self, image: np.ndarray, strength: float) -> np.ndarray:
        """Apply subtle color shifts that are imperceptible to humans"""
        logging.debug("Applying color shift")
        
        # Convert to LAB color space for perceptual color manipulation
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
        
        # Subtle shifts in a and b channels (color)
        np.random.seed(55)
        lab[:, :, 1] += np.random.uniform(-strength * 10, strength * 10)  # a channel
        lab[:, :, 2] += np.random.uniform(-strength * 10, strength * 10)  # b channel
        
        # Clip to valid range
        lab[:, :, 1] = np.clip(lab[:, :, 1], 0, 255)
        lab[:, :, 2] = np.clip(lab[:, :, 2], 0, 255)
        
        # Convert back to BGR
        protected = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
        return protected

    def _apply_high_freq_masking(self, image: np.ndarray) -> np.ndarray:
        """Mask high-frequency components that AI models rely on"""
        logging.debug("Applying high-frequency masking")
        
        # Apply high-pass filter
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]]) / 9.0
        
        high_freq = cv2.filter2D(image.astype(float), -1, kernel)
        
        # Add subtle high-frequency noise
        np.random.seed(88)
        noise = np.random.normal(0, 2, image.shape)
        
        protected = image.astype(float) + 0.3 * high_freq * 0.1 + noise
        return np.clip(protected, 0, 255).astype(np.uint8)

    def _apply_multi_model_adversarial(self, image: np.ndarray, epsilon: float) -> np.ndarray:
        """Apply adversarial perturbations using multiple models"""
        logging.debug("Applying multi-model adversarial perturbations")
        
        if not self.models:
            return image
        
        try:
            # Convert to PIL then to tensor
            pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            img_tensor = self.preprocess(pil_img).unsqueeze(0).to(self.device)
            img_tensor.requires_grad = True
            
            # Accumulate gradients from multiple models
            total_grad = torch.zeros_like(img_tensor)
            
            for model_name, model in self.models:
                # Forward pass
                output = model(img_tensor)
                
                # Use different target classes for diversity
                if model_name == 'resnet50':
                    target = torch.tensor([0]).to(self.device)  # Target class 0
                else:
                    target = torch.tensor([500]).to(self.device)  # Different target
                
                # Calculate loss
                loss = nn.functional.cross_entropy(output, target)
                
                # Backward pass
                model.zero_grad()
                if img_tensor.grad is not None:
                    img_tensor.grad.zero_()
                loss.backward()
                
                # Accumulate gradients
                total_grad += img_tensor.grad.data
            
            # Average gradients
            total_grad /= len(self.models)
            
            # Generate adversarial example using PGD-like approach
            sign_data_grad = total_grad.sign()
            perturbed_tensor = img_tensor + epsilon * sign_data_grad
            perturbed_tensor = torch.clamp(perturbed_tensor, 0, 1)
            
            # Convert back to numpy
            perturbed_image = perturbed_tensor.squeeze().permute(1, 2, 0).detach().cpu().numpy()
            perturbed_image = (perturbed_image * 255).astype(np.uint8)
            
            # Resize to original dimensions if needed
            if perturbed_image.shape[:2] != image.shape[:2]:
                perturbed_image = cv2.resize(perturbed_image, (image.shape[1], image.shape[0]))
            
            # Convert back to BGR
            perturbed_image = cv2.cvtColor(perturbed_image, cv2.COLOR_RGB2BGR)
            
            return perturbed_image
            
        except Exception as e:
            logging.error(f"Error in adversarial perturbation: {e}")
            return image

    def _apply_multi_scale_protection(self, image: np.ndarray) -> np.ndarray:
        """Apply protection at multiple scales"""
        logging.debug("Applying multi-scale protection")
        
        protected = image.copy().astype(float)
        
        # Create image pyramid
        scales = [1.0, 0.5, 0.25]
        
        for scale in scales:
            if scale != 1.0:
                # Downscale
                scaled = cv2.resize(image, None, fx=scale, fy=scale)
            else:
                scaled = image.copy()
            
            # Add noise at this scale
            np.random.seed(int(scale * 100))
            noise = np.random.normal(0, 1, scaled.shape)
            scaled = scaled.astype(float) + noise * 2.0 * scale
            scaled = np.clip(scaled, 0, 255).astype(np.uint8)
            
            # Upscale back if needed
            if scale != 1.0:
                scaled = cv2.resize(scaled, (image.shape[1], image.shape[0]))
            
            # Blend with protected image
            protected = 0.7 * protected + 0.3 * scaled.astype(float)
        
        return np.clip(protected, 0, 255).astype(np.uint8)

    def _apply_invisible_qr(self, image: np.ndarray, opacity: float, data: str) -> np.ndarray:
        """Apply invisible QR code with tracking data"""
        logging.debug("Applying invisible QR code")
        
        # Generate QR code
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_H,
            box_size=10,
            border=4
        )
        qr.add_data(data)
        qr.make(fit=True)
        
        qr_img = qr.make_image(fill_color="black", back_color="white")
        qr_array = np.array(qr_img.convert('L'))
        
        # Resize to match image
        qr_resized = cv2.resize(qr_array, (image.shape[1], image.shape[0]))
        
        # Normalize and apply with low opacity
        qr_float = qr_resized.astype(np.float32) / 255.0
        image_float = image.astype(np.float32)
        
        # Apply to blue channel primarily (less noticeable)
        for i in range(3):
            channel_opacity = opacity * (0.5 if i == 0 else 0.2)  # More in blue
            image_float[:, :, i] = image_float[:, :, i] * (1 - channel_opacity * (1 - qr_float)) + \
                                   channel_opacity * qr_float * 255
        
        return np.clip(image_float, 0, 255).astype(np.uint8)

    def _apply_robust_steganography(self, image: np.ndarray) -> np.ndarray:
        """Apply robust LSB steganography"""
        logging.debug("Applying steganography")
        
        # Create message with timestamp
        message = f"PROTECTED:{datetime.now().isoformat()}:{self._generate_tracking_id()}"
        
        # Convert to binary
        binary_message = ''.join(format(ord(char), '08b') for char in message)
        binary_message += '1111111111111110'  # End marker
        
        # Embed in LSB of all channels
        protected = image.copy()
        data_index = 0
        
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                for k in range(3):
                    if data_index < len(binary_message):
                        # Modify LSB
                        protected[i, j, k] = (protected[i, j, k] & 0xFE) | int(binary_message[data_index])
                        data_index += 1
                    else:
                        return protected
        
        return protected

    def _apply_visible_watermark(self, image: np.ndarray, text: str) -> np.ndarray:
        """Apply visible watermark"""
        logging.debug("Applying visible watermark")
        
        # Convert to PIL for text drawing
        pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)
        
        # Calculate watermark position (bottom right)
        img_width, img_height = pil_img.size
        
        try:
            # Try to use a nice font
            font_size = max(20, img_height // 30)
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
        except:
            # Fallback to default
            font = ImageFont.load_default()
        
        # Get text size
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # Position
        x = img_width - text_width - 10
        y = img_height - text_height - 10
        
        # Draw with transparency
        watermark = Image.new('RGBA', pil_img.size, (255, 255, 255, 0))
        watermark_draw = ImageDraw.Draw(watermark)
        watermark_draw.text((x, y), text, font=font, fill=(255, 255, 255, 128))
        
        # Composite
        pil_img = pil_img.convert('RGBA')
        pil_img = Image.alpha_composite(pil_img, watermark)
        pil_img = pil_img.convert('RGB')
        
        # Convert back to OpenCV
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    def _generate_tracking_id(self) -> str:
        """Generate unique tracking ID"""
        import uuid
        return str(uuid.uuid4())

    def _generate_signature(self, data: bytes) -> Dict:
        """Generate cryptographic signature"""
        signature = self.private_key.sign(
            data,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        
        return {
            'signature': base64.b64encode(signature).decode('utf-8'),
            'hash': hashlib.sha256(data).hexdigest(),
            'algorithm': 'RSA-PSS-SHA256'
        }

    def _create_protection_metadata(self, image: np.ndarray, signature_data: Dict, 
                                   config: Dict) -> Dict:
        """Create comprehensive protection metadata"""
        # Convert to PIL for hashing
        pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        metadata = {
            'version': '2.0',
            'timestamp': datetime.now().isoformat(),
            'signature': signature_data['signature'],
            'image_hash': signature_data['hash'],
            'perceptual_hash': str(imagehash.phash(pil_img)),
            'average_hash': str(imagehash.average_hash(pil_img)),
            'dhash': str(imagehash.dhash(pil_img)),
            'protection_methods': [
                'multi_channel_dct',
                'wavelet_decomposition',
                'fourier_masking',
                'texture_perturbation',
                'strategic_noise',
                'color_shift',
                'high_frequency_masking',
                'adversarial_perturbation' if config['use_adversarial'] and TORCH_AVAILABLE else None,
                'multi_scale_protection',
                'invisible_qr',
                'steganography'
            ],
            'protection_strength': {
                'dct': config['dct_strength'],
                'wavelet': config['wavelet_strength'],
                'fourier': config['fourier_strength'],
                'adversarial': config['adversarial_strength'] if config['use_adversarial'] else 0
            },
            'tracking_id': self._generate_tracking_id()
        }
        
        # Remove None values
        metadata['protection_methods'] = [m for m in metadata['protection_methods'] if m]
        
        return metadata

    def _save_protected_image(self, image: np.ndarray, original_path: str,
                             output_dir: str, metadata: Dict, 
                             original_info: Dict, config: Dict) -> str:
        """Save protected image with metadata"""
        
        # Determine output format
        original_ext = Path(original_path).suffix.lower()
        if config['output_format'] == 'same':
            output_ext = original_ext
        elif config['output_format'] == 'png':
            output_ext = '.png'
        elif config['output_format'] == 'jpg':
            output_ext = '.jpg'
        else:
            output_ext = original_ext
        
        # Create output path
        output_path = Path(output_dir) / f"protected_{Path(original_path).stem}{output_ext}"
        
        # Convert to PIL
        pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Embed metadata
        if output_ext in ['.jpg', '.jpeg']:
            # JPEG: Use EXIF
            try:
                exif_dict = {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}, "thumbnail": None}
                if 'exif' in original_info:
                    try:
                        exif_dict = piexif.load(original_info['exif'])
                    except:
                        pass
                
                # Add protection info to ImageDescription
                exif_dict["0th"][piexif.ImageIFD.ImageDescription] = json.dumps(metadata)
                exif_dict["0th"][piexif.ImageIFD.Software] = "PixelGuard Advanced v2.0"
                exif_bytes = piexif.dump(exif_dict)
                
                pil_img.save(str(output_path), exif=exif_bytes, quality=config['jpeg_quality'], optimize=True)
            except Exception as e:
                logging.warning(f"Failed to embed EXIF: {e}")
                pil_img.save(str(output_path), quality=config['jpeg_quality'], optimize=True)
        
        else:
            # PNG and others: Use PNG metadata
            png_info = PngInfo()
            png_info.add_text("PixelGuard-Protection", json.dumps(metadata))
            png_info.add_text("Software", "PixelGuard Advanced v2.0")
            
            pil_img.save(str(output_path), pnginfo=png_info, 
                        compress_level=config['png_compression'], optimize=True)
        
        logging.info(f"Saved: {output_path}")
        return str(output_path)

    def verify_image(self, image_path: str) -> Dict:
        """
        Verify if an image is protected and check its integrity
        
        Returns:
            Dictionary with verification results
        """
        logging.info(f"Verifying: {image_path}")
        
        try:
            with Image.open(image_path) as img:
                # Try to extract metadata
                metadata = None
                
                # Try EXIF first
                if "exif" in img.info:
                    try:
                        exif_dict = piexif.load(img.info["exif"])
                        desc = exif_dict["0th"].get(piexif.ImageIFD.ImageDescription, b"")
                        if isinstance(desc, bytes):
                            desc = desc.decode('utf-8')
                        metadata = json.loads(desc)
                    except:
                        pass
                
                # Try PNG metadata
                if not metadata and "PixelGuard-Protection" in img.info:
                    metadata = json.loads(img.info["PixelGuard-Protection"])
                
                if not metadata:
                    return {
                        'protected': False,
                        'message': 'No protection metadata found'
                    }
                
                # Load image for verification
                image = np.array(img.convert('RGB'))
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Verify hashes
            image_bytes = cv2.imencode('.png', image)[1].tobytes()
            current_hash = hashlib.sha256(image_bytes).hexdigest()
            
            # Perceptual hashes
            pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            current_phash = str(imagehash.phash(pil_img))
            current_ahash = str(imagehash.average_hash(pil_img))
            
            # Calculate hash differences
            stored_phash = imagehash.hex_to_hash(metadata['perceptual_hash'])
            current_phash_obj = imagehash.hex_to_hash(current_phash)
            hash_difference = stored_phash - current_phash_obj
            
            # Verify signature
            signature_valid = False
            try:
                signature = base64.b64decode(metadata['signature'].encode('utf-8'))
                self.public_key.verify(
                    signature,
                    image_bytes,
                    padding.PSS(
                        mgf=padding.MGF1(hashes.SHA256()),
                        salt_length=padding.PSS.MAX_LENGTH
                    ),
                    hashes.SHA256()
                )
                signature_valid = True
            except:
                pass
            
            # Calculate protection age
            protection_time = datetime.fromisoformat(metadata['timestamp'])
            age_days = (datetime.now() - protection_time).days
            
            # Determine verification status
            if hash_difference == 0 and signature_valid:
                status = 'VERIFIED'
                message = 'Image is authentic and unmodified'
            elif hash_difference <= 5 and signature_valid:
                status = 'LIKELY_AUTHENTIC'
                message = 'Image appears authentic with minor acceptable differences'
            elif signature_valid:
                status = 'MODIFIED'
                message = 'Image signature is valid but content has been modified'
            else:
                status = 'TAMPERED'
                message = 'Image may have been tampered with'
            
            return {
                'protected': True,
                'status': status,
                'message': message,
                'details': {
                    'signature_valid': signature_valid,
                    'hash_difference': hash_difference,
                    'protection_age_days': age_days,
                    'protection_methods': metadata.get('protection_methods', []),
                    'timestamp': metadata['timestamp'],
                    'tracking_id': metadata.get('tracking_id', 'N/A')
                }
            }
            
        except Exception as e:
            logging.error(f"Verification error: {e}", exc_info=True)
            return {
                'protected': False,
                'error': str(e)
            }

    def batch_process(self, image_paths: List[str], output_dir: str = 'protected_images_batch',
                     num_workers: Optional[int] = None, custom_config: Optional[Dict] = None):
        """
        Process multiple images in parallel
        
        Args:
            image_paths: List of image paths
            output_dir: Output directory
            num_workers: Number of parallel workers (None = CPU count)
            custom_config: Optional configuration overrides
        """
        if num_workers is None:
            num_workers = min(multiprocessing.cpu_count(), len(image_paths))
        
        os.makedirs(output_dir, exist_ok=True)
        
        logging.info(f"Processing {len(image_paths)} images with {num_workers} workers")
        
        results = []
        
        # Use process pool for parallel processing
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # Submit all tasks
            future_to_path = {
                executor.submit(self._process_single, path, output_dir, custom_config): path
                for path in image_paths
            }
            
            # Progress bar
            with tqdm(total=len(image_paths), desc="Protecting images") as pbar:
                for future in as_completed(future_to_path):
                    path = future_to_path[future]
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        logging.error(f"Failed to process {path}: {e}")
                        results.append(f"✗ Error: {path}")
                    pbar.update(1)
        
        # Print summary
        self._print_summary()
        
        return results

    def _process_single(self, image_path: str, output_dir: str, 
                       custom_config: Optional[Dict]) -> str:
        """Process single image (used for multiprocessing)"""
        return self.protect_image(image_path, output_dir, custom_config)

    def _print_summary(self):
        """Print processing summary"""
        total = self.stats['processed'] + self.stats['failed'] + self.stats['skipped']
        
        print("\n" + "="*50)
        print("PROTECTION SUMMARY")
        print("="*50)
        print(f"Total images: {total}")
        print(f"✓ Successfully protected: {self.stats['processed']}")
        print(f"✗ Failed: {self.stats['failed']}")
        print(f"⊘ Skipped: {self.stats['skipped']}")
        print("="*50 + "\n")

    def export_config(self, filepath: str):
        """Export current configuration to file"""
        with open(filepath, 'w') as f:
            json.dump(self.config, f, indent=2)
        logging.info(f"Configuration exported to {filepath}")

    def load_config(self, filepath: str):
        """Load configuration from file"""
        with open(filepath, 'r') as f:
            self.config = json.load(f)
        logging.info(f"Configuration loaded from {filepath}")


def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(
        description='PixelGuard Advanced - Protect images from AI scraping',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Protect a single image
  python advanced_image_protector.py protect image.jpg
  
  # Protect multiple images
  python advanced_image_protector.py protect image1.jpg image2.png image3.jpg
  
  # Batch protect all images in a directory
  python advanced_image_protector.py protect ./images/*.jpg -o protected_output
  
  # Use custom protection strength
  python advanced_image_protector.py protect image.jpg --dct-strength 0.1 --adversarial-strength 0.03
  
  # Verify a protected image
  python advanced_image_protector.py verify protected_image.jpg
  
  # Export/import configuration
  python advanced_image_protector.py export-config my_settings.json
  python advanced_image_protector.py protect image.jpg --config my_settings.json
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Protect command
    protect_parser = subparsers.add_parser('protect', help='Protect images')
    protect_parser.add_argument('images', nargs='+', help='Image files to protect')
    protect_parser.add_argument('-o', '--output', default='protected_images', 
                               help='Output directory (default: protected_images)')
    protect_parser.add_argument('--config', help='Load configuration from JSON file')
    protect_parser.add_argument('--dct-strength', type=float, help='DCT watermark strength')
    protect_parser.add_argument('--wavelet-strength', type=float, help='Wavelet watermark strength')
    protect_parser.add_argument('--adversarial-strength', type=float, help='Adversarial perturbation strength')
    protect_parser.add_argument('--no-adversarial', action='store_true', 
                               help='Disable adversarial perturbations')
    protect_parser.add_argument('--visible-watermark', action='store_true',
                               help='Add visible watermark')
    protect_parser.add_argument('--watermark-text', default='Protected',
                               help='Text for visible watermark')
    protect_parser.add_argument('--workers', type=int, help='Number of parallel workers')
    protect_parser.add_argument('--output-format', choices=['same', 'png', 'jpg'],
                               default='same', help='Output format')
    
    # Verify command
    verify_parser = subparsers.add_parser('verify', help='Verify protected image')
    verify_parser.add_argument('image', help='Image file to verify')
    verify_parser.add_argument('--json', action='store_true', help='Output as JSON')
    
    # Export config command
    export_parser = subparsers.add_parser('export-config', help='Export configuration')
    export_parser.add_argument('output', help='Output JSON file')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Initialize protector
    protector = AdvancedImageProtector()
    
    if args.command == 'protect':
        # Load config if specified
        if args.config:
            protector.load_config(args.config)
        
        # Apply command-line overrides
        custom_config = {}
        if args.dct_strength:
            custom_config['dct_strength'] = args.dct_strength
        if args.wavelet_strength:
            custom_config['wavelet_strength'] = args.wavelet_strength
        if args.adversarial_strength:
            custom_config['adversarial_strength'] = args.adversarial_strength
        if args.no_adversarial:
            custom_config['use_adversarial'] = False
        if args.visible_watermark:
            custom_config['add_visible_watermark'] = True
            custom_config['watermark_text'] = args.watermark_text
        if args.output_format:
            custom_config['output_format'] = args.output_format
        
        # Process images
        if len(args.images) == 1:
            result = protector.protect_image(args.images[0], args.output, custom_config)
            print(result)
        else:
            results = protector.batch_process(
                args.images, 
                args.output,
                num_workers=args.workers,
                custom_config=custom_config
            )
            for result in results:
                print(result)
    
    elif args.command == 'verify':
        result = protector.verify_image(args.image)
        
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print("\n" + "="*50)
            print("VERIFICATION RESULT")
            print("="*50)
            print(f"Protected: {result.get('protected', False)}")
            print(f"Status: {result.get('status', 'UNKNOWN')}")
            print(f"Message: {result.get('message', 'N/A')}")
            
            if 'details' in result:
                print("\nDetails:")
                for key, value in result['details'].items():
                    print(f"  {key}: {value}")
            
            if 'error' in result:
                print(f"\nError: {result['error']}")
            
            print("="*50 + "\n")
    
    elif args.command == 'export-config':
        protector.export_config(args.output)
        print(f"Configuration exported to {args.output}")


if __name__ == "__main__":
    main()
