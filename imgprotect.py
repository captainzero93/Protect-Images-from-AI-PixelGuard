import numpy as np
import cv2
from PIL import Image
from PIL.PngImagePlugin import PngInfo
import os
import logging
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding, rsa
import base64
import json
import hashlib
import piexif
from scipy.fftpack import dct, idct
import pywt
import qrcode
import uuid
import imagehash
from datetime import datetime, timedelta
import torch
import torchvision.models as models
import torchvision.transforms as transforms

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class AdvancedImageProtector:
    def __init__(self):
        self.private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048
        )
        self.public_key = self.private_key.public_key()
        self.supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        
        # Load pre-trained ResNet50 model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = models.resnet50(pretrained=True).to(self.device)
        self.model.eval()
        
        # Define image preprocessing
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def protect_image(self, image_path, output_dir='protected_images', dct_strength=0.05, wavelet_strength=0.05, fourier_strength=0.05, adversarial_strength=0.01, qr_opacity=0.05):
        try:
            os.makedirs(output_dir, exist_ok=True)
            logging.debug(f"Processing image: {image_path}")

            file_extension = os.path.splitext(image_path)[1].lower()
            if file_extension not in self.supported_formats:
                return f"Unsupported file format: {file_extension}"

            # Read image using PIL to ensure consistency
            with Image.open(image_path) as img:
                original_exif = img.info.get('exif', b'')
                image = np.array(img)
                if len(image.shape) == 2:  # Convert grayscale to RGB
                    image = np.stack((image,)*3, axis=-1)
                elif image.shape[2] == 4:  # Remove alpha channel if present
                    image = image[:,:,:3]
            
            # Convert to BGR for OpenCV operations
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Apply multiple protection techniques with custom strengths
            protected_image = self.apply_dct_watermark(image, strength=dct_strength)
            protected_image = self.apply_wavelet_watermark(protected_image, strength=wavelet_strength)
            protected_image = self.apply_fourier_watermark(protected_image, strength=fourier_strength)
            protected_image = self.apply_adversarial_perturbation(protected_image, epsilon=adversarial_strength)
            protected_image = self.apply_color_jittering(protected_image)
            protected_image = self.apply_invisible_qr(protected_image, opacity=qr_opacity)
            protected_image = self.apply_steganography(protected_image)

            # Generate signature and hash AFTER all protections are applied
            image_bytes = cv2.imencode('.png', protected_image)[1].tobytes()
            image_hash = hashlib.sha256(image_bytes).hexdigest()
            perceptual_hash = str(imagehash.phash(Image.fromarray(cv2.cvtColor(protected_image, cv2.COLOR_BGR2RGB))))
            signature = self.sign_image(image_bytes)

            # Prepare protection info
            protection_info = {
                "signature": signature,
                "image_hash": image_hash,
                "perceptual_hash": perceptual_hash,
                "timestamp": datetime.now().isoformat()
            }

            # Save the protected image
            final_image_path = os.path.join(output_dir, f'protected_{os.path.basename(image_path)}')
            
            # Convert back to PIL Image for saving
            pil_image = Image.fromarray(cv2.cvtColor(protected_image, cv2.COLOR_BGR2RGB))
            
            # Embed protection info based on file type
            if file_extension.lower() in ['.jpg', '.jpeg']:
                # For JPEG, use EXIF
                exif_dict = piexif.load(original_exif)
                exif_dict["0th"][piexif.ImageIFD.ImageDescription] = json.dumps(protection_info)
                exif_bytes = piexif.dump(exif_dict)
                pil_image.save(final_image_path, exif=exif_bytes, quality=95)
            else:
                # For other formats (including PNG), use metadata
                metadata = PngInfo()
                metadata.add_text("Description", json.dumps(protection_info))
                pil_image.save(final_image_path, pnginfo=metadata)

            logging.debug(f"Saved protected image with embedded info: {final_image_path}")

            return f"Image processing complete. Protected image saved as {final_image_path}"
        except Exception as e:
            logging.error(f"Error processing image: {str(e)}", exc_info=True)
            return f"Error processing image {image_path}: {str(e)}"

    def apply_dct_watermark(self, image, strength):
        logging.debug("Applying DCT watermark")
        blue_channel = image[:,:,0].astype(float)
        dct_blue = dct(dct(blue_channel.T, norm='ortho').T, norm='ortho')
        
        np.random.seed(42)
        watermark = np.random.normal(0, 2, blue_channel.shape)
        
        dct_blue += strength * watermark
        
        blue_channel_watermarked = idct(idct(dct_blue.T, norm='ortho').T, norm='ortho')
        image[:,:,0] = np.clip(blue_channel_watermarked, 0, 255).astype(np.uint8)
        
        return image

    def apply_wavelet_watermark(self, image, strength):
        logging.debug("Applying wavelet watermark")
        green_channel = image[:,:,1].astype(float)
        coeffs = pywt.dwt2(green_channel, 'haar')
        cA, (cH, cV, cD) = coeffs
        
        np.random.seed(24)
        watermark = np.random.normal(0, 1, cA.shape)
        
        cA += strength * watermark
        
        green_channel_watermarked = pywt.idwt2((cA, (cH, cV, cD)), 'haar')
        image[:,:,1] = np.clip(green_channel_watermarked, 0, 255).astype(np.uint8)
        
        return image

    def apply_fourier_watermark(self, image, strength):
        logging.debug("Applying Fourier watermark")
        red_channel = image[:,:,2].astype(float)
        f_transform = np.fft.fft2(red_channel)
        
        np.random.seed(36)
        watermark = np.random.normal(0, 1, f_transform.shape)
        
        f_transform += strength * watermark
        
        red_channel_watermarked = np.fft.ifft2(f_transform).real
        image[:,:,2] = np.clip(red_channel_watermarked, 0, 255).astype(np.uint8)
        
        return image

    def apply_adversarial_perturbation(self, image, epsilon):
        logging.debug("Applying adversarial perturbation")
        
        # Convert image to PyTorch tensor
        img_tensor = self.preprocess(Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))).unsqueeze(0).to(self.device)
        
        # Set requires_grad attribute of tensor
        img_tensor.requires_grad = True
        
        # Forward pass
        output = self.model(img_tensor)
        
        # Calculate loss
        loss = torch.nn.functional.cross_entropy(output, torch.tensor([0]).to(self.device))
        
        # Backward pass
        loss.backward()
        
        # Generate adversarial example
        data_grad = img_tensor.grad.data
        sign_data_grad = data_grad.sign()
        perturbed_image = img_tensor + epsilon * sign_data_grad
        perturbed_image = torch.clamp(perturbed_image, 0, 1)
        
        # Convert back to numpy array
        perturbed_image = perturbed_image.squeeze().permute(1, 2, 0).detach().cpu().numpy()
        perturbed_image = (perturbed_image * 255).astype(np.uint8)
        
        # Convert back to BGR
        perturbed_image = cv2.cvtColor(perturbed_image, cv2.COLOR_RGB2BGR)
        
        return perturbed_image

    def apply_color_jittering(self, image):
        logging.debug("Applying color jittering")
        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        
        # Randomly adjust hue, saturation, and value
        hsv[:,:,0] += np.random.uniform(-10, 10)  # Hue
        hsv[:,:,1] *= np.random.uniform(0.8, 1.2)  # Saturation
        hsv[:,:,2] *= np.random.uniform(0.8, 1.2)  # Value
        
        # Ensure values are within valid ranges
        hsv[:,:,0] = np.clip(hsv[:,:,0], 0, 179)
        hsv[:,:,1] = np.clip(hsv[:,:,1], 0, 255)
        hsv[:,:,2] = np.clip(hsv[:,:,2], 0, 255)
        
        # Convert back to BGR
        jittered_image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        
        return jittered_image

    def apply_invisible_qr(self, image, opacity):
        logging.debug("Applying invisible QR code")
        qr = qrcode.QRCode(version=1, box_size=10, border=5)
        qr.add_data("Protected Image")
        qr.make(fit=True)
        qr_image = qr.make_image(fill_color="black", back_color="white")
        qr_array = np.array(qr_image.convert('L'))
        qr_array = cv2.resize(qr_array, (image.shape[1], image.shape[0]))
        
        # Convert QR code to float and normalize
        qr_float = qr_array.astype(np.float32) / 255.0
        
        # Apply QR code with specified opacity
        image_float = image.astype(np.float32) / 255.0
        image_with_qr = image_float * (1 - opacity * qr_float[:,:,np.newaxis]) + opacity * qr_float[:,:,np.newaxis]
        
        return (image_with_qr * 255).astype(np.uint8)

    def apply_steganography(self, image):
        logging.debug("Applying steganography")
        secret_message = "This image is protected"
        binary_message = ''.join(format(ord(char), '08b') for char in secret_message)
        
        data_index = 0
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                for k in range(3):  # RGB channels
                    if data_index < len(binary_message):
                        image[i, j, k] = (image[i, j, k] & 254) | int(binary_message[data_index])
                        data_index += 1
                    else:
                        return image
        return image

    def sign_image(self, image_bytes):
        logging.debug("Signing image")
        signature = self.private_key.sign(
            image_bytes,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        return base64.b64encode(signature).decode('utf-8')

    def verify_image(self, image_path):
        logging.debug(f"Verifying image: {image_path}")
        try:
            with Image.open(image_path) as img:
                # Check if the image has EXIF data
                if "exif" in img.info:
                    exif_dict = piexif.load(img.info["exif"])
                    protection_info = json.loads(exif_dict["0th"].get(piexif.ImageIFD.ImageDescription, "{}"))
                else:
                    # For PNG files or images without EXIF, try to get the protection info from metadata
                    protection_info = json.loads(img.info.get("Description", "{}"))

            if not protection_info:
                return "This image does not contain protection information."

            # Read image using PIL to ensure consistency with protection process
            with Image.open(image_path) as img:
                image = np.array(img)
                if len(image.shape) == 2:  # Convert grayscale to RGB
                    image = np.stack((image,)*3, axis=-1)
                elif image.shape[2] == 4:  # Remove alpha channel if present
                    image = image[:,:,:3]
            
            # Convert to BGR for OpenCV operations
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            image_bytes = cv2.imencode('.png', image)[1].tobytes()
            current_hash = hashlib.sha256(image_bytes).hexdigest()
            current_perceptual_hash = str(imagehash.phash(Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))))

            logging.debug(f"Current image hash: {current_hash}")
            logging.debug(f"Stored image hash: {protection_info['image_hash']}")
            logging.debug(f"Current perceptual hash: {current_perceptual_hash}")
            logging.debug(f"Stored perceptual hash: {protection_info['perceptual_hash']}")

            if current_hash != protection_info['image_hash']:
                return "Image hash mismatch. The image may have been altered."

            if current_perceptual_hash != protection_info['perceptual_hash']:
                return "Perceptual hash mismatch. The image content may have been changed."

            signature = base64.b64decode(protection_info['signature'].encode('utf-8'))
            try:
                self.public_key.verify(
                    signature,
                    image_bytes,
                    padding.PSS(
                        mgf=padding.MGF1(hashes.SHA256()),
                        salt_length=padding.PSS.MAX_LENGTH
                    ),
                    hashes.SHA256()
                )
                
                # Check timestamp
                protection_time = datetime.fromisoformat(protection_info['timestamp'])
                time_since_protection = datetime.now() - protection_time
                if time_since_protection > timedelta(days=30):
                    return f"Image signature is valid, but the protection is {time_since_protection.days} days old. Consider re-protecting the image."
                
                return f"Image signature is valid. The image is authentic and was protected {time_since_protection.days} days ago."
            except:
                return "Image signature is invalid. The image may have been tampered with."

        except Exception as e:
            logging.error(f"Error verifying image: {str(e)}", exc_info=True)
            return f"Failed to verify image: {str(e)}"

    def batch_process(self, image_paths, output_dir='protected_images_batch', **kwargs):
        os.makedirs(output_dir, exist_ok=True)
        total_images = len(image_paths)
        results = []
        for i, image_path in enumerate(image_paths):
            result = self.protect_image(image_path, output_dir, **kwargs)
            results.append(result)
            yield (i + 1) / total_images  # Yield progress
        return results

if __name__ == "__main__":
    protector = AdvancedImageProtector()
    
    # Protect a single image
    result = protector.protect_image("path/to/your/image.jpg")
    print(result)
    
    # Verify a protected image
    verification_result = protector.verify_image("path/to/protected_image.png")
    print(verification_result)
    
    # Batch process images
    image_paths = ["path/to/image1.jpg", "path/to/image2.png", "path/to/image3.jpeg"]
    for progress in protector.batch_process(image_paths):
        print(f"Progress: {progress * 100:.2f}%")
