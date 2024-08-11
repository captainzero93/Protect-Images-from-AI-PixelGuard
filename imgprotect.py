import numpy as np
import cv2
from PIL import Image
import os
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog, messagebox, IntVar
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding, rsa
import base64
import json
import logging
import hashlib
import piexif
from concurrent.futures import ThreadPoolExecutor
from scipy.fftpack import dct, idct
import pywt
import qrcode

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class AdvancedImageProtector:
    def __init__(self):
        self.private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048
        )
        self.public_key = self.private_key.public_key()
        self.supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}

    def protect_image(self, image_path, output_dir='protected_images'):
        try:
            os.makedirs(output_dir, exist_ok=True)
            logging.debug(f"Processing image: {image_path}")

            file_extension = os.path.splitext(image_path)[1].lower()
            if file_extension not in self.supported_formats:
                return f"Unsupported file format: {file_extension}"

            # Read image
            image = cv2.imread(image_path)
            
            # Apply multiple protection techniques
            protected_image = self.apply_dct_watermark(image)
            protected_image = self.apply_wavelet_watermark(protected_image)
            protected_image = self.apply_fourier_watermark(protected_image)
            protected_image = self.apply_adversarial_perturbation(protected_image)
            protected_image = self.apply_color_jittering(protected_image)
            protected_image = self.apply_invisible_qr(protected_image)
            protected_image = self.apply_steganography(protected_image)

            # Generate signature and hash AFTER all protections are applied
            image_bytes = cv2.imencode('.png', protected_image)[1].tobytes()
            image_hash = hashlib.sha256(image_bytes).hexdigest()
            signature = self.sign_image(image_bytes)

            # Prepare protection info
            protection_info = {
                "signature": signature,
                "image_hash": image_hash
            }

            # Save the protected image
            final_image_path = os.path.join(output_dir, f'protected_{os.path.basename(image_path)}')
            
            # Convert back to PIL Image for saving with EXIF
            pil_image = Image.fromarray(cv2.cvtColor(protected_image, cv2.COLOR_BGR2RGB))
            
            # Embed protection info in EXIF
            exif_dict = {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}, "thumbnail": None}
            exif_dict["0th"][piexif.ImageIFD.ImageDescription] = json.dumps(protection_info)
            exif_bytes = piexif.dump(exif_dict)

            pil_image.save(final_image_path, exif=exif_bytes)

            logging.debug(f"Saved protected image with embedded info: {final_image_path}")

            return f"Image processing complete. Protected image saved as {final_image_path}"
        except Exception as e:
            logging.error(f"Error processing image: {str(e)}", exc_info=True)
            return f"Error processing image {image_path}: {str(e)}"

    def apply_dct_watermark(self, image):
        logging.debug("Applying DCT watermark")
        blue_channel = image[:,:,0].astype(float)
        dct_blue = dct(dct(blue_channel.T, norm='ortho').T, norm='ortho')
        
        np.random.seed(42)
        watermark = np.random.normal(0, 2, blue_channel.shape)
        
        alpha = 0.05  # Reduced from 0.1
        dct_blue += alpha * watermark
        
        blue_channel_watermarked = idct(idct(dct_blue.T, norm='ortho').T, norm='ortho')
        image[:,:,0] = np.clip(blue_channel_watermarked, 0, 255).astype(np.uint8)
        
        return image

    def apply_wavelet_watermark(self, image):
        logging.debug("Applying wavelet watermark")
        green_channel = image[:,:,1].astype(float)
        coeffs = pywt.dwt2(green_channel, 'haar')
        cA, (cH, cV, cD) = coeffs
        
        np.random.seed(24)
        watermark = np.random.normal(0, 1, cA.shape)
        
        alpha = 0.05  # Reduced from 0.1
        cA += alpha * watermark
        
        green_channel_watermarked = pywt.idwt2((cA, (cH, cV, cD)), 'haar')
        image[:,:,1] = np.clip(green_channel_watermarked, 0, 255).astype(np.uint8)
        
        return image

    def apply_fourier_watermark(self, image):
        logging.debug("Applying Fourier watermark")
        red_channel = image[:,:,2].astype(float)
        f_transform = np.fft.fft2(red_channel)
        
        np.random.seed(36)
        watermark = np.random.normal(0, 1, f_transform.shape)
        
        alpha = 0.05  # Reduced from 0.1
        f_transform += alpha * watermark
        
        red_channel_watermarked = np.fft.ifft2(f_transform).real
        image[:,:,2] = np.clip(red_channel_watermarked, 0, 255).astype(np.uint8)
        
        return image

    def apply_adversarial_perturbation(self, image):
        logging.debug("Applying adversarial perturbation")
        epsilon = 1.0  # Reduced from 2.0
        
        np.random.seed(48)
        perturbation = np.random.normal(0, 1, image.shape).astype(np.float32)
        
        # Normalize perturbation
        perturbation = epsilon * perturbation / np.linalg.norm(perturbation)
        
        # Apply perturbation
        image_perturbed = image.astype(np.float32) + perturbation
        image_perturbed = np.clip(image_perturbed, 0, 255).astype(np.uint8)
        
        return image_perturbed

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

    def apply_invisible_qr(self, image):
        logging.debug("Applying invisible QR code")
        qr = qrcode.QRCode(version=1, box_size=10, border=5)
        qr.add_data("Protected Image")
        qr.make(fit=True)
        qr_image = qr.make_image(fill_color="black", back_color="white")
        qr_array = np.array(qr_image.convert('L'))
        qr_array = cv2.resize(qr_array, (image.shape[1], image.shape[0]))
        
        # Convert QR code to float and normalize
        qr_float = qr_array.astype(np.float32) / 255.0
        
        # Apply QR code with very low opacity
        alpha = 0.05  # Adjust this value to make QR code more or less visible
        image_float = image.astype(np.float32) / 255.0
        image_with_qr = image_float * (1 - alpha * qr_float[:,:,np.newaxis]) + alpha * qr_float[:,:,np.newaxis]
        
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
                exif_dict = piexif.load(img.info.get("exif", b""))
                protection_info = json.loads(exif_dict["0th"].get(piexif.ImageIFD.ImageDescription, "{}"))

            if not protection_info:
                return "This image does not contain protection information."

            image = cv2.imread(image_path)
            image_bytes = cv2.imencode('.png', image)[1].tobytes()
            current_hash = hashlib.sha256(image_bytes).hexdigest()

            logging.debug(f"Current image hash: {current_hash}")
            logging.debug(f"Stored image hash: {protection_info['image_hash']}")

            if current_hash != protection_info['image_hash']:
                return "Image hash mismatch. The image may have been altered."

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
                return "Image signature is valid. The image is authentic."
            except:
                return "Image signature is invalid. The image may have been tampered with."

        except Exception as e:
            logging.error(f"Error verifying image: {str(e)}", exc_info=True)
            return f"Failed to verify image: {str(e)}"

    def batch_process(self, image_paths, output_dir='protected_images_batch'):
        os.makedirs(output_dir, exist_ok=True)
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.protect_image, image_path, output_dir) for image_path in image_paths]
            results = [future.result() for future in futures]
        return results

class AdvancedImageProtectorGUI:
    def __init__(self, master):
        self.master = master
        self.master.title("Advanced Image Protector")
        self.protector = AdvancedImageProtector()

        tk.Button(master, text="Protect Single Image", command=self.protect_single_image).grid(row=0, column=0, pady=10)
        tk.Button(master, text="Batch Protect Images", command=self.batch_protect_images).grid(row=1, column=0, pady=10)
        tk.Button(master, text="Verify Image", command=self.verify_image).grid(row=2, column=0, pady=10)

        self.progress_var = IntVar()
        self.progress_bar = ttk.Progressbar(master, variable=self.progress_var, maximum=100)
        self.progress_bar.grid(row=3, column=0, pady=10, sticky="ew")

    def protect_single_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.webp")])
        if file_path:
            output_dir = filedialog.askdirectory(title="Select Output Directory")
            if output_dir:
                self.progress_var.set(0)
                self.master.update_idletasks()
                result = self.protector.protect_image(file_path, output_dir)
                self.progress_var.set(100)
                messagebox.showinfo("Result", result)

    def batch_protect_images(self):
        file_paths = filedialog.askopenfilenames(filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.webp")])
        if file_paths:
            output_dir = filedialog.askdirectory(title="Select Output Directory")
            if output_dir:
                self.progress_var.set(0)
                self.master.update_idletasks()
                results = self.protector.batch_process(file_paths, output_dir)
                self.progress_var.set(100)
                messagebox.showinfo("Batch Result", "\n".join(results))

    def verify_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.webp")])
        if file_path:
            result = self.protector.verify_image(file_path)
            messagebox.showinfo("Verification Result", result)

if __name__ == "__main__":
    root = tk.Tk()
    app = AdvancedImageProtectorGUI(root)
    root.mainloop()
