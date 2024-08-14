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
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.protect_image, image_path, output_dir, **kwargs) for image_path in image_paths]
            results = [future.result() for future in futures]
        return results

class AdvancedImageProtectorGUI:
    def __init__(self, master):
        self.master = master
        self.master.title("PixelGuard AI - Advanced Image Protector")
        self.protector = AdvancedImageProtector()

        # Create a frame for the sliders
        self.settings_frame = ttk.LabelFrame(master, text="Protection Settings")
        self.settings_frame.grid(row=0, column=0, columnspan=3, padx=10, pady=10, sticky="nsew")

        # DCT Watermark Strength
        ttk.Label(self.settings_frame, text="DCT Watermark Strength:").grid(row=0, column=0, sticky="w")
        self.dct_strength = DoubleVar(value=0.05)
        ttk.Scale(self.settings_frame, from_=0.01, to=0.1, variable=self.dct_strength, orient="horizontal").grid(row=0, column=1, sticky="ew")

        # Wavelet Watermark Strength
        ttk.Label(self.settings_frame, text="Wavelet Watermark Strength:").grid(row=1, column=0, sticky="w")
        self.wavelet_strength = DoubleVar(value=0.05)
        ttk.Scale(self.settings_frame, from_=0.01, to=0.1, variable=self.wavelet_strength, orient="horizontal").grid(row=1, column=1, sticky="ew")

        # Fourier Watermark Strength
        ttk.Label(self.settings_frame, text="Fourier Watermark Strength:").grid(row=2, column=0, sticky="w")
        self.fourier_strength = DoubleVar(value=0.05)
        ttk.Scale(self.settings_frame, from_=0.01, to=0.1, variable=self.fourier_strength, orient="horizontal").grid(row=2, column=1, sticky="ew")

        # Adversarial Perturbation Strength
        ttk.Label(self.settings_frame, text="Adversarial Perturbation:").grid(row=3, column=0, sticky="w")
        self.adversarial_strength = DoubleVar(value=0.01)
        ttk.Scale(self.settings_frame, from_=0.001, to=0.1, variable=self.adversarial_strength, orient="horizontal").grid(row=3, column=1, sticky="ew")

        # QR Code Opacity
        ttk.Label(self.settings_frame, text="QR Code Opacity:").grid(row=4, column=0, sticky="w")
        self.qr_opacity = DoubleVar(value=0.05)
        ttk.Scale(self.settings_frame, from_=0.01, to=0.1, variable=self.qr_opacity, orient="horizontal").grid(row=4, column=1, sticky="ew")

        # Preset buttons
        self.preset_frame = ttk.LabelFrame(master, text="Preset Settings")
        self.preset_frame.grid(row=1, column=0, columnspan=3, padx=10, pady=10, sticky="nsew")

        ttk.Button(self.preset_frame, text="Recommended", command=self.set_recommended).grid(row=0, column=0, padx=5, pady=5)
        ttk.Button(self.preset_frame, text="Lighter", command=self.set_lighter).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(self.preset_frame, text="Stronger", command=self.set_stronger).grid(row=0, column=2, padx=5, pady=5)

        # Action buttons
        ttk.Button(master, text="Protect Single Image", command=self.protect_single_image).grid(row=2, column=0, pady=10)
        ttk.Button(master, text="Batch Protect Images", command=self.batch_protect_images).grid(row=2, column=1, pady=10)
        ttk.Button(master, text="Verify Image", command=self.verify_image).grid(row=2, column=2, pady=10)

        # Progress bar
        self.progress_var = IntVar()
        self.progress_bar = ttk.Progressbar(master, variable=self.progress_var, maximum=100)
        self.progress_bar.grid(row=3, column=0, columnspan=3, pady=10, sticky="ew")

    def set_recommended(self):
        self.dct_strength.set(0.05)
        self.wavelet_strength.set(0.05)
        self.fourier_strength.set(0.05)
        self.adversarial_strength.set(0.01)
        self.qr_opacity.set(0.05)

    def set_lighter(self):
        self.dct_strength.set(0.03)
        self.wavelet_strength.set(0.03)
        self.fourier_strength.set(0.03)
        self.adversarial_strength.set(0.005)
        self.qr_opacity.set(0.03)

    def set_stronger(self):
        self.dct_strength.set(0.08)
        self.wavelet_strength.set(0.08)
        self.fourier_strength.set(0.08)
        self.adversarial_strength.set(0.02)
        self.qr_opacity.set(0.08)

    def protect_single_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.webp")])
        if file_path:
            output_dir = filedialog.askdirectory(title="Select Output Directory")
            if output_dir:
                self.progress_var.set(0)
                self.master.update_idletasks()
                result = self.protector.protect_image(
                    file_path, 
                    output_dir,
                    dct_strength=self.dct_strength.get(),
                    wavelet_strength=self.wavelet_strength.get(),
                    fourier_strength=self.fourier_strength.get(),
                    adversarial_strength=self.adversarial_strength.get(),
                    qr_opacity=self.qr_opacity.get()
                )
                self.progress_var.set(100)
                messagebox.showinfo("Result", result)

    def batch_protect_images(self):
        file_paths = filedialog.askopenfilenames(filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.webp")])
        if file_paths:
            output_dir = filedialog.askdirectory(title="Select Output Directory")
            if output_dir:
                self.progress_var.set(0)
                self.master.update_idletasks()
                results = self.protector.batch_process(
                    file_paths, 
                    output_dir,
                    dct_strength=self.dct_strength.get(),
                    wavelet_strength=self.wavelet_strength.get(),
                    fourier_strength=self.fourier_strength.get(),
                    adversarial_strength=self.adversarial_strength.get(),
                    qr_opacity=self.qr_opacity.get()
                )
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
