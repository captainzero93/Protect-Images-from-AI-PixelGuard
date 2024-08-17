# UNet Extractor and Remover for Stable Diffusion 1.5 and SDXL

This Python script processes SafeTensors files for Stable Diffusion 1.5 (SD 1.5) and Stable Diffusion XL (SDXL) models. It extracts the UNet into a separate file and creates a new file with the remaining model components (without the UNet).

## Features

- Supports both SD 1.5 and SDXL model architectures
- Extracts UNet tensors from SafeTensors files
- Creates a separate SafeTensors file with non-UNet components
- Saves the extracted UNet as a new SafeTensors file
- Command-line interface for easy use

## Requirements

- Python 3.6+
- safetensors library

## Installation

1. Clone this repository or download the `unet_extractor.py` script.

2. Install the required `safetensors` library:

   ```
   pip install safetensors
   ```

## Usage

Run the script from the command line with the following syntax:

```
python unet_extractor.py <input_file> <unet_output_file> <non_unet_output_file> --model_type <sd15|sdxl>
```

### Arguments

- `<input_file>`: Path to the input SafeTensors file (full model)
- `<unet_output_file>`: Path where the extracted UNet will be saved
- `<non_unet_output_file>`: Path where the model without UNet will be saved
- `--model_type`: Specify the model type, either `sd15` for Stable Diffusion 1.5 or `sdxl` for Stable Diffusion XL

### Examples

For Stable Diffusion 1.5:
```
python unet_extractor.py path/to/sd15_model.safetensors path/to/output_sd15_unet.safetensors path/to/output_sd15_non_unet.safetensors --model_type sd15
```

For Stable Diffusion XL:
```
python unet_extractor.py path/to/sdxl_model.safetensors path/to/output_sdxl_unet.safetensors path/to/output_sdxl_non_unet.safetensors --model_type sdxl
```

## How It Works

1. The script opens the input SafeTensors file using the `safetensors` library.
2. It iterates through all tensors in the file, separating UNet-related tensors from other tensors.
3. For SD 1.5, it removes the "model.diffusion_model." prefix from UNet tensor keys.
4. For SDXL, it keeps the original key names for both UNet and non-UNet tensors.
5. The extracted UNet tensors are saved to a new SafeTensors file.
6. The remaining non-UNet tensors are saved to a separate SafeTensors file.

## Notes

- Ensure you have sufficient disk space to save both output files.
- The script processes the tensors in CPU memory, so it should work even on machines without a GPU.
- Processing large models may take some time, depending on your system's performance.

## Troubleshooting

If you encounter any issues:

1. Ensure you have the latest version of the `safetensors` library installed.
2. Check that your input file is a valid SafeTensors file for the specified model type.
3. Make sure you have read permissions for the input file and write permissions for the output directory.

## Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/captainzero93/unet-extractor/issues) if you want to contribute.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Citation

If you use UNet Extractor and Remover in your research or projects, please cite it as follows:

```
[Joe Faulkner] (captainzero93). (2024). UNet Extractor and Remover for Stable Diffusion 1.5 and SDXL. GitHub. https://github.com/captainzero93/unet-extractor
```

## Acknowledgements

- This script uses the `safetensors` library developed by the Hugging Face team.
- Inspired by the Stable Diffusion and SDXL projects from Stability AI.
