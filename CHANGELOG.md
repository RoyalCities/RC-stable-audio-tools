# Changelog

## [1.1.1] - 2024-11-16
###Added 

- Support for non-16bit audio input in gradio interface

## [1.1.0] - 2024-09-10
### Added
- Implemented support for loading 16-bit (FP16) models directly - FP16 checkpoints can be loaded or unloaded along with 32-bit (FP32) models with correct inferencing.
- Changed text in Init Audio header to highlight the need for 16-bit WAV files and model compatibility.

### Fixed
- Optimized code to save VRAM after each audio generation + some resource management.
