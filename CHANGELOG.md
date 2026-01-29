# Changelog

## [1.3] - 2026-01-28
### Added
- TorchAO Quantization support (Windows / Linux)
- Improved Gradio workflow
- Synced with upstream Stable Audio Tools

## [1.2] - 2024-12-04
### Added
- Better random prompt managment.
- Key Signatures are now tied to UI rather than prompt and can be locked or unlocked similiar to BPM.

## [1.1.2] - 2024-11-16
### Added

- updated autoencoder so 16 bit models should now be able to do init audio / AI style transfers.

## [1.1.1] - 2024-11-16
### Added 

- Support for non-16bit audio input in gradio interface

## [1.1.0] - 2024-09-10
### Added
- Implemented support for loading 16-bit (FP16) models directly - FP16 checkpoints can be loaded or unloaded along with 32-bit (FP32) models with correct inferencing.
- Changed text in Init Audio header to highlight the need for 16-bit WAV files and model compatibility.

### Fixed
- Optimized code to save VRAM after each audio generation + some resource management.
