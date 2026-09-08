from setuptools import setup, find_packages

setup(
    name='stable-audio-tools',
    version='0.0.16',
    url='https://github.com/RoyalCities/RC-stable-audio-tools.git',
    author='Stability AI',
    description='Training and inference tools for generative audio models from Stability AI',
    packages=find_packages(),
    install_requires=[
        'aeiou<=0.0.20',
        'alias-free-torch<=0.0.6',
        'auraloss<=0.4.0',
        'descript-audio-codec<=1.0.0',
        'einops<=0.7.0',
        'einops-exts<=0.0.4',
        'ema-pytorch<=0.2.3',
        'encodec<=0.1.1',

        'gradio==6.5.1',

        'huggingface_hub',
        'importlib-resources<=5.12.0',
        'k-diffusion<=0.1.1',
        'laion-clap<=1.1.4',
        'librosa<=0.10.0',
        'local-attention<=1.8.6',
        'matplotlib<=3.7.1',
        'pandas<=2.0.2',
        'pedalboard<=0.7.4',
        'prefigure<=0.0.9',
        'pretty_midi<=0.2.9',
        'pytorch_lightning<=2.1.0',
        'pydub<=0.25.1',
        'PyWavelets<=1.4.1',
        'safetensors',
        'sentencepiece<=0.1.99',
        's3fs',
        'soxr',
        'scipy==1.8.1',

        # Required by older librosa/pkg_resources stack
        'setuptools<81',

        # Tested against PyTorch 2.5.1.
        # Platform-specific CUDA/MPS installation is handled in the README.
        'torch==2.5.1',
        'torchaudio==2.5.1',

        'torchmetrics<=0.11.4',
        'tqdm',

        # Tested text-conditioning stack
        'transformers==5.0.0',

        'v-diffusion-pytorch<=0.0.2',
        'vector-quantize-pytorch<=1.9.14',
        'wandb<=0.15.4',
        'webdataset<=0.2.48',
        'x-transformers<1.27.0',

        # MIDI extraction / UI additions
        'basic_pitch==0.4.0',
        'hffs',
    ],
)