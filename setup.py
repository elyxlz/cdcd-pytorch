from setuptools import find_packages, setup

setup(
    name="cdcd-pytorch",
    packages=find_packages(exclude=[]),
    version="0.0.01",
    description="Continuous Diffusion for Categorical Data - PyTorch",
    long_description_content_type="text/markdown",
    author="Elio Pascarelli",
    author_email="elio@pascarelli.com",
    url="https://github.com/elyxlz/cdcd-pytorch",
    keywords=["artificial intelligence", "deep learning", "generative models"],
    install_requires=[
        "tqdm",
        "torch>=2.0",
        "torchaudio",
        "data-science-types>=0.2",
        "einops>=0.6",
        "diffusers>=0.1.0",
        "torchtyping",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3.9",
    ],
)