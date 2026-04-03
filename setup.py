from setuptools import setup, find_packages

setup(
    name="sovereign-allocator",
    version="0.1.0",
    description="Multi-strategy dynamic portfolio allocator: TCN + Graph Diffusion + ETF Shock",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.26",
        "scipy>=1.12",
        "pyyaml>=6.0",
    ],
    extras_require={
        "dev": ["pytest>=8.0"],
    },
)
