from setuptools import setup, find_packages

setup(
    name="xrfm",
    version="0.1.0",
    description="xRFM library",
    author="Daniel Beaglehole, David Holzmüller",
    author_email="dbeaglehole@ucsd.edu",
    packages=find_packages(),
    install_requires=[
        "torch>=1.8.0",
        "numpy>=1.19.0",
        "scikit-learn>=0.24.0",
        "torchmetrics>=0.6.0",
        "tqdm>=4.60.0",
    ],
    python_requires=">=3.7",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    keywords="machine learning, feature machine, recursive, tree-based",
) 