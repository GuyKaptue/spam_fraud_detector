# setup.py
# type: ignore
from setuptools import setup, find_packages
from pathlib import Path

# Load README for long description
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="spam-fraud-detector",
    version="0.1.0",
    author="Guy Kaptue",
    author_email="guyKaptue24@gmail.com",
    description="Modular benchmarking framework for binary classification tasks (spam, fraud, etc.)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/GuyKaptue/spam-fraud-detector",
    packages=find_packages(include=["spam_fraud_detector", "spam_fraud_detector.*"]),
    include_package_data=True,
    python_requires=">=3.8",
    install_requires=[
        "pandas",
        "numpy",
        "scikit-learn",
        "matplotlib",
        "seaborn",
        "xgboost",
        "lightgbm",
        "mlflow",
        "kagglehub",
        "openpyxl",
        "dash",
        "plotly",
        "jinja2",
        "streamlit"
    ],
    extras_require={
        "viz": ["matplotlib", "seaborn"],
        "ml": ["xgboost", "lightgbm"],
        "tracking": ["mlflow"],
        "dev": ["pytest", "tox", "black", "flake8"]
    },
    entry_points={
        "console_scripts": [
            "run-detector=spam_fraud_detector.main:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="spam fraud detection classification benchmarking machine-learning",
)
