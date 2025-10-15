# setup.py
# type: ignore
from setuptools import setup, find_packages

setup(
    name="spam-fraud-detector",
    version="0.1.0",
    author="Guy Kaptue",
    description="Modular benchmarking framework for binary classification tasks (spam, fraud, etc.)",
    url="https://github.com/GuyKaptue/spam-fraud-detector", 
    packages=find_packages(include=['spam_fraud_detector', 'spam_fraud_detector.*']),
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
        "openpyxl"
    ],
    python_requires=">=3.8",
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "License :: OSI Approved :: MIT License",
    ],
    # Console entry point for running the main script
    entry_points={
        'console_scripts': [
            'run-detector=spam_fraud_detector.main:main',
        ],
    },
)