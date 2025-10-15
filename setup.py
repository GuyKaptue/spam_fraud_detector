# setup.py
# type: ignore
from setuptools import setup, find_packages

setup(
    name="spam-fraud-detector",  # Updated to match project directory
    version="0.1.0",
    author="Guy Kaptue",
    description="Modular benchmarking framework for binary classification tasks (spam, fraud, etc.)",
    url="https://github.com/GuyKaptue/classifier-bench", # Keep this if it's the correct repo
    packages=find_packages(include=['spam_fraud_detector', 'spam_fraud_detector.*']), # Explicitly include the main package
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
        "License :: OSI Approved :: MIT License", # A common addition
    ],
    # You might also want to add an entry point for your main script
    entry_points={
        'console_scripts': [
            'run-detector=spam_fraud_detector.main:main',
        ],
    },
)