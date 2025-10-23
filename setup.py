from setuptools import setup, find_packages

setup(
    name="pynlu",
    version="1.0.0",
    description="Lightweight Python NLU library for intent recognition and slot extraction",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Paul Wolf",
    packages=find_packages(),  # automatically finds 'pynlu'
    python_requires=">=3.10",
    install_requires=[
        "scikit-learn",
        "spacy",
        "langdetect",
        "pyyaml",
        "joblib"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
