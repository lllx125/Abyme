from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="abyme",
    version="0.1.0",
    author="Abyme Research Team",
    author_email="contact@abyme.ai",
    description="Recursive Language Model with XML-based elaboration system for advanced reasoning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Abyme/abyme",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    install_requires=[
        "openai>=1.0.0",
        "python-dotenv>=1.2.2",
        "tqdm>=4.65.0",
    ],
    extras_require={
        "pytorch": [
            "torch>=2.0.0",
            "transformers>=4.35.0",
            "accelerate>=0.25.0",
            "bitsandbytes>=0.41.0",
        ],
        "sft": [
            "torch>=2.0.0",
            "transformers>=4.35.0",
            "accelerate>=0.25.0",
            "bitsandbytes>=0.41.0",
            "datasets>=2.14.0",
            "pandas>=2.0.0",
            "scipy>=1.10.0",
            "trl>=0.7.0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
    },
    python_requires=">=3.8",
    keywords="language-model recursive-reasoning llm transformers ai machine-learning",
    project_urls={
        "Bug Reports": "https://github.com/Abyme/abyme/issues",
        "Source": "https://github.com/Abyme/abyme",
        "Documentation": "https://github.com/Abyme/abyme#readme",
    },
)
