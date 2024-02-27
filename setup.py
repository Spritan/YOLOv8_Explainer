from setuptools import find_packages, setup

with open("./README.md", "r") as f:
    long_description = f.read()

setup(
    name="YOLOv8_Explainer",
    version="0.0.03",
    description="Python packages that enable XAI methods for YOLOv8",
    packages=find_packages(),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Spritan/YOLOv8_Explainer",
    author="Spritan",
    author_email="proypabsab@gmail.com",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Education",
        "Programming Language :: Python",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "grad-cam==1.4.8",
        "ultralytics",
        "Pillow",
        "tqdm",
        "torch",
        "matplotlib",
    ],
    extras_require={
        "dev": ["twine>=4.0.2"],
    },
    # python_requires=">=3.10",
)
