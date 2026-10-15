from setuptools import find_packages, setup

setup(
    name="basicsr",
    version="1.4.2",
    description="BasicSR architecture for Retinal Image Segmentation (STD-Net)",
    author="Nahil El Bezzari, Yassine Lazizi, Pascal Le, Anis Melaimi, and Nadir Nehili",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    install_requires=[
        "torch",
        "torchvision",
        "numpy<2",
        "opencv-python<5",
        "pillow",
        "scikit-learn",
        "scipy",
        "pandas",
        "matplotlib",
        "pyyaml",
        "tqdm",
    ],
)

