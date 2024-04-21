from setuptools import find_packages, setup

setup(
    name="dl_project",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "lightning",
        "torchvision",
        "ipykernel",
        "hydra-core",
        "tensorboard==2.15.1",
        "dropbox",
        "matplotlib",
    ],
)
