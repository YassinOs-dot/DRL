from setuptools import setup, find_packages

setup(
    name="drl_welding",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "abb-robot-client",
        "stable-baselines3",
        "gymnasium",
        "numpy",
        "pyyaml",
        "torch",
        "tensorboard",
        "matplotlib",
        "scipy",
    ],
    author="Yassine",
    description="DRL for ABB welding robot collision-free trajectory learning",
)
