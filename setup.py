from setuptools import setup, find_packages

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="deep_rl",
    description="My RL package.",
    author="Quentin GALLOUÉDEC",
    author_email="gallouedec.quentin@gmail.com",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    version="0.0.0",
    install_requires=["gym", "torch", "numpy", "pyglet==1.5.11"],
)
