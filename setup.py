from setuptools import setup, find_packages

setup(
    name="randgraph",
    version="0.1",
    description="A package for generating and drawing randomly generated graphs",
    author="Dominic Nidy",
    author_email="domnidy01@gmail.com",
    url="https://github.com/DomNidy/randgraph",
    packages=find_packages(),
    install_requires=["matplotlib", "networkx"],
)
