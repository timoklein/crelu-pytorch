from setuptools import setup, find_packages

setup(
    name="crelu",
    packages=find_packages(),
    version="0.0.1",
    license="MIT",
    description="CReLU activation",
    author="Timo Klein",
    url="https://github.com/timoklein/crelu-pytorch",
    keywords=[
        "artificial intelligence",
        "deep learning",
        "activation function",
    ],
    install_requires=["torch>=1.12"],
)
