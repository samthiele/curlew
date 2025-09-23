from setuptools import setup
import setuptools
from pathlib import Path

this_dir = Path(__file__).parent
long_description = (this_dir / "README.md").read_text() # read README file as long description

setup(
    name='curlew',
    version='1.0',
    packages=setuptools.find_packages(),
    url='https://github.com/samthiele/curlew',
    license='MIT',
    author='Sam Thiele',
    author_email='s.thiele@hzdr.de',
    description='A python package for constructing complex geological models from various types of neural fields.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    include_package_data=True,
    install_requires=['numpy', 'torch', 'tqdm'],
    package_data = {"":["*.png","*.ttl"]}
)
