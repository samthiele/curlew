from setuptools import setup
import setuptools

setup(
    name='curlew',
    version='0.02',
    packages=setuptools.find_packages(),
    url='https://github.com/samthiele/curlew',
    license='',
    author='Sam Thiele',
    author_email='s.thiele@hzdr.de',
    description='A python package for constructing complex geological models from various types of neural fields.',
    include_package_data=True,
    install_requires=['numpy', 'torch', 'tqdm'],
    package_data = {"":["*.png","*.ttl"]}
)
