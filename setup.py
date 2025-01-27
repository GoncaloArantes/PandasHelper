from setuptools import setup, find_packages
from os import path
working_directory = path.abspath(path.dirname(__file__))

with open(path.join(working_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='Automation_Pandas_Library',
    version='0.0.1',
    url='https://github.com/Gonçalo/PandasHelper',
    author='Gonçalo Arantes',
    author_email='goncaloarantes18@gmail.com',
    description='Helper for Pandas Dataframe Handling',
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=find_packages(),
    classifiers=[
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: MIT License",
            "Operating System :: OS Independent"
    ],
    install_requires=[
            "numpy>=1.26.3",              
            "pandas>=2.2.3",
            "scikit-learn>=1.6.0"
    ],
)