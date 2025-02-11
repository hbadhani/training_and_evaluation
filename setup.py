from setuptools import setup, find_packages

setup(
    name='training_and_evaluation',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'torch',
        'matplotlib',
    ],
    description='A package for training and evaluating models',
    author='Hemant Badhani',
    author_email='hbadhani@adobe.com',
)
