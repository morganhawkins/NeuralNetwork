from setuptools import find_packages, setup


setup(
    name = 'nnlearn',
    packages = find_packages(include = ['nnlearn']),
    version = '1.0',
    description = 'numpy implementation of a neural network',
    author = 'Mo',
    license = 'MIT',

    install_requires = ["numpy"],
    setup_requires = ['pytest-runner'],
    tests_require = ['pytest'],
    test_suite = 'tests',

)

