from setuptools import setup, find_packages

setup(
    name="PRESTO",
    version='0.0.1',
    setup_requires=['pytest-runner'],
    tests_require=['pytest', 'pytest-cov', 'pytest-mock'],
    install_requires=['elliptic'],
    packages=find_packages(),
    license='LICENSE'
)
