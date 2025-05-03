from setuptools import setup, find_packages

setup(
    name='sbb',
    version='0.1.0',
    packages=['sbb'],
    entry_points={
        'console_scripts': [
            'sbb = sbb.cli:main',
        ],
    },
    install_requires=[
        # Add any dependencies here, e.g., 'ply' if we use it for parsing
        'ply>=3.11',
    ],
    author='icarogamer2441', # Replace with your name
    description='A stack-based compiler backend.',
    long_description=open('README.md').read() if open('README.md') else '', # Create a README.md later
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/sbb', # Replace with your repo URL
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License', # Choose your license
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
)
