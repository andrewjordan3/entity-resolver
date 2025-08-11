from setuptools import setup, find_packages

# Read the contents of your README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read the contents of your requirements file
with open("requirements.txt", "r", encoding="utf-8") as f:
    # Filter out comments and empty lines
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name='entity-resolver',
    version='0.1.0',
    author='Andrew Jordan',
    author_email='andrewjordan3@gmail.com',
    description='A high-performance, GPU-accelerated entity resolution pipeline.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/andrewjordan3/entity-resolver', 
    packages=find_packages(),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    python_requires='>=3.8',
)