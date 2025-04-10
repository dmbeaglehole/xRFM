from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='tabrfm',
    version='1.0',
    author='Daniel Beaglehole',
    author_email='dbeaglehole@ucsd.edu',
    description='TabRFM - Recursive Feature Machines optimized for tabular data',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/dbeaglehole/tabrfm',
    project_urls = {
        "Bug Tracker": "https://github.com/dbeaglehole/tabrfm/issues"
    },
    license='MIT license',
    packages=find_packages(),
    install_requires=[
      'torch>=2.0',
      'tqdm'
    ],
)
