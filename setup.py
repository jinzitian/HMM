# -*- coding: utf-8 -*-

from setuptools import setup
from setuptools import find_packages
import hmm

with open('README.rst') as f:
    readme = f.read()


setup(
    name='hmm_tool',
    version=hmm.__version__,
    packages=['hmm'],
    author='Chef_J',
    author_email='hedge_jzt@hotmail.com',
    maintainer='Chef_J',
    maintainer_email='hedge_jzt@hotmail.com',
    description='A fast Python implementation of HMM.',
    long_description=readme,
    long_description_content_type='text/markdown',
    license='MIT',
    url='https://github.com/jinzitian/HMM',
    install_requires=[],
    classifiers=[
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Topic :: Software Development :: Libraries',
        ],

)