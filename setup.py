#!/usr/bin/env python

from setuptools import setup

install_requires=[
   # 'numpy',
   # 'matplotlib',
   # 'pandas',
]

setup(name='amcc',
      version='1.0.1',
      description='PHIDL',
      install_requires=install_requires,
      author='Adam McCaughan',
      author_email='amccaugh@gmail.com',
      packages=['amcc'],
      py_modules=['amcc.instruments', 'amcc.standard_measurements', 'amcc.utilities'],
      package_dir = {'amcc': 'amcc'},
     )
