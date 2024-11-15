#!/usr/bin/env python

from setuptools import setup

install_requires=[
   # 'numpy',
   # 'matplotlib',
   # 'pandas',
]

setup(name='QITSDE',
      version='0.1',
      description='QITSDE',
      install_requires=install_requires,
      author='Samuel Adler',
      author_email='samadler85@gmail.com',
      packages=['amcc'],
      py_modules=['amcc.instruments', 'amcc.standard_measurements', 'amcc.utilities'],
      package_dir = {'amcc': 'amcc'},
     )
