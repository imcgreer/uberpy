#!/usr/bin/env python

from distutils.core import setup, Command

try:  # Python 3.x
    from distutils.command.build_py import build_py_2to3 as build_py
except ImportError:  # Python 2.x
    from distutils.command.build_py import build_py

setup(name='uberpy',
      version='0.1.1',
      description='Ubercalibration in python',
      author='Ian McGreer',
      author_email='imcgreer@gmail.com,',
      license='GPL',
      url='http://github.com/imcgreer/rapala/uberpy',
      packages=['uberpy'],
      provides=['uberpy'],
      requires=['numpy', 'matplotlib','scipy','astropy'],
      keywords=['Scientific/Engineering'],
     )

