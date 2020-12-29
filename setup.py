#! /usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import io
import os
from setuptools import find_packages
from setuptools import setup


requirements = {
    'install': [
        'configargparse>=1.0',
        'editdistance>=0.5.3',
        'kaldiio>=2.15.1',
        'matplotlib>=3.1.2',
        'nltk>=3.4.5'
        'pandas>=1.0.0',
        'pyyaml>=5.3',
        'seaborn>=0.10.0',
        'sentencepiece>= 0.1.85',
        'setproctitle>=1.1.10',
        'tensorboardX>=2.0',
        'tqdm>=4.42.0',
    ],
    'setup': [

    ],
    'test': [
        "pytest>=4.6",
        "pytest-cov",
        "pycodestyle",
    ],
    'doc': [

    ]
}
install_requires = requirements['install']
setup_requires = requirements['setup']
tests_require = requirements['test']
extras_require = {k: v for k, v in requirements.items()
                  if k not in ['install', 'setup']}

dirname = os.path.dirname(__file__)
setup(name='neural_sp',
      version='0.1.0',
      url='http://github.com/neural_sp/neural_sp',
      author='Hirofumi Inaguma',
      author_email='hiro.mhbc@gmail.com',
      description='NeuralSP: Neural network based end-to-end Speech Processing toolkit',
      long_description=io.open(os.path.join(dirname, 'README.md'),
                               encoding='utf-8').read(),
      license='Apache Software License',
      packages=find_packages(include=['neural_sp*']),
      install_requires=install_requires,
      setup_requires=setup_requires,
      tests_require=tests_require,
      extras_require=extras_require,
      classifiers=[
          'Programming Language :: Python',
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 3.7',
          'Development Status :: 5 - Production/Stable',
          'Intended Audience :: Science/Research',
          'Operating System :: POSIX :: Linux',
          'License :: OSI Approved :: Apache Software License',
          'Topic :: Software Development :: Libraries :: Python Modules'],
      )
