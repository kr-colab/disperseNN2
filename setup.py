import os
from setuptools import setup

# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(name='disperseNN2',
      version='4.0.5',
      description='neural net for estimating dispersal',
      long_description=read('README'),
      long_description_content_type="text/markdown",
      url='https://github.com/kr-colab/disperseNN2',
      author='Chris Smith',
      author_email='chriscs@uoregon.edu',
      license='MIT',
      packages=['disperseNN2',],
      # install_requires=['numpy',
      #                   'geopy',
      #                   'attrs',
      #                   'scikit-learn',
      #                   'msprime',
      #                   'tskit',
      #                   'utm',
      #                   'matplotlib'],
      scripts=['disperseNN2/disperseNN2',
               'disperseNN2/data_generation.py',
               'disperseNN2/read_input.py',
               'disperseNN2/check_params.py',
               'disperseNN2/process_input.py'],
      classifiers=[
          "Development Status :: 3 - Alpha",
          "Topic :: Utilities",
          "License :: OSI Approved :: BSD License",
      ],      
      #zip_safe=False,
      # extras_require={
      #     'dev': [],
      # },
      # setup_requires=[],
      # ext_modules=[]
)
