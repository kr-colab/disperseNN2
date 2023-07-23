from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name='disperseNN2',
      version='1.0.7',
      description='neural net for estimating dispersal',
      long_description=long_description,
      long_description_content_type="text/markdown",
      url='https://github.com/kr-colab/disperseNN2',
      author='Chris Smith',
      author_email='chriscs@uoregon.edu',
      license='MIT',
      packages=['disperseNN2'],
      install_requires=['tensorflow==2.11.0',
                        'gpustat',
                        'numpy',
                        'geopy',
                        'attrs',
                        'scikit-learn',
                        'msprime',
                        'tskit',
                        'utm',
                        'matplotlib'],
      scripts=['disperseNN2/disperseNN2',
               'disperseNN2/data_generation.py',
               'disperseNN2/read_input.py',
               'disperseNN2/check_params.py',
               'disperseNN2/process_input.py'],
      zip_safe=False,
      extras_require={
          'dev': [],
      },
      setup_requires=[],
      ext_modules=[]
)
