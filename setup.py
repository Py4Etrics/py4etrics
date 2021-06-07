from setuptools import setup, find_packages

setup(name='py4etrics',
      version='0.1.6',
      author='Tetsu HARUYAMA',
      author_email='haruyama@econ.kobe-u.ac.jp',
      packages=find_packages(),
      install_requires=['statsmodels'],
      package_dir={'py4etrics': './py4etrics'},
      url='https://github.com/Py4Etrics/py4etrics',
      license='MIT',
      description='A package for for py4etrics.github.io',
      long_description=open('README.md').read(),
      long_description_content_type='text/markdown',
      classifiers=['Intended Audience :: Education',
                   'Intended Audience :: Science/Research',
                   'License :: OSI Approved :: MIT License',
                   'Operating System :: OS Independent',
                   'Programming Language :: Python',
                   ],
      keywords=['Truncated Regression', 'Tobit Model', 'Heckit Model']
      )



