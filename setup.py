from setuptools import setup


def readme():
    with open('README.md') as f:
        return f.read()


setup(name='irisSeg',
      version='0.1',
      description='Daugman implementation to segement iris and pupil',
      long_description=readme(),
      classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
      ],
      keywords="Daugman Daugman's integrodifferential operator iris pupil segementation computer vision",
      url='http://github.com/mzmmoazam/',
      author='mzm',
      author_email='mzm.moazam@gmail.com',
      license='MIT',
      long_description_content_type='text/markdown',
      packages=['irisSeg'],
      install_requires=[
          'numpy',
          'opencv-python',
          'scipy',
          'scikit-image',
          'matplotlib'
      ],
      test_suite='nose.collector',
      tests_require=['nose', 'nose-cover3'],
      entry_points={
          'console_scripts': ['iris-seg=irisSeg.command_line:main'],
      },
      include_package_data=True,
      zip_safe=False)