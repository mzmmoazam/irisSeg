from setuptools import setup
try:  # for pip >= 10
    from pip._internal.req import parse_requirements
    from pip._internal.download import PipSession
except ImportError:  # for pip <= 9.0.3
    from pip.req import parse_requirements
    from pip.download import PipSession
    
    
def readme():
    with open('README.md') as f:
        return f.read()
requirements = parse_requirements(os.path.join(os.path.dirname(__file__), 'requirements.txt'), session=PipSession())


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
      install_requires=[str(requirement.requirement) for requirement in requirements],
      test_suite='nose.collector',
      tests_require=['nose', 'nose-cover3'],
      entry_points={
          'console_scripts': ['iris-seg=irisSeg.command_line:main'],
      },
      include_package_data=True,
      zip_safe=False)