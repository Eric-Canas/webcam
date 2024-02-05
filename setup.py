from setuptools import setup, find_packages

setup(
    name='webcam',
    version='1.34',
    author='Eric-Canas',
    author_email='eric@ericcanas.com',
    url='https://github.com/Eric-Canas/webcam',
    description='A simple and convenient library to interact with webcams in Python without having to address Hardware Limitations',

    long_description=open('README.md', 'r').read(),
    long_description_content_type='text/markdown',
    license='MIT',
    packages=find_packages(),
    python_requires='>=3.7',
    install_requires=[
        'numpy',
        'opencv-python',
        'imutils'
    ],

    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
        'Intended Audience :: Developers',
        'Topic :: Multimedia :: Graphics',
        'Topic :: Multimedia :: Graphics :: Capture',
        'Topic :: Multimedia :: Graphics :: Capture :: Digital Camera',
        'Topic :: Multimedia :: Graphics :: Capture :: Screen Capture',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
)
