import setuptools

with open('README.md', 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name='birg_chemometrics_tools',
    version='20.01',
    author='BiRG @ Wright State University',
    author_email='foose.3@wright.edu',
    description='Chemometric Tools for NMR Metabolomics',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/BiRG/birg_chemometrics_tools',
    packages=setuptools.find_packages(),
    python_requires='>=3.5',
    instal_requires=[
        'numpy>=1.11.0',
        'scipy>=0.18.0',
        'scikit-learn>=0.18.0',
        'eli5'
    ],
    classifiers=[
        'Natural Language :: English',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)
