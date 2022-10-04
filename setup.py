from os import path

import setuptools

path_to_repo = path.abspath(path.dirname(__file__))
with open(path.join(path_to_repo, 'readme.md'), encoding='utf-8') as f:
    long_description = f.read()

required_pypi = [
    'transformers[torch]',
    'numpy',
    'datasets',
    'scikit-learn',
    'pandas',
    'spacy',
    'torch',
    'tqdm',
]

setuptools.setup(
    name="embgam",
    version="0.3",
    author="Chandan Singh",
    author_email="chansingh@microsoft.com",
    description="Emb-GAM: an Interpretable and Efficient Predictor using Pre-trained Language Models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/csinva/emb-gam",
    packages=setuptools.find_packages(
        exclude=['tests', 'tests.*', '*.test.*']
    ),
    install_requires=required_pypi,
    python_requires='>=3.6',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
