from os import path

import setuptools

path_to_repo = path.abspath(path.dirname(__file__))
with open(path.join(path_to_repo, 'readme.md'), encoding='utf-8') as f:
    long_description = f.read()

required_pypi = [
    'datasets',
    'dict_hash',
    'imodels',
    'langchain',
    'numpy',
    'pandas',
    'scikit-learn>=1.2.0',
    'scipy',
    'spacy',
    'torch',
    'tqdm',
    'transformers[torch] >= 4.23.1',
    'pyyaml',


    # 'InstructorEmbedding', # embeddings for emb_diff_module
    # 'sentence-transformers', # embeddings for emb_diff_module
    # pdoc3 # for generating docs
]

setuptools.setup(
    name="imodelsx",
    version="1.0.10",
    author="Chandan Singh, John X. Morris, Armin Askari, Divyanshu Aggarwal, Aliyah Hsu, Yuntian Deng",
    author_email="chansingh@microsoft.com",
    description="Library to explain a dataset in natural language.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/csinva/imodelsX",
    packages=setuptools.find_packages(
        exclude=['tests', 'tests.*', '*.test.*']
    ),
    install_requires=required_pypi,
    python_requires='>=3.9',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
