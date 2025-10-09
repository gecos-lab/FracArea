# Template python project
This folder is a base template for python projects developped by Bicocca.

## Best practice of code
A clean code is easy to read, to understand, and to maintain... Prioritizes clarity over cleverness.
* Use the single responsibility principle.
* Use self documenting naming.
* Avoid code duplication.

## Software engineering toolbox
The project use:
* Black for code format
* Pylint for code analysis (note that the actual pylintrc will be use in every child repository throug github actions).

## Automation and pipeline
The main branch is protected and can be changed only by merging a branch after a pull-request.

A pull request will trigger the prepare_pr workflow:
1. ensure the code formatting
1. analyse the code with pylint

Pushing changing in the repository will trigger the CICD and run the test.

A merge of a pull request will use semantic release to increment and eventually create a new version on the github repository.

Semantic release use commit comments that follow the following following convention:
* fix(Title): detail of the fix --> increase the minor version digit. 
* feat(Title): detail of the feature --> increase the mid version digit. 

The two previous message can be followed (Leave a blank line) by:
BREAKING CHANGE: description of the breaking change. In order to increase the major version digit.

see https://github.com/semantic-release/semantic-release for more information.

## dependencies ... and installation
The 'requirements.txt' can be use to install project dependencies either:
* using conda: "conda install --file requirements.txt" 
* using pip: "pip install -r requirements.txt"

It is strongly recommended to use a virtual environement that can be created with the "venv" module (python -m venv venv_name) and activated with the "activate" script in the folder "venv_name/Scripts".

You can also directly use pycharm (or another IDE) to create a virtual environement from the file "requirements.txt".

DO NOT EDIT the "requirements.txt" it should be automatically created from the file "requirements.in".
1. create and activate a venv
1. activate the venv
1. install the pip-tools module
1. run the pip-compile command in the same folder as the file "requirements.in".

see https://pypi.org/project/pip-tools/2.0.1/ for more information.

