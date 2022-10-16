# FAQ:

## Q: What python version should this project use?
3.9

## Q: Anything I should do first?
Install https://pre-commit.com/

## Q: Gitlab is rejecting my code push but there is no code conflict

A: This project uses a few packages to help us to produce high quality code
 - pytest is used for writing and running automated tests
 - pytest-cov generates code coverage reports
 - black looks at code formatting 
 - flake8 looks for PEP8 compliance
 - isort looks for import orders

Git checks for pytest, black, flake8 and isort before allowing a push to take place.
You can test these locally before submitting them.

Below are some examples of the codes for conda, if you're using Anaconda's distribution.
 
### Pytest
conda run pytest --cov="." # Run pytest
conda run pytest --cov="." --cov-report html # Export results as interactive html

### Flake8
conda run flake8 . # Run test

This code returns nothing if no problem is found, otherwise you should see something similar to...


PS D:\Git\churn> conda run flake8 .
ERROR conda.cli.main_run:execute(34): Subprocess for 'conda run ['flake8', '.']' command failed.  (See above for error)
.\project\main.py:15:11: W292 no newline at end of file

### Black
conda run black . --check 	# Checks if there is any formatting error
conda run black . --diff	# Shows changes that black would apply without actually applying
conda run black .			# Apply changes to formatting errors

### isort
conda run isort . --check --diff # Similar to black
conda run isort .