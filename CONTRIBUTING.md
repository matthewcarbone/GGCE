# Contributing

Contributions are welcome, and they are greatly appreciated! Every
little bit helps, and credit will always be given.

You can contribute in many ways:

## Types of Contributions

### Report Bugs

Report bugs at `https://github.com/x94carbone/GGCE/issues`.

If you are reporting a bug, please include:

* Any details about your local setup that might be helpful in troubleshooting.
* Detailed steps to reproduce the bug.

### Fix Bugs

Look through the GitHub issues for bugs. Anything tagged with "bug"
is open to whoever wants to implement it.

### Implement Features

Look through the GitHub issues for features. Anything tagged with "feature"
is open to whoever wants to implement it.

### Write Documentation

GGCE could always use more documentation, whether
as part of the official GGCE docs, in docstrings,
or even on the web in blog posts, articles, and such.

### Submit Feedback

The best way to send feedback is to file an issue at `https://github.com/x94carbone/GGCE/issues`.

If you are proposing a feature:

* Explain in detail how it would work.
* Keep the scope as narrow as possible, to make it easier to implement.
* Remember that this is a volunteer-driven project, and that contributions
  are welcome :)

## Get Started!

Ready to contribute? Here's how to set up `GGCE` for local development.

1. Fork the `GGCE` repo on GitHub.
2. Clone your fork locally:

```bash
git clone git@github.com:YOUR_GIT_USERNAME/GGCE.git
```

3. Install your local copy into a virtualenv. We prefer `conda`, but other methods will suffice as well. Additionally, we highly recommend installing the pre-commit hooks. This will help you to write code that will pass the CI tests before pushing. Changes that do not pass the testing suite, including style checks e.g. flake8, will not be merged.

```bash
conda create -n ENV_NAME python=3.9
conda activate ENV_NAME
cd GGCE/
pip install -e ".[dev]"
pre-commit install
```

4. Create a branch for local development. Now you can make your changes locally.

```bash
git checkout -b name-of-your-bugfix-or-feature
```

5. Commit your changes and push your branch to GitHub. If you installed the pre-commit hooks, various checks including flake8 and black will be checked and possibly automatically fixed before your commit will be finalized.

```bash
git add .
git commit -m "Your detailed description of your changes."
git push origin name-of-your-bugfix-or-feature
````

6. Submit a pull request through the GitHub website.

## Pull Request Guidelines

Before you submit a pull request, check that it meets these guidelines:

1. The pull request should include tests.
2. If the pull request adds functionality, the docs should be updated. Put
   your new functionality into a function with a docstring, and add the
   feature to the list in README.md.
3. The pull request should work for Python 3.9 and for PyPy.

