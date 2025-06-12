# Contributing to Trylon Gateway

First off, thank you for considering contributing! We love your input and want to make contributing to this project as easy and transparent as possible. Whether it's reporting a bug, discussing new features, or contributing code, we welcome your involvement.

This document provides guidelines for contributing to Trylon Gateway.

## How Can I Contribute?

There are many ways to contribute, and all of them are appreciated:

### Reporting Bugs
If you find a bug, please open an issue and provide the following information:
- Your operating system and Python version.
- A clear and concise description of the bug.
- Steps to reproduce the bug.
- Any relevant logs or error messages.

### Suggesting Enhancements
If you have an idea for a new feature or an improvement to an existing one, please open an issue to start a discussion. This allows us to align on the proposal before any development work begins.

### Pull Requests
We welcome pull requests for bug fixes, new features, and improvements to documentation. Please follow the workflow described below.

## Setting Up Your Development Environment

To get started with development, you'll need to set up a local environment.

**Prerequisites:**
- Git
- Python 3.10+
- Poetry (for dependency management)

**Steps:**

1.  **Fork the repository** on GitHub.

2.  **Clone your fork** locally:
    ```bash
    git clone https://github.com/YOUR_USERNAME/gateway.git
    cd gateway
    ```

3.  **Install all dependencies**, including development tools like `pytest` and `black`:
    ```bash
    poetry install
    ```
    This command creates a virtual environment and installs all the necessary libraries from `pyproject.toml`.

4.  **Download the required spaCy model** for Presidio:
    ```bash
    poetry run python -m spacy download en_core_web_sm
    ```
    
5.  **(Optional but Recommended) Set up pre-commit hooks:**
    We use `pre-commit` to automatically run linters and formatters before each commit. This helps maintain a consistent code style.
    ```bash
    # Install the git hook scripts
    poetry run pre-commit install
    ```
    Now, `black` and `ruff` will run automatically on any staged files when you run `git commit`.

You are now ready to start developing!

## Development Workflow

1.  **Create a new branch** for your changes:
    ```bash
    git checkout -b feature/your-new-feature-name
    ```

2.  **Make your changes.** Write clean, readable code with comments where necessary.

3.  **Add or update tests.** If you are adding a new feature, please include tests for it. If you are fixing a bug, add a test that reproduces the bug and verifies the fix.

4.  **Run tests and linters** to ensure everything is working correctly and follows the project's style guide.
    - Run tests: `poetry run pytest`
    - Check formatting: `poetry run black --check .`
    - Check linting: `poetry run ruff check .`
    (If you set up pre-commit hooks, this is handled automatically on commit.)

5.  **Commit your changes** with a clear and descriptive commit message.
    ```bash
    git commit -m "feat: Add new validator for topic modeling"
    ```

6.  **Push your branch** to your fork on GitHub:
    ```bash
    git push origin feature/your-new-feature-name
    ```

7.  **Open a Pull Request** to the `main` branch of the original `trylonai/gateway` repository. Provide a clear description of your changes and link to any relevant issues.

## Running Tests

We use `pytest` for our test suite. A passing test suite is required for all pull requests.

### Running the Full Test Suite
This is the command our CI pipeline runs. It's the best way to ensure all changes are safe.
```bash
poetry run pytest -v
```

### Running a Specific Test File
To focus on a specific part of the codebase, you can run tests for a single file:
```bash
poetry run pytest tests/domain/validators/test_toxicity.py
```

### Running a Specific Test by Name
Use the `-k` flag to run tests whose names match a keyword expression.
```bash
poetry run pytest -k "block_pii"
```

### Test Coverage
We aim for high test coverage. You can generate a coverage report to see how much of your code is tested.
```bash
# Generate a summary report in the terminal
poetry run pytest --cov=src

# For a more detailed, browsable report, generate an HTML version
poetry run pytest --cov=src --cov-report=html
```
After running the HTML report, open `htmlcov/index.html` in your web browser. Please ensure your contributions do not decrease the overall test coverage.

## Code Style

We use **Black** for code formatting and **Ruff** for linting to maintain a consistent style.

- **To format your code:** `poetry run black .`
- **To check for linting errors:** `poetry run ruff check .`

As mentioned, setting up the `pre-commit` hooks is the easiest way to manage this automatically.