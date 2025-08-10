# Contributing to CLAIRE

Thank you for your interest in contributing to CLAIRE! We welcome contributions from the community.

## How to Contribute

### Reporting Bugs
- Use the issue tracker to report bugs
- Describe the bug and include specific details to help us reproduce it
- Include the version of CLAIRE you're using

### Suggesting Features
- Use the issue tracker to suggest features
- Explain your use case and how the feature would be useful

### Code Contributions

1. **Fork the repository**
   ```bash
   git clone https://github.com/username/claire.git
   cd claire
   git remote add upstream https://github.com/original/claire.git
   ```

2. **Create a branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make your changes**
   - Write clear, commented code
   - Follow PEP 8 style guidelines
   - Add unit tests for new functionality

4. **Format your code**
   ```bash
   black claire/
   flake8 claire/
   ```

5. **Run tests**
   ```bash
   pytest tests/
   ```

6. **Commit your changes**
   ```bash
   git commit -m "Add feature: description"
   ```

7. **Push to your fork**
   ```bash
   git push origin feature/your-feature-name
   ```

8. **Submit a pull request**

## Development Setup

```bash
# Clone the repository
git clone https://github.com/username/claire.git
cd claire

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e .
pip install -r requirements-dev.txt

# Run tests
pytest tests/
```

## Code Style

- Use black for formatting: `black claire/`
- Use flake8 for linting: `flake8 claire/`
- Follow NumPy docstring conventions
- Write descriptive variable names
- Add type hints where appropriate

## Testing

- Write tests for all new functionality
- Ensure all tests pass before submitting PR
- Aim for >80% code coverage
- Use pytest for testing

## Documentation

- Update docstrings for new functions/classes
- Update README if adding new features
- Add examples for new functionality

## Questions?

Feel free to open an issue for any questions about contributing.