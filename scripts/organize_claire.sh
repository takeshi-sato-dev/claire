#!/bin/bash
# CLAIREプロジェクトを整理するスクリプト

echo "Organizing CLAIRE project structure..."

# Create proper package structure
mkdir -p claire
mkdir -p tests
mkdir -p docs
mkdir -p examples
mkdir -p scripts

# Move Python modules to claire package
echo "Moving Python modules..."
mv analysis claire/ 2>/dev/null || true
mv visualization claire/ 2>/dev/null || true
mv physics claire/ 2>/dev/null || true
mv core claire/ 2>/dev/null || true

# Move main scripts to claire package
mv run_analysis.py claire/ 2>/dev/null || true
mv cli.py claire/ 2>/dev/null || true
mv config.py claire/ 2>/dev/null || true

# Create __init__.py files
echo "Creating __init__.py files..."
touch claire/__init__.py
touch claire/analysis/__init__.py
touch claire/visualization/__init__.py
touch claire/physics/__init__.py
touch claire/core/__init__.py

# Move debug files to scripts/debug
mkdir -p scripts/debug
echo "Moving debug files..."
mv debug*.py scripts/debug/ 2>/dev/null || true
mv check_protein.py scripts/debug/ 2>/dev/null || true
mv leaflet_debug.py scripts/debug/ 2>/dev/null || true

# Move test files
echo "Moving test files..."
mv test_analysis.py tests/ 2>/dev/null || true

# Fix LICENSE filename
mv LISENCE.md LICENSE 2>/dev/null || true

# Clean up
echo "Cleaning up..."
rm -rf __pycache__ 2>/dev/null || true
rm -rf claire_output 2>/dev/null || true
rm -rf debug_output 2>/dev/null || true
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true

# Create directory structure file
cat > PROJECT_STRUCTURE.md << 'EOF'
# CLAIRE Project Structure

```
claire/
├── claire/                 # Main package
│   ├── __init__.py
│   ├── run_analysis.py     # Main entry point
│   ├── cli.py              # CLI interface
│   ├── config.py           # Configuration
│   ├── analysis/           # Analysis modules
│   │   ├── __init__.py
│   │   ├── ml_analysis.py
│   │   ├── temporal.py
│   │   └── conservation.py
│   ├── core/               # Core functionality
│   │   ├── __init__.py
│   │   ├── membrane.py
│   │   └── utils.py
│   ├── physics/            # Physics calculations
│   │   ├── __init__.py
│   │   └── distances.py
│   └── visualization/      # Plotting modules
│       ├── __init__.py
│       └── figures.py
├── tests/                  # Test suite
│   └── test_analysis.py
├── test_data/              # Test data files
├── examples/               # Example scripts
├── scripts/                # Utility scripts
│   └── debug/              # Debug scripts
├── docs/                   # Documentation
├── paper.md                # JOSS paper
├── paper.bib               # References
├── README.md               # Main documentation
├── LICENSE                 # MIT License
├── pyproject.toml          # Package configuration
├── setup.py                # Setup script
├── requirements.txt        # Dependencies
└── .gitignore              # Git ignore rules
```
EOF

echo "Done! Check PROJECT_STRUCTURE.md for the new structure."
echo ""
echo "Next steps:"
echo "1. Review the new structure"
echo "2. Update imports in Python files if needed"
echo "3. Remove CI workflow or create minimal tests"