#!/bin/bash
# Script to replace all occurrences of "Yoshida" with "Sato" in project files

echo "Replacing 'Yoshida' with 'Sato' in all project files..."

# Files to process
FILES=(
    "paper.md"
    "README.md"
    "pyproject.toml"
    "pyproject_simplified.toml"
    "setup.py"
    "LICENSE"
    "CITATION.cff"
    "paper.bib"
    "CHANGELOG.md"
    ".github/CONTRIBUTING.md"
    ".github/FUNDING.yml"
    "claire/__init__.py"
)

# Process each file
for file in "${FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "Processing: $file"
        # Use sed to replace Yoshida with Sato (works on both Mac and Linux)
        if [[ "$OSTYPE" == "darwin"* ]]; then
            # macOS
            sed -i '' 's/Yoshida/Sato/g' "$file"
        else
            # Linux
            sed -i 's/Yoshida/Sato/g' "$file"
        fi
        echo "  ✓ Updated $file"
    else
        echo "  ⚠ File not found: $file"
    fi
done

# Also check Python files recursively
echo ""
echo "Checking Python files..."
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    find . -name "*.py" -type f -exec sed -i '' 's/Yoshida/Sato/g' {} \; -print | while read file; do
        echo "  ✓ Updated $file"
    done
else
    # Linux
    find . -name "*.py" -type f -exec sed -i 's/Yoshida/Sato/g' {} \; -print | while read file; do
        echo "  ✓ Updated $file"
    done
fi

# Also check YAML files in .github
echo ""
echo "Checking GitHub workflow files..."
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    find .github -name "*.yml" -o -name "*.yaml" -type f 2>/dev/null | while read file; do
        sed -i '' 's/Yoshida/Sato/g' "$file"
        echo "  ✓ Updated $file"
    done
else
    # Linux
    find .github -name "*.yml" -o -name "*.yaml" -type f 2>/dev/null | while read file; do
        sed -i 's/Yoshida/Sato/g' "$file"
        echo "  ✓ Updated $file"
    done
fi

echo ""
echo "Replacement complete!"
echo ""
echo "Summary of changes:"
grep -r "Sato" --include="*.py" --include="*.md" --include="*.toml" --include="*.yml" --include="*.yaml" . 2>/dev/null | head -10

echo ""
echo "To verify no 'Yoshida' remains, run:"
echo "  grep -r 'Yoshida' --exclude-dir=.git --exclude='*.sh' ."