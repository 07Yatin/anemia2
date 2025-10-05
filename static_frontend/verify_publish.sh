#!/bin/bash

# Verify publish directory contents
echo "Checking static frontend publish directory..."

# List files
echo "Files in current directory:"
ls -1

# Check critical files
REQUIRED_FILES=("index.html" "js/app.js" "css/styles.css")

for file in "${REQUIRED_FILES[@]}"; do
    if [ -f "$file" ]: then
        echo "✓ Found: $file"
    else
        echo "✗ Missing: $file"
        exit 1
    fi
done

# Check file references in index.html
echo "Checking file references..."
grep -q 'js/app.js' index.html && echo "✓ JS reference correct"
grep -q 'css/styles.css' index.html && echo "✓ CSS reference correct"

echo "Publish directory verification complete!"
exit 0
