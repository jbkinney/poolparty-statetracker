#!/bin/bash
# Clean LaTeX intermediate files
# Usage: clean_latex.sh [directory]
# If no directory is specified, cleans the current directory

if [ $# -eq 0 ]; then
    # No directory specified, use current directory
    TARGET_DIR="."
else
    # Directory specified as argument
    TARGET_DIR="$1"
fi

# Check if directory exists
if [ ! -d "$TARGET_DIR" ]; then
    echo "Error: Directory '$TARGET_DIR' does not exist."
    exit 1
fi

# Change to target directory
cd "$TARGET_DIR"

# Remove LaTeX intermediate files
rm -f *.aux *.bbl *.blg *.log *.out *.toc *.fls *.fdb_latexmk *.synctex.gz *.bcf *.run.xml

echo "LaTeX intermediate files cleaned in: $(pwd)"
