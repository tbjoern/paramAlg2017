#!/bin/bash

BUILD_DIR="build"
PDF_DIR="pdf"

if [ "$#" -ne 1 ]; then
    echo "Usage: ./pdflatex.sh <# of exercise>"
    exit 0
fi

if [ ! -d "$BUILD_DIR" ]; then
    mkdir "$BUILD_DIR"
fi

if [ ! -d "$PDF_DIR" ]; then
    mkdir "$PDF_DIR"
fi 

cd "src"
pdflatex --output-directory "../build" --interaction "errorstopmode" "blatt$1.tex"
cd ".."
cp "$BUILD_DIR/blatt$1.pdf" "$PDF_DIR/"