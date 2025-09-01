# Text Line Segmentation Tool

A robust tool for segmenting handwritten documents into individual lines of text, inspired by the 2007 paper proposed by Manivannan Arivazhagan, Harish Srinivasan, and Sargur Srihari.

> [A Statistical Approach to Line Segmentation in Handwritten Documents](https://cedar.buffalo.edu/~srihari/papers/SPIE-2007-lineSeg.pdf)

## Key Features
- Gaussian-based line modeling.
- Probabilistic decision-making for ambiguous components.
- Handles overlapping and skewed lines.

## Methodology
### Image Preprocessing:
- Noise reduction with a 3x3 filter.
- Binarization with Otsu's thresholding.

### Contour Analysis:
- OpenCV's findContours & overlapping component merging.
- Vertical chunk processing.

### Local Line Detection
- Vertical projection histogram for each chunk.
- Histogram smoothing w/ a moving average filter.
- Histogram peaks (local maxima) corresponding to the center of text lines.
- Histogram valleys (local minima) corresponding to the gaps between text lines.
- Valley connection across chunks.

### Refinement and Region Extraction
- Line adjusting through Gaussian modelling & distance calculation.
