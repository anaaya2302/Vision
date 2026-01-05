# Edge and angle detection 
A PyTorch based computer vision project for detecting edges and calculating their respective angles. Also has a provision for visualizing the results in HSV color space, allowing users to see the edges, the angles, and the image plotted together. Uses basic methods like sobel followed by arctan to do the same.

## Folder structure
- `src/` : Core functions for batch processing, edge detection, and angle detection  
- `frontend/` : Scripts to visualize results; running the main file generates plots of edges, angles, and the original image in HSV  
- `datasets/` : Sample images for testing (small size recommended)  
- `checkpoints/` : Optional saved model checkpoints  
- `experiments/` : Experiment scripts and logs  
- `setup.py` : Package installation script

## Installation

1. Clone the repo:
```bash
git clone https://github.com/anaaya2302/vision_project.git
cd vision_project
```

2. Create and activate a virtual environment
I use miniconda because it works smoothly with cuda and torch, but venv works too.

3. Install dependencies
pip install -r requirements.txt

## Usage
To plot an image alongside edges and angles, run frontend_1.py [expects one jpg image path as command line argument].

To use underling functions for edge detection, angle detection, luminance grading, thresholding etc, access data.py
