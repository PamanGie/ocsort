# OC-SORT Tracking Implementation

This repository contains implementation of OC-SORT (Observation-Centric SORT) for multi-object tracking using YOLOv8 as the detector. OC-SORT is an improved version of SORT algorithm that handles occlusions better.

![OC-SORT Tracking Demo](https://i.postimg.cc/nLSkgP5g/i-Screen-Shoter-Any-Desk-250402174754.jpg)

## Features

- Multi-object tracking using OC-SORT algorithm
- YOLOv8 integration for object detection
- Real-time tracking visualization
- Support for both webcam and video file input
- Customizable detection and tracking parameters

## Installation

### Environment Setup with Conda

The recommended way to set up the environment is using Conda. Follow these steps:

```bash
# Create a new conda environment
conda create -n tracking python=3.9
conda activate tracking

# Install PyTorch (with CUDA support if needed)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install other dependencies
conda install numpy scipy opencv matplotlib tqdm
conda install -c conda-forge lap

# Install Ultralytics for YOLOv8
pip install ultralytics

# Install OC-SORT
pip install ocsort
```

If you encounter issues with the `lap` package when using pip, install it via conda first:

```bash
conda install -c conda-forge lap
```

### Potential Issues & Solutions

If you encounter "ModuleNotFoundError: No module named 'distutils.msvccompiler'" when installing packages, install:

```bash
conda install setuptools
```

## Comparison with Other Trackers

OC-SORT performs well in occlusion scenarios compared to other trackers:

1. **DeepSORT**: Requires appearance features, computationally more expensive
2. **ByteTrack**: Good for crowded scenes but may struggle with long-term occlusions
3. **OC-SORT**: Observation-centric approach handles occlusions better than SORT/DeepSORT
4. **StrongSORT**: More complex but may provide better accuracy in some scenarios

## License

This implementation uses:
- OC-SORT: [MIT License](https://github.com/noahcao/OC_SORT/blob/master/LICENSE)
- YOLOv8: [AGPL-3.0 License](https://github.com/ultralytics/ultralytics/blob/main/LICENSE)

## Acknowledgements

- [OC-SORT Repository](https://github.com/noahcao/OC_SORT)
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
