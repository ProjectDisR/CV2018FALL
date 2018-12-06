# Assignment 4: Stereo Matching
Stereo Matching [[slides](http://media.ee.ntu.edu.tw/courses/cv/18F/hw/cv2018_hw04.pdf)]
## Disparity Estimation

### Paper: Fast Cost-Volume Filtering for Visual Correspondence and Beyond <sup>[[1](#references)]</sup> [[pdf](http://wwwpub.zih.tu-dresden.de/~cvweb/publications/papers/2012/FastCost-VolumeFiltering.pdf)]

## Results

Left View | Right View | Disparity Eval | Ground Truth
--- | --- | --- | --- 
![tsukuba_l](testdata/tsukuba/im3.png) | ![tsukuba_r](testdata/tsukuba/im4.png) | ![tsukuba_d](tsukuba.png) | ![tsukuba_gt](testdata/tsukuba/disp2.png)
![venus_l](testdata/venus/im2.png) | ![venus_r](testdata/venus/im6.png) | ![venus_d](venus.png) | ![venus_gt](testdata/venus/disp2.png)
![teddy_l](testdata/teddy/im2.png) | ![teddy_r](testdata/teddy/im6.png) | ![teddy_d](teddy.png) | ![teddy_gt](testdata/teddy/disp2.png)
![cones_l](testdata/cones/im2.png) | ![cones_r](testdata/cones/im6.png) | ![cones_d](cones.png) | ![cones_gt](testdata/cones/disp2.png)

## Requirements
* numpy
* scikit-image

## References
[1] C. Rhemann, A. Hosni, M. Bleyer, C. Rother, and M. Gelautz, “Fast cost-volume filtering for visual correspondence and beyond,” in CVPR, 2011.
