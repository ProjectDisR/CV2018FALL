# Assignment 4: Stereo Matching
Stereo Matching [[slides](http://media.ee.ntu.edu.tw/courses/cv/18F/hw/cv2018_hw04.pdf)]
## Disparity Estimation

### Paper: Fast Cost-Volume Filtering for Visual Correspondence and Beyond <sup>[[1](#references)]</sup> [[pdf](http://wwwpub.zih.tu-dresden.de/~cvweb/publications/papers/2012/FastCost-VolumeFiltering.pdf)]

## Results

Left View | Right View | Disparity
--- | --- | ---
![tsukuba_l](testdata/tsukuba/img3.png) | ![tsukuba_r](testdata/tsukuba/img4.png) | ![tsukuba_d](tsukuba.png)
![venus_l](testdata/venus/img2.png) | ![venus_r](testdata/venus/img6.png) | ![venus_d](venus.png)
![teddy_l](testdata/teddy/img2.png) | ![teddy_r](testdata/teddy/img6.png) | ![teddy_d](teddy.png)
![cone_l](testdata/cone/img2.png) | ![cone_r](testdata/cone/img6.png) | ![cone_d](cone.png)

## Requirements
* numpy
* scikit-image

## References
[1] C. Rhemann, A. Hosni, M. Bleyer, C. Rother, and M. Gelautz, “Fast cost-volume filtering for visual correspondence and beyond,” in CVPR, 2011.
