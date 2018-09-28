# Assignment 1: Advanced Color-to-Gray Conversion

## Conventional rgb2gray

### Y = 0.299R + 0.587G + 0.114B

## Advanced rgb2gray

### Paper: Decolorization: Is rgb2gray() Out?<sup>[[1](#references)]</sup> [[project](https://ybsong00.github.io/siga13tb/)][[pdf](https://ybsong00.github.io/siga13tb/siga13tb_final.pdf)]
![Overview](Overview.png)

## Results

(wr, wg, wb)

Input | Conventional | Advanced 1 | Advanced 2 | Advanced 3
--- | --- | --- | --- | --- 
![1a](testdata/1a.png) | ![1a_y](testdata/1a_y.png) | ![1a_y1](testdata/1a_y1.png) (0, 0, 1) votes=6 | ![1a_y2](testdata/1a_y2.png) (0.8, 0.2, 0) votes=4 | ![1a_y3](testdata/1a_y3.png) (1, 0, 0) votes=3
![1b](testdata/1b.png) | ![1b_y](testdata/1b_y.png) | ![1b_y1](testdata/1b_y1.png) (0, 0.7, 0.3) votes=3 | ![1b_y2](testdata/1b_y2.png) (0.1, 0.5, 0.4) votes=3 | ![1b_y3](testdata/1b_y3.png) (0.3, 0.3, 0.4) votes=2
![1c](testdata/1c.png) | ![1c_y](testdata/1c_y.png) | ![1c_y1](testdata/1c_y1.png) (0.5, 0.5, 0) votes=3 | ![1c_y2](testdata/1c_y2.png) (0.6, 0.4, 0) votes=3 | ![1c_y3](testdata/1c_y3.png) (0.9, 0.1, 0) votes=3

## Requirements
* numpy
* scikit-image

## References
[1] Y. Song, L. Bao, X. Xu, and Q. Yang, “Decolorization: Is rgb2gray()  out?” in Proc. ACM SIGGRAPH Asia Tech. Briefs, 2013, Art. ID 15.
