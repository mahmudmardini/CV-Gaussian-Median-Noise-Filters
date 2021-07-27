# CV-Gaussian-Median-Noise-Filters
Computer-Vision-Gaussian-Median-Noise-Filters is an open-source code for applying different image filters like Gaussian, Median, and Noise filters. I developed this project with pure python programming language and NumPy framework without using any Computer Vision built-in functions as an assignment for Istanbul Technical University Computer Vision Course.

## Content
- Convolution Filter
- Gaussian Blur Kernel
- Median Kernel
- Noise Kernel

## Built With
- Python

## What is a convolution? 
Convolution is a general purpose filter effect for images.
Is a matrix applied to an image and a mathematical operation comprised of integers.
It works by determining the value of a central pixel by adding the weighted values of all its neighbors together.
The output is a new modified filtered image.
A convolution is done by multiplying a pixel’s and its neighboring pixels color value by a matrix.

## What is a gaussian blur filter? 
In image processing, a Gaussian blur (also known as Gaussian smoothing) is the result of blurring an image by a Gaussian function (named after mathematician and scientist Carl Friedrich Gauss).

It is a widely used effect in graphics software, typically to reduce image noise and reduce detail. The visual effect of this blurring technique is a smooth blur resembling that of viewing the image through a translucent screen, distinctly different from the bokeh effect produced by an out-of-focus lens or the shadow of an object under usual illumination.

Gaussian smoothing is also used as a pre-processing stage in computer vision algorithms in order to enhance image structures at different scales see scale space representation and scale space implementation.

## What is a median filter? 
The median filter is a non-linear digital filtering technique, often used to remove noise from an image or signal. Such noise reduction is a typical pre-processing step to improve the results of later processing (for example, edge detection on an image). Median filtering is very widely used in digital image processing because, under certain conditions, it preserves edges while removing noise (but see the discussion below), also having applications in signal processing.

## What is a noise filter? 
Noise filtering is the process of removing noise from a signal. Noise reduction techniques exist for audio and images. Noise reduction algorithms may distort the signal to some degree.

All signal processing devices, both analog and digital, have traits that make them susceptible to noise. Noise can be random or white noise with an even frequency distribution, or frequency-dependent noise introduced by a device's mechanism or signal processing algorithms.

### Clone Repo
```sh
git clone https://github.com/mahmudmardini/CV-Gaussian-Median-Noise-Filters.git
```

<!-- ### Note: 
You can see project code without using any IDE in 'CV Filters Codes.py' file. 
To open the project you need to import 'main.ipynb' file using Jupyter Notebook. -->

## Preview
![Preview](./preview/project_preview.gif)

## License
[MIT](LICENSE)
