# chapter 5: Filters and Convolution 

Tam Ky city, Quang Nam Province

20- /7/2021

## Filters, Kernels, and Convolution

A filter is any algorithm that starts with some image I(x, y) and computes a new image I’(x, y) though some functions of region around that point. Template define shape and how to combine, is called a *filter* or a *kernel* .By sum of product of weights and arounding pixels respectively. This is called *linear kernels*.

The size of the array is called the *support* of the *kernel*.

Any *filter* which can be expressed with a *linear kernel* is also known as *convolutions*.

## Border Extrapolation and Boundary Conditions

OpenCV fuunction (cv::blur(), cv::erode(), cv::dilate(), etc.) produce output images of the same size as the input. To handle border issues, there are some virtual pixels out of image at the borders.

Making Borders Yoursel