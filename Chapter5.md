# chapter 5: Filters and Convolution

Tam Ky city, Quang Nam Province

20- /7/2021

[@KIM]: Keep In Mind

[@NUY]: Not Understand Yet

## Before We Begin

### Filters, Kernels, and Convolution

A filter is any algorithm that starts with some image I(x, y) and computes a new image I’(x, y) though some functions of region around that point. Template define shape and how to combine, is called a *filter* or a *kernel* .By sum of product of weights and arounding pixels respectively. This is called *linear kernels*.

The size of the array is called the *support* of the *kernel*.

Any *filter* which can be expressed with a *linear kernel* is also known as *convolutions*.

### Border Extrapolation and Boundary Conditions

OpenCV function (cv::blur(), cv::erode(), cv::dilate(), etc.) produce output images of the same size as the input. To handle border issues, there are some virtual pixels out of image at the borders.

Making Borders Yourself `cv::copyMakeBorder()`

## Threshold Operations

Make decision about pixels such as reject those pixels below or above some value (threshold)

Threshold as a very simple **convolution** operation that use 1-by-1 kernel

One thing is that we don’t want to add directly into an 8-bit array (with the idea of normalizing next) because the higher bits will overflow. Instead, we use equally weighted addition of the three color channels (`cv::addWeighted()`);

```c++
void sum_rgb(const cv::Mat &src, cv::Mat &dst)
{

   // Split image onto the color planes.
   vector<cv::Mat> planes;
   cv::split(src, planes);

   cv::Mat b = planes[0], g = planes[1], r = planes[2], s;
   // Add equally weighted rgb values.
   cv::addWeighted(r, 1. / 3., g, 0.0, 0.0, s);
   cv::addWeighted(s, 1., b, 1. / 3., 0.0, s);
   // Truncate values above 100.
   cv::threshold(s, dst, 256, 100, cv::THRESH_TRUNC);
}
```

### Otsu’s Algorithm [@NYT]

Pass `cv::THRESH_OTSU` as the value of `thresh` in `cv::threshold()`

Otsu's algorithm minimize:
$$
\sigma_{w}^2 \equiv w_1(t) \times \sigma_1^2 + w_2(t) \times \sigma_2^2 
$$

Minimize the variance of 2 classes is the same as maximize the variance between 2 classes.

Not a particular fast process

### Adaptive Threshold

Modified threshold technique that use threshold level as not **constant** but **variable** depend on image.

Implemented in the `cv::adaptiveThreshold()`  [Jain86]

```c++
void cv::adaptiveThreshold(
    cv::InputArray src,  // Input image
    cv::OutputArray dst, // Result image
    double maxValue,     // Max value for upward operations
    int adaptiveMethod,  // mean or Gaussian
    int thresholdType    // Threshold type to use
    int blockSize,       // Block size
    double C             // Constant
);
```

`blockSize`(b): *area* b x b around each pixels

`thresh` =
$$ \frac{1}{b\times b} \sum_{i \in area} W_i \times I_i  - C$$
where,

* $W_i$ is weight at pixel i-th
* $I_i$ is intensity at pixel i-th

`thresh` is not a global number but a variable that depend on 2 value:

* Constant C acts like offset
* weights (not intensity, depend on `adaptiveMethod`) of surrounding pixels in a *area*
  * If there is equal (or very similar) about weight and intensity between these pixels, thresh = weight x intensity - C; so all equal pixels in that *area* has weight x intensity greater than thresh (assume C > 0)
  * Otherwise, thresh = mean_of_weights - C, some of them is greater, some is less than or equal

The `adaptiveMethod` decides how the threshold value is calculated:

* `cv::ADAPTIVE_THRESH_MEAN_C`: The threshold value is the mean of the neighbourhood area minus the constant C.
* `cv::ADAPTIVE_THRESH_GAUSSIAN_C`: The threshold value is a gaussian-weighted sum of the neighbourhood values minus the constant C.

`thresholdType` is the same as in `cv::threshold()`

The **adaptive threshold** technique is useful when there are strong illumination or reflectance gradients that you need to threshold relative to the general intensity gradient

## Smoothing (blurring)

reduce noise or camera artifacts
reduce the resolution of an image (Image Pyramids)

### Simple Blur and the Box Filter

`cv::blur()`, output  = mean (kernel)

The simple blur is a specialized version of the box filter

### Median Filter

middle value

Median filter suits for skew and outlier image rather than simple blur that will be affect badly

### Gaussian Filter

Most useful

Gaussian idea: pixels in real image should vary slowly under a region

OpenCV provides a higher performance optimization for several common kernels. 3 × 3, 5 × 5, and 7 × 7 kernels, with "standard" sigma (i.e., sigmaX = 0.0) give better performance than other kernels

### Bilateral Filter

known as edge-preserving smoothing

Gaussian smoothing reduces noise while preserving signal, however this method break down near edges, where you do expect pixels to be uncorrelated with their neighbors across the edge.

Bilateral Filter like Gaussian proposes a Gaussian-based weight kernel on spatial distance from the center of the kernel. Moreover, it adds a Gaussian weighting that not based on spatial distance, but rather on **different** in intensity from the center.
$$
\Delta I = I_{center} - I_i (intensity)\\
\Delta S = S_{center} - S_i (spatial)
$$

>[!SHORT]Smoothing the weighs similar pixels more highly than less similar ones (edge)

Userful to segmenting the image
More complexity and huge computation than Gaussian filter

## Derivatives and Gradients

One of the most basic and important convolution is computing derivative

### The Sobel Derivative

Useful to detect edges by computing gradient according to axis x and y

But it prefer to use kernel to computing gradient rather than use the derivative formula:

* $G_x= K * I$ ,where K =

| -1  | 0   | +1  |
| --- | --- | --- |
| -2  | 0   | +2  |
| -1  | 0   | +1  |

* $G_y= K * I$, where K =

| -1  | -2  | 1   |
| --- | --- | --- |
| 0   | 0   | 0   |
| +1  | +2  | +1  |

* Then $G=\sqrt{G_x^2 + G_y ^2}$ or the simpler equation $G = |G_x| + |G_y|$

2 types of order of derivatives for 2 axis: horizontal (x) and vertical (y). Value = 0, 1, 2 respectively zero, first, second derivative order.

Defined for kernels of any size, it is quick, iterative.

Larger kernel give a better approximation to the derivative because they are less sensitive to noise.

```c++
Mat grad_x, grad_y;
Mat abs_grad_x, abs_grad_y;
Sobel(src_gray, grad_x, ddepth, 1, 0, ksize, scale, delta, BORDER_DEFAULT);
Sobel(src_gray, grad_y, ddepth, 0, 1, ksize, scale, delta, BORDER_DEFAULT);

// converting back to CV_8U
convertScaleAbs(grad_x, abs_grad_x);
convertScaleAbs(grad_y, abs_grad_y);

addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad);

imshow(window_name, grad);
char key = (char)waitKey(0);
```

When kernel is small, however, the `Sobel filter` will be inaccurate because of random noises.

### Scharr Filter

Overcome the downside of `Sobel filter` that is less accurate in small kernel (i.e 3x3)

As fast but more accurate than `Sobel filter` for small kernel.

The surrounding pixels will have a larger influence on the edge, and the edge will be more.

### The Laplacian

$laplace(f)=\frac{\partial^2 f}{\partial x^2}+\frac{\partial^2 f}{\partial y^2}$

In OpenCV, implematation is something like second-order of `Sobel filter`

In special case, `ksize=1`, the Laplacian operator
is computed by:

| 0   | 1   | 0   |
| --- | --- | --- |
| 1   | -4  | 1   |
| 0   | 1   | 0   |

A common application is to detect *"blobs"*, which used for edge detection because edge is similar to just a lot of *"blobs"* in some order.

```c++
Laplacian( src_gray, dst, ddepth, kernel_size, scale, delta, BORDER_DEFAULT );
```

**Unfortunately**, both substantial and less meaningful edges will be 0s of the Laplance

## Image Morphology

based on just 2 primitive operations 

### Dilation and Erosion

Basic morphological transformations.

Apply to a variety of contexts:

* remove noise
* isolating individual element
* join the disparate elements in an image

More sophisticate morphology application:

* find intensity peaks (or holes)
* define (yet another)  particular form of image gradient

Dilation is a convolution operator. It is the same as `max pool` algorithm in Machine Learning. It take the local maximum over the area of the kernel, which is a **non-linear** funtion.

Application: fill a region in a image, more detail in Figure 10-19, page 277 of the book.

In contrast, Erision is converse operation. It is `min pool` in ML that takes the local minimum over the area of the kernel.

>![!NOTE]
>Image morphology is often done on Boolean images (just 0 and 1 over the whole pixels intensity of the image)
>
> However, because dilation is just a max operator and erosion is just a min operator, morphology may be used on intensity images as well.

In general, `dilation` expands a bright region and `erosion` reduces such a bright region.

Formula:

$erode(x, y)=min_{(i,j)\in kernel} src(x + i, y + j)$

$dilate(x, y)=max_{(i,j)\in kernel} src(x + i, y + j)$

In OpenCV, we use `cv::erode()` and`cv::dilate()`

The erode operation is often used to eliminate “speckle" noise(blobs) in an image. Results of the erosion, or “min,” operator: bright regions are isolated and shrunk

The dilate operation is often used to try to find connected components. Results of the dilation, or “max,” operator: bright regions are expanded and often joined

### The General Morphology Function

When working with Boolean images and image masks where image pixels are either on (>0) or off (=0), dilate and erode are usually sufficient.

When working with grayscale or color images, however, a number of addtional operations are useful. These operation are called by just 1 function with multipurpose `cv::morphologyEx()`.

### Opening and Closing

simple combinations of the erosion and dilation operators.
opening: erode first than dilate

* erode first to separate out cells that are near each other before counting the regions
* => Count regions in a Boolean image
closing: dilate first than erode

* used in most of sophisticated connected-component algorithms [@KIM]
* reduce unwanted or noise-driven segments

For **connected components**, `erode` or `closing` first to eliminate *"blobs"* elements that arise purely from noise because pure noise has a property that almost the surrounding pixels of the noise pixel is very different from that pixel. And then use `opening` operation to connect **nearby large** regions.

>![NOTE]
> Although the end result of using `opening` or `closing` is similar to using `erosion` or `dilation`, these new operations tend to **preserve the area of connected regions more accurately**.)

When used on non-Boolean images,

* opening:
  * the upward outliers are eliminated
  * small bright regions are removed, and the remaining bright regions are isolated but retain their size
* closing:
  * downward outliers are eliminated
  * bright regions are joined but retain their basic size

### Morphological gradient [@NUY]

gradient(src) = dilate(src) – erode(src)

This formula creates the complete perimeter of a region

Telling us how fast the image brightness is changing

Used when we want to isolate the perimeters of bright regions so we can treat them as whole objects (or as whole parts of objects). [@NUY]

This differs from calculating gradient, which is less likely to work around the full perimeter of an object

>[!NOTE] Why it works? What is the advantage compared to other methods, such as Derivative gradient?

* Top Hat & Black Hat

1. Top Hat: used to isolate patches that brighter than their immediate neighbors, reveal areas that are lighter than the surrounding region, relative to the size of the kernel

  TopHat(src) = src – open(src) // Isolate brighter

  Gray image: bright local peaks are isolated
2. Black Hat: used to isolate patches dimmer than their immediate neighbors, reveals areas that are darker than the surrounding

  BlackHat(src) = close(src) – src // Isolate dimmer

  Gray Image: dark holes are isolated

>[!FACT] Both of these operations (Top Hat and Black Hat) are most useful in grayscale morphology

### Making Your Own Kernel

## Convolution with an Arbitrary Linear Filter

There are 2 kinds of kernel: separable kernel and inseparable kernel.

* Separable kernel is nxn kernel that can represent by product of 2 nx1 and 1xn kernel. It is effective to computing. Its complexity: O(n). We can apply with the first x-dimention kernel and then with the second y-dimension kernel. Sobel and Scharr are separable kernels.

* In contrast, Inseparable kernel is not able to separable, therefore the computing is slower than separable kernel. Its complexity: O(n^2)

### Applying a General Filter with `cv::filter2D()`

This is internal optimization function that you will not worry about the efficiency computing, such as separable kernel.

>[!NOTE] The function uses the DFT-based algorithm in case of sufficiently large kernels (~11 x 11 or larger) and the direct algorithm for small kernels.


### Applying a General Separable Filter with `cv::sepFilter2D`

`cv::sepFilter2D()` is like cv::filter2D(), except that it expects these two one-dimensional kernels instead of one two-dimensional kernel.

### Kernel Builders

`cv::getDerivKernel()`

Used to construct the **Sobel** and **Scharr** kernels. They are separable.

>[@NUY] For situations where you are operating on **floating-point**
images, there is no reason not to set normalize to true, but when you are doing
operations on **integer** arrays, it is often more sensible to not normalize the arrays
until some later point in your processing, so you won’t throw away precision that you
will later need.2

`cv::getGaussianKernel()`

Gaussian filter is generated by `cv::getGaussianKernel()`. It is separable.

Return 1-dimension [ksize x 1] matrix of Gaussian filter coefficients. Sum of them are 1.

Two of such generated kernels can be passed to `sepFilter2D`

## Summary

general image convolution
how boundaries are handled in convolutions
image kernels
difference between linear and nonlinear kernels
common image filters

## Exercises

Outdated function: `cv::smooth()`, instead we use:

* cv::blur
* cv::GaussianBlur
* cv::medianBlur
* cv::bilateralFilter

1. Load an image with interesting textures. Smooth the image in several ways using
cv::smooth() with smoothtype=cv::GAUSSIAN.
a. Use a symmetric 3 × 3, 5 × 5, 9 × 9, and 11 × 11 smoothing window size and
display the results.
b. Are the output results nearly the same by smoothing the image twice with a
5 × 5 Gaussian filter as when you smooth once with two 11 × 11 filters? Why
or why not?

```c++
CV_Assert(argc==2);
cv::Mat image = cv::imread(argv[1]);
if(image.empty()){
  cout << "Can not read image\n";
}

cv::Mat gauss_image5, gauss_image11;
cv::GaussianBlur(image, gauss_image5, cv::Size(5,5),1,1);
cv::GaussianBlur(gauss_image5, gauss_image5, cv::Size(5,5),1,1);
cv::GaussianBlur(image, gauss_image11, cv::Size(11,11),1,1);

imshow("Original", image);
imshow("Gauss applying kernel 5x5 twice",gauss_image5);
imshow("Gauss applying kernel 11x11 once",gauss_image11);
Mat tmp = gauss_image11 - gauss_image5;
double minPixelValue, maxPixelValue;
minMaxIdx(tmp, &minPixelValue, &maxPixelValue);
cout << "Max Pixel value:" << maxPixelValue << ". Min Pixel Value:"<< minPixelValue << '\n';
```

Result:

```
Max Pixel value:14. Min Pixel Value:0
```

Conclusion:

Apply 5x5 kernel twice is quite close to apply 11x11 kernel once.

2. Create a 100 × 100 single-channel image. Set all pixels to 0. Finally, set the center
pixel equal to 255.
a. Smooth this image with a 5 × 5 Gaussian filter and display the results. What
did you find?
b. Do this again but with a 9 × 9 Gaussian filter.
c. What does it look like if you start over and smooth the image twice with the 5
× 5 filter? Compare this with the 9 × 9 results. Are they nearly the same? Why
or why not?

```c++
CV_Assert(argc == 1);
int mat_size[] = {100, 100};

cv::Mat_<uchar> m = cv::Mat(2, mat_size, CV_8UC1, cv::Scalar(0));
m.ptr(50)[50] = 255;

imshow("original", m);
//a
Mat_<uchar> m55;
GaussianBlur(m, m55, cv::Size(5, 5), 1);
imshow("5x5", m55);

//b
Mat_<uchar> m99;
GaussianBlur(m, m99, cv::Size(9, 9), 1);
imshow("9x9", m99);

//c
GaussianBlur(m55, m55, cv::Size(5, 5), 1);
Mat tmp = m99 - m55;
imshow("Diff between 5x5 twice and 9x9 once", tmp);

double minPixelValue, maxPixelValue;
minMaxIdx(tmp, &minPixelValue, &maxPixelValue);
cout << "Max Pixel value:" << maxPixelValue << ". Min Pixel Value:" << minPixelValue << '\n';
```

Result

```
Max Pixel value:1. Min Pixel Value:0
```

Conclusion: nearly the same. Why?
3. Load an interesting image, and then blur it with cv::smooth() using a Gaussian
filter.
a. Set param1=param2=9. Try several settings of param3 (e.g., 1, 4, and 6). Display
the results.
b. Set param1=param2=0 before setting param3 to 1, 4, and 6. Display the results.
Are they different? Why?
c. Use param1=param2=0 again, but this time set param3=1 and param4=9.
Smooth the picture and display the results.
d. Repeat Exercise 3c but with param3=9 and param4=1. Display the results.
e. Now smooth the image once with the settings of Exercise 3c and once with
the settings of Exercise 3d. Display the results.
f. Compare the results in Exercise 3e with smoothings that use
param3=param4=9 and param3=param4=0 (i.e., a 9 × 9 filter). Are the results the
same? Why or why not?

```c++
CV_Assert(argc == 2);
Mat image = imread(argv[1]);
if (image.empty())
{
  return -1;
}

imshow("original", image);
//a
Mat m99;
GaussianBlur(image, m99, cv::Size(9, 9), 1);
imshow("9x9 with sigma= 1", m99);
GaussianBlur(image, m99, cv::Size(9, 9), 4);
imshow("9x9 with sigma= 4", m99);
GaussianBlur(image, m99, cv::Size(9, 9), 6);
imshow("9x9 with sigma= 6", m99);

//b
Mat m00;
GaussianBlur(image, m00, cv::Size(0, 0), 1);
imshow("0x0 with sigma= 1", m00);

//c
GaussianBlur(image, m00, cv::Size(0, 0), 1, 9);
imshow("0x0 with sigmaX= 1, sigmaY= 9", m00);

//d
GaussianBlur(image, m00, cv::Size(0, 0), 9, 1);
imshow("0x0 with sigmaX= 9, sigmaY= 1", m00);

//e
GaussianBlur(image, m00, cv::Size(0, 0), 1, 9);
GaussianBlur(m00, m00, cv::Size(0, 0), 9, 1);
imshow("0x0 with 2 times applying kernel with (sigmaX,sigmaY)= (1,9) and (9,1)", m00);

//f 
GaussianBlur(image, m99, Size(9,9), 0,0);
imshow("9x9 with sigma= 0", m99);
Mat diff = m99 - m00;

double minPixelValue, maxPixelValue;
minMaxIdx(diff, &minPixelValue, &maxPixelValue);
cout << "Max Pixel value:" << maxPixelValue << ". Min Pixel Value:" << minPixelValue << '\n';
```

Result:

```
Max Pixel value:107. Min Pixel Value:0
```

Conclusion: very different

4. Use a camera to take two pictures of the same scene while moving the camera as
little as possible. Load these images into the computer as src1 and src1.
a. Take the absolute value of src1 minus src1 (subtract the images); call it
diff12 and display. If this were done perfectly, diff12 would be black. Why
isn’t it?
b. Create cleandiff by using cv::erode() and then cv::dilate() on diff12.
Display the results.
c. Create dirtydiff by using cv::dilate() and then cv::erode() on diff12
and then display.
d. Explain the difference between cleandiff and dirtydiff.
5. Create an outline of an object. Take a picture of a scene. Then, without moving
the camera, put a coffee cup in the scene and take a second picture. Load these
images and convert both to 8-bit grayscale images.
a. Take the absolute value of their difference. Display the result, which should
look like a noisy mask of a coffee mug.
b. Do a binary threshold of the resulting image using a level that preserves most
of the coffee mug but removes some of the noise. Display the result. The “on”
values should be set to 255.
c. Do a cv::MOP_OPEN on the image to further clean up noise.
d. Using the erosion operator and logical XOR function, turn the mask of the
coffee cup image into an outline of the coffee cup (only the edge pixels
remaining).
6. High dynamic range: go into a room with strong overhead lighting and tables
that shade the light. Take a picture. With most cameras, either the lighted parts
of the scene are well exposed and the parts in shadow are too dark, or the lighted
parts are overexposed and the shadowed parts are OK. Create an adaptive filter
to help balance out such an image; that is, in regions that are dark on average,
boost the pixels up some, and in regions that are very light on average, decrease
the pixels somewhat.
7. Sky filter: create an adaptive “sky” filter that smooths only bluish regions of a
scene so that only the sky or lake regions of a scene are smoothed, not ground
regions.
8. Create a clean mask from noise. After completing Exercise 5, continue by keep‐
ing only the largest remaining shape in the image. Set a pointer to the upper left
of the image and then traverse the image. When you find a pixel of value 255
(“on”), store the location and then flood-fill it using a value of 100. Read the con‐
nected component returned from flood fill and record the area of filled region. If
there is another larger region in the image, then flood-fill the smaller region
using a value of 0 and delete its recorded area. If the new region is larger than the
previous region, then flood-fill the previous region using the value 0 and delete
its location. Finally, fill the remaining largest region with 255. Display the results.
We now have a single, solid mask for the coffee mug.
9. Use the mask created in Exercise 8 or create another mask of your own (perhaps
by drawing a digital picture, or simply use a square). Load an outdoor scene.
Now use this mask with copyTo() to copy an image of a mug into the scene.
10. Create a low-variance random image (use a random number call such that the
numbers don’t differ by much more than three and most numbers are near zero).
Load the image into a drawing program such as PowerPoint, and then draw a
wheel of lines meeting at a single point. Use bilateral filtering on the resulting
image and explain the results.
11. Load an image of a scene and convert it to grayscale.
a. Run the morphological Top Hat operation on your image and display the
results.
b. Convert the resulting image into an 8-bit mask.
c. Copy a grayscale value into the original image where the Top Hat mask (from
Part b of this exercise) is nonzero. Display the results.
12. Load an image with many details.
a. Use resize() to reduce the image by a factor of 2 in each dimension (hence
the image will be reduced by a factor of 4). Do this three times and display the
results.
b. Now take the original image and use cv::pyrDown() to reduce it three times,
and then display the results.
c. How are the two results different? Why are the approaches different?
13. Load an image of an interesting or sufficiently “rich” scene. Using cv::thres
hold(), set the threshold to 128. Use each setting type in Figure 10-4 on the
image and display the results. You should familiarize yourself with thresholding
functions because they will prove quite useful.
a. Repeat the exercise but use cv::adaptiveThreshold() instead. Set param1=5.
b. Repeat part a of this exercise using param1=0 and then param1=-5.
14. Approximate a bilateral (edge preserving) smoothing filter. Find the major edges
in an image and hold these aside. Then use cv::pyrMeanShiftFiltering() to
segment the image into regions. Smooth each of these regions separately and
then alpha-blend these smooth regions together with the edge image into one
whole image that smooths regions but preserves the edges.
15. Use cv::filter2D() to create a filter that detects only 60-degree lines in an
image. Display the results on a sufficiently interesting image scene.
16. Separable kernels: create a 3 × 3 Gaussian kernel using rows [(1/16, 2/16, 1/16),
(2/16, 4/16, 2/16), (1/16, 2/16, 1/16)] and with anchor point in the middle.
a. Run this kernel on an image and display the results.
b. Now create two one-dimensional kernels with anchors in the center: one
going “across” (1/4, 2/4, 1/4), and one going down (1/4, 2/4, 1/4). Load the
same original image and use cv::filter2D() to convolve the image twice,
once with the first 1D kernel and once with the second 1D kernel. Describe
the results.
c. Describe the order of complexity (number of operations) for the kernel in
part a and for the kernels in part b. The difference is the advantage of being
able to use separable kernels and the entire Gaussian class of filters—or any
linearly decomposable filter that is separable, since convolution is a linear
operation.
17. Can you make a separable kernel from the Scharr filter shown in Figure 10-15? If
so, show what it looks like.
18. In a drawing program such as PowerPoint, draw a series of concentric circles
forming a bull’s-eye.
a. Make a series of lines going into the bull’s-eye. Save the image.
b. Using a 3 × 3 aperture size, take and display the first-order x- and yderivatives of your picture. Then increase the aperture size to 5 × 5, 9 × 9, and
13 × 13. Describe the results.
19. Create a new image that is just a 45-degree line, white on black. For a given series
of aperture sizes, we will take the image’s first-order x-derivative (dx) and firstorder y-derivative (dy). We will then take measurements of this line as follows.
The (dx) and (dy) images constitute the gradient of the input image. The magni‐
tude at location (i, j) is mag(i, j) = d x
2
(i, j) + d y
2
(i, j) and the angle is
Θ(i, j) = atan2(dy(i, j), dx(i, j)). Scan across the image and find places where
the magnitude is at or near maximum. Record the angle at these places. Average
the angles and report that as the measured line angle.
a. Do this for a 3 × 3 aperture Sobel filter.
b. Do this for a 5 × 5 filter.
c. Do this for a 9 × 9 filter.
d. Do the results change? If so, why?
