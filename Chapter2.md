# chapter 2: introduction

## First Program—Display a Picture

```c
#include <opencv2/highgui/highgui.hpp>
int main(int argc, char **argv)
{
    cv::Mat img = cv::imread(argv[1], -1);
    if (img.empty())
        return -1;
    cv::namedWindow("Example1", cv::WINDOW_AUTOSIZE); // imshow() does not require create windows first, it will create a default one if there is no existing window.
    cv::imshow("Example1", img);
    cv::waitKey(0);
    cv::destroyWindow("Example1");
}
```

* Include specifically `"opencv2/highgui/highgui.hpp"` will compile more faster then `"opencv2/opencv.hpp"`
* `cv::imread` return `cv::Mat` structure, which handle all kinds of images: single-channel, multichannel, integer-valued, floating-point-valued, and so on.
* `cv::namedWindow( "Example2", cv::WINDOW_AUTOSIZE )` is the function provided by HighGUI library, it assigns a name and the window will `autosize` to fit the size of the image.
* `cv::imshow( "Example2", img )` display when we have an image in `cv::Mat` structure
* `cv::waitKey(arg)`, when `arg` = 0 or negative, the program will wait indefinitely for a keypress. If a positive argument
is given, no wait happens. `Note`: it must be used after `cv::imshow()` in order to display that image.
* `cv::Mat`, images are automatically deallocated when they go out of scope.
* `cv::destroyWindow( "Example2" );` will close the window
and deallocate any associated memory usage. This avoids memory leaks when work with a complex programs.

## Second Program—Video

```c
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
int main( int argc, char** argv ) { 
    cv::namedWindow( "Example3", cv::WINDOW_AUTOSIZE );
    cv::VideoCapture cap;
    cap.open( string(argv[1]) );
    cv::Mat frame;
    while( 1 ) {
        cap >> frame; 
        if( !frame.data ) break; // Ran out of film
        cv::imshow( "Example3", frame );
        if( cv::waitKey(33) >= 0 ) break;
    }
    return 0;
}
```

* The video capture object `cv::VideoCapture` is then instantiated. This object can **open and close** video files of as many types as *ffmpeg* supports.
* Videos is just many images (frames) combine in time order.`cap >> frame`. `cap` read frame by frame inside a  while loop.
* `if( cv::waitKey(33) >= 0 ) break;` we want to wait 33 ms = 0.33 s ~ 1/(30 fps), which allow users to interrupt between each frame. If the user hits a key during that time then we will exit the read loop. Other wise, 33 ms will  pass and we do other stuffs and loop again to repeat `cv::waitKey(33)`. 

> [!TIP]
> This is more convinient in most of case rather than `cv:waitKey([<=]0)` that will wait indefinitely until the user press a ketstroke.

## Moving Around

```c
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <fstream>
using namespace std;
int g_slider_position = 0; // keep the trackbar slider position state.
int g_run = 1, g_dontset = 0; //start out in single step mode
cv::VideoCapture g_cap;

void onTrackbarSlide(int pos, void *)
{
    g_cap.set(cv::CAP_PROP_POS_FRAMES, pos); // next frame
    if (!g_dontset) // set state to single step after the next frame comes in.
        g_run = 1;
    g_dontset = 0;
}
int main(int argc, char **argv)
{
    cv::namedWindow("Example2_4", cv::WINDOW_AUTOSIZE);
    g_cap.open(string(argv[1]));
    int frames = (int)g_cap.get(cv::CAP_PROP_FRAME_COUNT);
    int tmpw = (int)g_cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int tmph = (int)g_cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    cout << "Video has " << frames << " frames of dimensions("
         << tmpw << ", " << tmph << ")." << endl;
    cv::createTrackbar("Position", "Example2_4", &g_slider_position, frames,
                       onTrackbarSlide);    // Upon creation, the slider position is
                                            // defined by this g_slider_position.
    cv::Mat frame;
    while (1)
    {
        if (g_run != 0)
        {
            g_cap >> frame;
            if (!frame.data)
                break;
            int current_pos = (int)g_cap.get(cv::CAP_PROP_POS_FRAMES); 
            g_dontset = 1; //set to 1 so that next trackbar not put us into single step mode
            cv::setTrackbarPos("Position", "Example2_4", current_pos);
            cv::imshow("Example2_4", frame);
            g_run -= 1;
        }
        char c = (char)cv::waitKey(10);
        if (c == 's') // single step
        {
            g_run = 1;
            cout << "Single step, run = " << g_run << endl;
        }
        if (c == 'r') // run mode
        {
            g_run = -1;
            cout << "Run mode, run = " << g_run << endl;
        }
        if (c == 27)
            break;
    }
    return (0);
}
```

* Adding a leading `g_` to any global variable.
* `g_run` displays new frames as long it is different from zero.
  * A positive number tells how many frames are displayed before stop. For example, `g_run` = 1 means single step mode.
  * A negative number means the video run continously (video mode).
* `g_dontset` update trackbar position without trigger single state mode.
* `cv::g_cap.set()` to actually advance the video playback to the new position.

## A Simple Transformation

```c
#include <opencv2/opencv.hpp>
void example2_5(cv::Mat &image)
{
    // Create some windows to show the input
    // and output images in.
    //
    cv::namedWindow("Example2_5-in", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Example2_5-out", cv::WINDOW_AUTOSIZE);

    // Create a window to show our input image
    //
    cv::imshow("Example2_5-in", image);

    // Create an image to hold the smoothed output
    cv::Mat out;

    // Do the smoothing
    // Could use GaussianBlur(), blur(), medianBlur() or bilateralFilter().
    cv::GaussianBlur(image, out, cv::Size(5, 5), 3, 3);
    cv::GaussianBlur(out, out, cv::Size(5, 5), 3, 3); // double-blurred image

    // Show the smoothed image in the output window
    //
    cv::imshow("Example2_5-out", out); // Wait for the user to hit a key, windows will self destruct
    cv::waitKey(0);
}
int main(int argc, char** argv){
    if (argc != 2){
        fprintf(stderr, "Use ./%s <path/to/image>", argv[0]);
    }
    cv::Mat image = cv::imread(argv[1], -1);
    example2_5(image);
    return 0;
}
```

## A Not-So-Simple Transformation

### Gaussian blur

```c
#include <opencv2/opencv.hpp>
int main(int argc, char **argv)
{
    cv::Mat img = cv::imread(argv[1]), img2;
    cv::namedWindow("Example1", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Example2", cv::WINDOW_AUTOSIZE);
    cv::imshow("Example1", img);
    cv::pyrDown(img, img2);
    cv::imshow("Example2", img2);
    cv::waitKey(0);
    return 0;
};
```

* In OpenCV, this Gaussian blurring and downsampling is accomplished by the function cv::pyrDown()

Some name keep in mind:

* Nyquist-Shannon Sampling Theorem, [Shannon49] (downsample)
* [Rosenfeld80] (Gaussian blurring)
* [Canny86]

### Canny edge detector

```c
#include <opencv2/opencv.hpp>
int main(int argc, char **argv)
{
    cv::Mat img_rgb = cv::imread(argv[1]);
    cv::Mat img_gry, img_cny;
    cv::cvtColor(img_rgb, img_gry, cv::COLOR_BGR2GRAY);
    cv::namedWindow("Example Gray", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Example Canny", cv::WINDOW_AUTOSIZE);
    cv::imshow("Example Gray", img_gry);
    cv::Canny(img_gry, img_cny, 10, 100, 3, true);
    cv::imshow("Example Canny", img_cny);
    cv::waitKey(0);
}
```

* Edge detect needs only a single channel image and so we convert to a gray scale using `cv::cvtColor()`.

### Getting and setting pixels

```c
#include <opencv2/opencv.hpp>
int main(int argc, char **argv)
{
    cv::Mat img_rgb = cv::imread(argv[1]);
    cv::Mat img_gry, img_cny, img_pyr, img_pyr2;

    cv::cvtColor(img_rgb, img_gry, cv::COLOR_BGR2GRAY);

    cv::pyrDown(img_gry, img_pyr);
    cv::pyrDown(img_pyr, img_pyr2);
    cv::Canny(img_pyr2, img_cny, 10, 100, 3, true);

    int x = 160, y = 320;
    /* cv::Vec3b == cv::Vec<T,cn> where T:uchar, cn: 3 */
    cv::Vec3b intensity = img_rgb.at<cv::Vec3b>(y, x);
    /* We could write img_rgb.at< cv::Vec3b >(x,y)[0] */
    uchar blue = intensity.val[0];
    uchar green = intensity.val[1];
    uchar red = intensity.val[2];
    std::cout << "At (x,y) = (" << x << ", " << y
              << "): (blue, green, red) = ("
              << (unsigned int)blue << ", "
              << (unsigned int)green << ", "
              << (unsigned int)red << ")" << std::endl;
    std::cout << "Gray pixel there is: "
              << (unsigned int)img_gry.at<uchar>(x, y) << std::endl;
    x /= 4;
    y /= 4;
    std::cout << "Pyramid2 pixel there is: "
              << (unsigned int)img_pyr2.at<uchar>(x, y) << std::endl;
    img_cny.at<uchar>(x, y) = 128; // Set the Canny pixel there to 128
}
```

* `img_rgb` has 3 color channels, which casts to a vector_3 type `img_rgb.at<cv::Vec3b>(y, x);`
* `img_gray` and `img_pyr2` has just 1 color channel, so they cast to an unsigned char type `(unsigned int)img_pyr2.at<uchar>(x, y)`

## Input from a Camera

```c
//exmaple 2-10
#include <opencv2/opencv.hpp>
#include <iostream>
int main(int argc, char **argv)
{
    cv::namedWindow("Example2_10", cv::WINDOW_AUTOSIZE);
    cv::VideoCapture cap;
    if (argc == 1)
    {
        cap.open(0); // open the default camera
    }
    else
    {
        cap.open(argv[1]);
    }
    if (!cap.isOpened())
    { // check if we succeeded
        std::cerr << "Couldn't open capture." << std::endl;
        return -1;
    }
    // The rest of program proceeds as in Example 2-3
    cv::Mat frame;
    while (1)
    {
        cap >> frame;
        if (!frame.data)
            break; // Ran out of film
        cv::imshow("Example3", frame);
        if (cv::waitKey(33) >= 0)
            break;
    }
    return 0;
}
```

## Writing to an AVI File

```c
// Example 2-11: A complete program to read in a color video and write out the log-polar transformed video
#include <opencv2/opencv.hpp>
#include <iostream>
int main(int argc, char *argv[])
{
    cv::namedWindow("Example2_10", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Log_Polar", cv::WINDOW_AUTOSIZE);

    cv::VideoCapture capture;
    capture.open(0);


    double fps = capture.get(cv::CAP_PROP_FPS);
    cv::Size size(
        (int)capture.get(cv::CAP_PROP_FRAME_WIDTH),
        (int)capture.get(cv::CAP_PROP_FRAME_HEIGHT));

    cv::VideoWriter writer;
    writer.open(argv[1], cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), fps, size);
    cv::Mat logpolar_frame(size, CV_8UC3), bgr_frame;
    for (;;)
    {
        capture >> bgr_frame;
        if (bgr_frame.empty()){
            printf("done\n");
            break; // end if done
            
        }
        cv::imshow("Example2_10", bgr_frame);
        cv::logPolar(
            bgr_frame,              // Input color frame
            logpolar_frame,         // Output log-polar frame
            cv::Point2f(            // Centerpoint for log-polar transformation
                bgr_frame.cols/2 , // x
                bgr_frame.rows/2 // y
                ),
            40,                    // Magnitude (scale parameter)
            cv::WARP_FILL_OUTLIERS // Fill outliers with ‘zero’
        );
        cv::imshow("Log_Polar", logpolar_frame);
        writer << logpolar_frame;
        char c = cv::waitKey(10);
        if (c == 27)
            break; // allow the user to break out
    }
    capture.release();
    char c = cv::waitKey(0);
}
```

* Video codec: to compress and decompress a video to a file and vice versa. The popular ones is MJPG (Motion Jpeg),  

> [!CAUTION]
> Video codec must match our own machine.

* cv::VideoWriter 's args:
  * file name for new file
  * frame rate (fps)
  * video codec using cv::VideoWriter::fourcc('M', 'J', 'P', 'G')
  * size of the images

## Summary

* Know how to read and write images and videos from and to files along with capturing video from cameras.
* Look at some library contain primitive funtions for manipulating these images.

## Exercises

Check out ../sample/cpp!

1. Build the sample in ../opencv/samples/
2. Go to  ../opencv/samples/ and look for *lkdemo.c*. Attach a camera to your system. 'r': initalize; 'n': toggle between 'night' and 'day' views.
3. Combine Ex2-11 and Ex2-6 to donwsample video capturing from camera.
4. Same above with Ex3.
5. Modify Ex4 with a slider control from Ex2-4 so that user can dynamically vary the pyramid downsampling reduction level by factors of between 2 and 8. (skip writing the disk, but you should display the results)
