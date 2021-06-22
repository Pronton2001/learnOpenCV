# chapter 2: introduction

Tam Ky city, Quang Nam Province

17-18/06/2021

## First Program—Display a Picture

```c
// Example 2-2: Same as Example 2-1 but employing the “using namespace” directive. For faster compile, we use only the needed header file, not the generic opencv.hpp
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
// Example 2-3: A simple OpenCV program for playing a video file from disk. In this example we only use specific module headers, rather than just opencv.hpp. This speeds up compilation, and so is sometimes preferable.
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
// Example 2-4: Program to add a trackbar slider to the basic viewer window for moving around within the video file
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
// Example 2-5: Loading and then smoothing an image before it is displayed on the screen
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
// Example 2-6: Using cv::pyrDown() to create a new image that is half the width and height of the input 
image
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
// Example 2-7: The Canny edge detector writes its output to a single-channel (grayscale) image
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
// Example 2-8: Combining the pyramid down operator (twice) and the Canny subroutine in a simple image pipeline
// Example 2-9: Getting and setting pixels in Example 2-8
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

    // cv::Size size(
    //     (int)capture.get(cv::CAP_PROP_FRAME_WIDTH),
    //     (int)capture.get(cv::CAP_PROP_FRAME_HEIGHT));
    // double fps = capture.get(cv::CAP_PROP_FPS);
    cv::Mat logpolar_frame(cv::Size(1280, 720), CV_8UC3), bgr_frame;
    capture >> bgr_frame;

    cv::VideoWriter writer;
    std::string filename = "./live2.avi";
    int codec = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');  // select desired codec (must be available at runtime)
    double fps = 25.0;
    // printf("size:(%d, %d)\n", _size.width, _size.height); // why zero
    writer.open(filename, codec, fps, bgr_frame.size()); // why Size(1280, 720) wrong
    for (;;)
    {
        capture >> bgr_frame;
        // printf("size:(%d, %d)\n", bgr_frame.size().width, bgr_frame.size().height);
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

- [x] Build the sample in ../opencv/samples/
- [x] Go to  ../opencv/samples/ and look for *lkdemo.c*. Attach a camera to your system. 'r': initalize; 'n': toggle between 'night' and 'day' views.
- [x] Combine Ex2-11 and Ex2-6 to donwsample video capturing from camera.

```c
// Exercise 3
#include <opencv2/opencv.hpp>

#include <iostream>

int main(int argc, char *argv[])
{
    cv::namedWindow("Exercise 3", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Pyramid_Down", cv::WINDOW_AUTOSIZE);

    cv::VideoCapture capture;
    capture.open(0);

    cv::Mat pyr_frame, bgr_frame;
    capture >> bgr_frame;
    cv::pyrDown(bgr_frame, pyr_frame);
    cv::VideoWriter writer;
    std::string filename = "./live3.avi";
    int codec = cv::VideoWriter::fourcc('M', 'J', 'P', 'G'); // select desired codec (must be available at runtime)
    double fps = 25.0;
    cv::Size _size = bgr_frame.size();
    printf("size:(%d, %d)\n", _size.width, _size.height); // why zero
    writer.open(filename, codec, fps, pyr_frame.size()); // why Size(1280, 720) wrong
    for (;;)
    {
        capture >> bgr_frame;
        printf("size:(%d, %d)\n", bgr_frame.size().width, bgr_frame.size().height);
        if (bgr_frame.empty())
            break; // end if done

        cv::imshow("Exercise 3", bgr_frame);
        cv::pyrDown(
            bgr_frame,              // Input color frame
            pyr_frame         // Output log-polar frame
        );
        cv::imshow("Pyramid_Down", pyr_frame);
        writer << pyr_frame;
        char c = cv::waitKey(10);
        if (c == 27)
            break; // allow the user to break out
    }
    capture.release();
    char c = cv::waitKey(0);
}

```
- [x] Modify the above code with Ex2-2.

- [x] Modify the above code with a slider control from Ex2-4 so that user can dynamically vary the pyramid downsampling reduction level by factors of between 2 and 8. (skip writing the disk, but you should display the results)
```c
// Exercise 5
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/opencv.hpp>

#include <iostream>
#include <fstream>

using namespace std;
enum state
{
    py0 = 0,
    py2 = 1,
    py4 = 2,
    py8 = 3
};
int g_slider_state = -1; // keep the trackbar slider position state.
int g_state = py0;

cv::Mat bgr_frame, pyr_frame;

void onTrackbarSlide(int state, void *)
{
    switch (state)
    {
    case py0:
        cv::imshow("Ex5", bgr_frame);
        g_writer << bgr_frame;
        return;
    case py2:
        cv::pyrDown(bgr_frame, pyr_frame);
        break;
    case py4:
        cv::pyrDown(bgr_frame, pyr_frame);
        cv::pyrDown(pyr_frame, pyr_frame);
        printf("py4\n");
        break;
    case py8:
        printf("py8\n");
        cv::pyrDown(bgr_frame, pyr_frame);
        cv::pyrDown(pyr_frame, pyr_frame);
        cv::pyrDown(pyr_frame, pyr_frame); 
        break;

    default:
        break;
    }
    cv::imshow("Ex5", pyr_frame);
    g_writer << pyr_frame;
}
int main(int argc, char *argv[])
{
    cv::namedWindow("Ex5", cv::WINDOW_AUTOSIZE);

    cv::VideoCapture capture;
    capture.open(0);
    cv::createTrackbar("Pyramid level", "Ex5", &g_slider_state, 3,
                       onTrackbarSlide); // [0..3]: num_states
                                         // trigger CallBack Function `onTrackbarSlide`
                                         // when g_state != g_slider_state

    capture >> bgr_frame;
    cv::pyrDown(bgr_frame, pyr_frame);
    std::string filename = "./live3.avi";
    for (;;)
    {
        capture >> bgr_frame;
        if (!bgr_frame.data)
        {
            break;
        }
        cv::setTrackbarPos("Pyramid level", "Ex5", g_state);

        char c = (char)cv::waitKey(10);
        if (c == '0')
        {
            g_state = py0;
        }
        if (c == '1')
        {
            g_state = py2;
        }
        if (c == '2')
        {
            g_state = py4;
        }
        if (c == '3')
        {
            g_state = py8;
        }
        if (c == 27)
            break;
    }
    char c = cv::waitKey(0);
}

```