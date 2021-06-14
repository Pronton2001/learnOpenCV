# **Exercises**

## Download and install the latest release of OpenCV. Compile it in debug and release mode

---
Download from [OpenCV](https://github.com/opencv/opencv/releases).

## Download and build the latest trunk version of OpenCV using Git

---

In Windows:

```bash
    cd C:\Users\trihu
    mkdir OpenCV && cd OpenCV
    git clone git@github.com:opencv/opencv source
```

1. Download latest version CMAKE, Vscode, OpenCV and Visual Studio 2019 (build tool)
2. Set Variable Environment `OPENCV_DIR=C:\Users\trihu\OpenCV\opencv\build\x64\vc15` 
3. Include these into environment Path: `%OPENCV_DIR%\bin`, `%OPENCV_DIR%\lib`, `C:\Program Files\CMake\bin`
4. Download 2 extensions: Cmake and Cmake tool in VScode.
5. Create a folder in `C:\Users\trihu\OpenCV\dev\cmake-tool` for me.
6. If do not hace CMakeList.txt configuration yet, a notification will aprear to ask to configure automatically. I saw 1 sample in: [sample_CMakeList.txt](https://github.com/opencv/opencv/blob/master/samples/cpp/example_cmake/CMakeLists.txt)
7. In VSCode, in bottom bar, click to `Cmake` and choose Visual Studio Community 2019 release - amd64_arm64, choose Debug.
8. In VSCode, in bottom bar, click to `Build`, after finish, click on the next button `Launch`.

## Describe at least three ambiguous aspects of converting 3D inputs into a 2D representation. How would you overcome these ambiguities?

3 ambigious aspect:

1. variations in the world (weather, lighting, reflection, movements)
2. imperfections in lens and mechanical setup (components of camera?)
3. finite intergration time on sensor (motion blur)
4. electric noise.
5. compression artifacts after image capture

How to overcome these ambiguities:

1. In the design of a practical system, additional contextual knowledge can often be used to work around the limitations imposed on us by visual sensors

    `Note: Contextual information can also be modeled explicitly with machine learning techniques`
2. The use of a laser range finder to measure depth allows us to accurately infer the size of an object
3. There are 2 ways to deal with noise

    1. statistical method (For example for edge detect,it is cosistent orientation between a point and its neighbor in a local region)
    2. building explicit models learned directly from the available data.
4. The more constrained a computer vision context is, the more we can rely on those constraints to simplify the problem and the more reliable our final solution will be.

## What shapes can a rectangle take when you look at in with perspective distortion (that is, when you look at it in the real world)?

1. parallelogram when look from one side
2. trapezoid or rectangle when look directly
3. smaller rectangle when rotate

## Describe how you might start processing the image in Figure 1-1 to identify the mirror on the car?

On around the limitation of converting 3D to 2D representation allow computer to understand, I think:

1. Convert to Edge detection to get the only-edge image.
2. Seek for the big trapezoid or rectangle.
3. Because the mirror reflect the light, wee will see there is a circle after processing only-edge step will locate in the bottom right of the mirror.

## How might you tell the difference between a edges in an image created by

1. A shadow?
2. Paint on a surface?
3. Two sides of a brick?
4. The side of an object and the background?

No solution
