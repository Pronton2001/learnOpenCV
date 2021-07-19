# chapter 4: Graphical User Interface

Tam Ky city, Quang Nam Province

20/7/2021

## HighGUI: Portable Graphics Toolkit

Allow us to interact with the operating system, the filesystem ,and hardware
such as cameras
3 part:

* Hardware (concerned with **camera**)
* Filesystem (loading and saving)
* GUI (display images in windows)

## Working with Image Files

### Loading and Saving Images

* Reading Files with cv::imread()
* Writing Files with cv::imwrite()

### Compression and Decompression

Compressing Files with cv::imencode()

* Return a simple character buffer

Uncompressing Files with cv::imdecode()

* decompress from a character buffer into an image array

## Working with Video

### Reading Video with the cv::VideoCapture Object

If the open is successful, the member function `cv::VideoCapture::isOpen()` will return *true*.

Some reason the open is fail:

* the file does not exist
* codec with which the video is compressed is not known

You can pass -1 to cv::VideoCapture::VideoCapture(), which will cause OpenCV to open a window that allows you to select the desired camera.

Reading Frames with `cv::VideoCapture::grab()` and `cv::VideoCapture::retrieve()`

* `cv::VideoCapture::grab()`: first grab all the frames and then come back and decode(`cv::VideoCapture::retrieve()`) them after you have them all safely in your **buffers**.
* cv::VideoCapture::retrieve(), which handles the decoding and the allocation and copying necessary to return the frame to you as a cv::Mat array

| `cv::VideoCapture::retrieve()`                                                                | `cv::VideoCapture::read()`              |
| --------------------------------------------------------------------------------------------- | --------------------------------------- |
| operates from the internal buffer from `cv::VideoCapture::grab()`                             | taking images and decoding them at once |
| argument `channel` indicate which image from the device is to be retrieved (multiple imagers) | no argument channel                     |

Camera Properties: cv::VideoCapture::get() and cv::VideoCapture::set()

### Writing Video with the cv::VideoWriter Object

Writing it out to disk.
`cv::VideoWriter::VideoWriter()`: constructor a *Video writer*
`cv::VideoWriter::isOpened()` method, which will return true if you are good to go.
`cv::VideoWriter::write()`: write frames to the the *Video writer*

## Working with Windows

Qt-based interface. Qt is a cross-platform toolkit, and so new features can be implemented only once in the library, rather than once each for each of the native platforms.

### HighGUI Native Graphical User Interface

Creating a Window with `cv::namedWindow()`

Creating a Window to free the memory with `cv::destroyWindow()`

Drawing an Image with `cv::imshow()`

Updating a Window and `cv::waitKey()`

There are a few other window-related functions:

* void cv::moveWindow( const char* name, int x, int y );
* void cv::destroyAllWindows( void );
* int cv::startWindowThread( void );

Mouse Events

* Unlike keyboard events, mouse events are handled by a more typical callback mechanism.
* need Call Back function to do stuff that response to mouse clicks

```c++
void your_mouse_callback(
	int event, // Event type (see Table 4-5)
	int x, // x-location of mouse event
	int y, // y-location of mouse event
	int flags, // More details on event (see Table 4-6) void* param // Parameters from cv::setMouseCallback()
);

void cv::setMouseCallback(
	const string& windowName, // Handle used to identify window
	cv::MouseCallback on_mouse, // Callback function
	void* param = NULL // Additional parameters for callback fn.
)
```

* Go to Page 140, example 4-3. Toy program for using a mouse to draw boxes on the screen

Sliders, Trackbars, and Switches

* In HighGUI, sliders are called trackbars.

No buttons

* Not support
* Use Sliders with only 2 position
* Use keyboard shortcuts

Switches are just sliders (trackbars) that have only two positions, “on” (1) and “off” (0)

## Working with the Qt Backend

skipped

## Interacting with OpenGL

Use OpenGL to render synthetic images and display them on top of camera of other processed images.

Application for : visualizing and debugging robotic or augmented-reality applications

Create a callback, The callback is then called every time the 
window is drawn. Your callback should match the prototype for `cv::OpenGLCallback()`: 

```c++
void your_opengl_callback(
	void* params // (Optional) Parameters from cv::createOpenGLCallback()
);
```

Once call back function is available, you configure OpenGL interface:

```c++
void cv::setOpenGlDrawCallback(
	const string& windowName, // Handle used to identify window
	cv::OpenGLCallback callback, // OpenGL callback routine
	void* params = NULL // (Optional) parameters for callback
);
```

Go to Page 154, example 4-5: Slightly modified code from the OpenCV documentation that draws a cube every frame; this 
modified version uses the global variables rotx and roty that are connected to the sliders in Figure 4-6

<!-- ### Integrating OpenCV with Full GUI Toolkits -->

Because this chapter is not important so I will skip the content of this Chapter from now on.
===