# chapter 3: Getting to Know OpenCV

Tam Ky city, Quang Nam Province

23/06 - 17/07/2021

## OpenCV Data Types

3 major categories:

 1. basic data types build from C++ primitives (*int*, *float*, etc.) including `simple vectors and matrices`, as well as representations of simple geometric concepts
like `points, rectangles, sizes, and the like`.

 2. contain *helper objects* represent more abstract concepts

 3. large array types contain arrays or other assemblies of primitives or the basic data types. These includes `cv::Mat` and `cv::SparseMat` for non-dense data such as histograms.

### Overview of the Basic Types

* `cv::Vec<>`: the fixed vector classes. Why not use STL C++? Becasue fixed vector intended for small vectors whose dimensions are known at compile time => *efficient code*. In fact the number of dimensions do not exceed 9 in any case.

> [!Short]
> The fixed vector classes as being handy and speedy for little guys.

* Anything of the form `cv::Vec{2,3,4,6}{b,s,i,f,d}` is valid for any combination of two to six dimensions2 and the five data types.

* **Asscociate**: fixed matrix classes `cv::Matx<>` used for 2x2, 3x3 and a few 4x4 matrix. `cv::Matx{1,2,3,4,6}{1,2,3,4,6}{f,d}.` is valid.

* **Asscociate**: point classes contain 2 or 3 values of 1 primitive object. `cv::Point2i, cv::Point2f, or cv::Point2d, or cv::Point3i, cv::Point3f, or cv::Point3d.`

> [!TIP]
> Point class can be cast to and from fixed vector classes.

| Point classes                                     | Fixed vector classes                  |
| ------------------------------------------------- | ------------------------------------- |
| Member accessed by name variable such as p.x, p.y | Member accessed by index (v[0], v[1]) |

### Basic Types: Getting Down to Details

1. `The Size Classes`:

   * can be cast to and from `the Point class`.

   * do not support casting to the fixed vector classes

2. `class cv::Rect`:

   * alias for the rectangle with integer member

   * locate relative to `upper-left corner` member

3. `class cv::RotatedRect`:

   * not template underneath but container, holding `cv::Point2f` called `center`, a `cv::Size2f` called `size`, and one additional `float` called `angle`

   * locate relative to `center` member

4. The Fixed Matrix Classes

   * *fixed* means "dimensions are known at compile time"

   > [!NOTE]
   > All memory loacted in stack => quickly allocate and clean up

   * used for doing matrix algebra

   * not used for big data array (image, etc.) which we should use `cv::Mat<>`

   > [!TIP]
   >many of function are static such as `eye()` that create a unit matrix, i.e we can call directly member of class(static object) rather than member of instantiation of that class (`cv::Mat33f::eye()`).

5. The Fixed Vector Classes

   * derived from the fixed matrix classes

   * `cv::Vec<>` is a `cv::Matx<>` whose number of columns is one

   * `w`, which indicates an `unsigned short`

6. The Complex Number Classes

   * can be cast to or from class STL complex number `complex<>`

   * The most different between **OpenCV** and **STL** complex number classes if in member access: member `re` and `im` in **OpenCV** instead of function `real()` and `imag()` in **STL**

### Helper Object

1. `class cv::TermCriteria`
   * finite number of iteration (called `COUNT` or `MAX_ITER`)

   * error parameter says *"if you are this close, you can quit"* (called `ESP` - short for epsilon)
2. `class cv::Range`

   * like `range(x)` in Python. For example, `cv::Range rng( 0, 4 )` include the value 0, 1, 2, 3 but not 4

   * using `cv::Range::size()` to get the number of elements in a range
  
3. The `cv::Ptr` template, and Garbage Collection 101

   * smart pointer allow 2 different pointer `p` and `q` pointing to just 1 object `cv::Mat33f`, which will increase the number of counter to 2 and both `p` and `q` know that they are each one of two pointers. And if `p` is disapear (out of scope), `q` will notice that it is the last one left. And if  `p` disapear, it should deallocate `cv::Mat33f`.
  
      > [!SHORT]
      > You can think of this like the last person out of a building being responsible for turning out the lights and locking the door

   * Some funtion: `empty()`, `addref()` and `release()`, `delete_obj()`.

   * Some clean-up function does not exist for what you want, you will have to define it yourself. For example,

      ```c
      template<> inline void cv::Ptr<FILE>::delete_obj() {
         fclose(obj); 
      }
      ```

      > [!IMPORTANT]
      > A tip for gurus: a serious programmer might worry that the incrementing and decrementing of the reference count might not be sufficiently atomic for the Ptr<> template to be safe in `multithreaded applications`. This, however, is not the case, and Ptr<> is thread safe. Similarly, the other reference counting objects in OpenCV are all `thread-safe` in this same sense.

4. class `cv::Exception` and Exception Handling

   * derived from the STL exception class std::exception.

5. The `cv::DataType<>` Template
   * contain runtime information about the type
   * contain typedef statements in its own definition that allow it to refer to the same type at compile time

      > [!CAUTION]
      > I will skip this and learn later.
      >
      >Key word: `channel_type`, `CV::S32`, `0x3069`, etc.

6. class InputArray and class OutputArray

   * Pass and return not 1 but many number (array)

   * `cv::InputArray` and `cv::OutputArray`, these types mean “any of the above”
   * `cv::noArray()`, `cv::InputOutputArray`

### Large Array Types

* The overwhelming majority of functions in the OpenCV library are members of the cv::Mat class or take a cv::Mat as an argument or return cv::Mat as a return value.

#### **class `cv::Mat`: N-Dimensional Dense Arrays**

* raster scan order

* contains an element signaling a `flags`

* dimension :`dims`

* A reference counter : `cv::Ptr<>` called **refcount**

* `step[]`, `data`, ...

#### Creating an Array

* Types of array: `cv::{U8,S16,U16,S32,F32,F64}C{1,2,3}.`For example, cv::F32C3 would imply a 32-bit floating-point three-channel array.

```c++
cv::Mat m;
m.create( 3, 10, cv::F32C3 ); // 3 rows, 10 columns of 3-channel 32-bit floats
m.setTo( cv::Scalar( 1.0f, 0.0f, 1.0f ) ); // 1st channel is 1.0, 2nd 0.0, 3rd 1.0
```

is equivalent to:

```c++
cv::Mat m( 3, 10, cv::F32C3, cv::Scalar( 1.0f, 0.0f, 1.0f ) )
```

* when  assign one matrix n to another matrix : `allocate`, `deallocate` will share.

#### Accessing Array Elements Individually

* use `at<>()`
* a more sophisticated type:

```c++
cv::Mat m = cv::Mat::eye( 10, 10, cv::DataType<cv::Complexf>::type );
printf(
   “Element (3,3) is %f + i%f\n”, 
   m.at<cv::Complexf>(3,3).re, 
   m.at<cv::Complexf>(3,3).im, 
)
```

* use pointer `ptr<>` (fatest way). For example, if the array type is F32C3 , `mtx.ptr<Vec3f>(3)` return pointer to the first channel of the fisrt element in row `3` of `mtx`.

* use iterators : `cv::MatIterator<>`, `cv::MatConstIterator<>`

```c++
int sz[3] = { 4, 4, 4 };
cv::Mat m( 3, sz, cv::F32C3 ); // A three-dimensional array of size 4-by-4-by-4
cv::randu( m, -1.0f, 1.0f ); // fill with random numbers from -1.0 to 1.0 
float max = 0.0f; // minimum possible value of L2 norm
cv::MatConstIterator<cv::Vec3f> it = m.begin();
while( it != m.end() ) {
   len2 = (*it)[0]*(*it)[0]+(*it)[1]*(*it)[1]+(*it)[2]*(*it)[2];
   if( len2 > max ) max = len2;
   it++;
}
```

#### The N-ary Array Iterator: `NAryMatIterator`

* another form of iteration
  
* allows us to handle iteration over many arrays at once

* required only all of the arrays that are being
iterated over be of the same geometry (dims & extents in each dim)

* returning chunks of arrays, called `planes` instead of an element. `Plane`(1 or 2 dims) is a portion of input array in which data is guaranteed to be contigous in memory.

`Do not need to check for discontinuities inside chunks (planes)`

```c++
const int n_mat_size = 5;
const int n_mat_sz[] = { n_mat_size, n_mat_size, n_mat_size };
cv::Mat n_mat( 3, n_mat_sz, cv::F32C1 ); // 3 rows, n_mat_sz(=5) columns, 1 chanel with type float 32-bit
cv::RNG rng;
rng.fill( n_mat, cv::RNG::UNIFORM, 0.f, 1.f ); // fill it with 125 random floating point numbers between 0.0 and 1.0

const cv::Mat* arrays[] = { &n_mat, 0 };
cv::Mat my_planes[1];
cv::NAryMatIterator it( arrays, my_planes );
// On each iteration, it.planes[i] will be the current plane of the
// i-th array from ‘arrays’.
//
float s = 0.f; // Total sum over all planes
int n = 0; // Total number of planes
for (int p = 0; p < it.nplanes; p++, ++it) {
   s += cv::sum(it.planes[0])[0]; 
   n++; 
 } 
```

* To construct a `cv::NAryMatIterator` object, we need:
  * array containing pointers to all `cv::Mat`'s (in example, there is just 1). This array must always be terminated with a `0` or `NULL`.

  * another array `cv::Mat`'s used to refer to the individual `planes` (in example, there is also 1)

* Another example in which 2 arrays will sum over:

```c++
const int n_mat_size = 5;
const int n_mat_sz[] = {n_mat_size, n_mat_size, n_mat_size};
cv::Mat n_mat0(3, n_mat_sz, CV_32FC1); //3 dims : 5x5x5, 1 channel-32bit
cv::Mat n_mat1(3, n_mat_sz, CV_32FC1);
cv::RNG rng;
rng.fill(n_mat0, cv::RNG::UNIFORM, 0.f, 1.f);
rng.fill(n_mat1, cv::RNG::UNIFORM, 0.f, 1.f);

const cv::Mat *arrays[] = {&n_mat0, &n_mat1, 0};
cv::Mat my_planes[2];
cv::NAryMatIterator it(arrays, my_planes);
float s = 0.f; // Total sum over all planes in both arrays
int n = 0;     // Total number of planes
cout << "number:" << it.nplanes << '\n';
for (int p = 0; p < it.nplanes; p++, ++it)
{
   cout << "element:" << it.planes[0] << "\n";
   cout << "sum1:" << sum(it.planes[0]) << '\n';
   cout << "sum2:" << sum(it.planes[1]) << '\n';
   s += cv::sum(it.planes[0])[0];
   s += cv::sum(it.planes[1])[0];
   n++;
}
cout << "Sum:" << s;
```

* Result

```dos

C:\Users\trihu\OpenCV\dev\project2\build\Debug>"C:\Users\trihu\OpenCV\dev\project2\build\Debug\cmake-tool.exe"
number:1
element:[0.5302828, 0.19925919, 0.40105945, 0.81438506, 0.43713298, 0.2487897, 0.77310503, 0.76209372, 0.30779448, 0.70243168, 0.4784472, 0.79219002, 0.085843116, 0.075060248, 0.16342339, 0.29977924, 0.90565395, 0.7096858, 0.1497125, 0.76543993, 0.12428141, 0.0037285984, 0.55151361, 0.99819934, 0.15899214, 0.16117805, 0.038189322, 0.42586133, 0.84612763, 0.88760448, 0.26471239, 0.27078307, 0.9526664, 0.67513132, 0.81021637, 0.1839602, 0.094456196, 0.81590331, 0.52451766, 0.38083884, 0.41199476, 0.25553533, 0.94489586, 0.90012348, 0.45789155, 0.84130692, 0.28030014, 0.3854357, 0.45190147, 0.87543356, 0.15941507, 0.64928663, 0.42545688, 0.67995042, 0.34952945, 0.60071671, 0.27786469, 0.64982021, 0.43575865, 0.78762269, 0.30389759, 0.077580065, 0.63589597, 0.66734934, 0.7477535, 0.32611883, 0.3744764, 0.54064941, 0.035140187, 0.31352437, 0.81599391, 0.66986996, 0.64654934, 0.093020111, 0.80687904, 0.81840253, 0.87030983, 0.015843809, 0.82176256, 0.27341431, 0.74286938, 0.87833691, 0.25319767, 0.21054748, 0.4704462, 0.18503526, 0.60176688, 0.55045521, 0.52363247, 0.2119506, 0.77888972, 0.0016515851, 0.41418922, 0.77817345, 0.3332077, 0.79873753, 0.89428955, 0.8482796, 0.93545049, 0.46671736, 0.84443766, 0.037859529, 0.72475517, 0.5581553, 0.85918629, 0.13863379, 0.54506946, 0.83290988, 0.17355248, 0.10345989, 0.26784888, 0.12765667, 0.22433367, 0.52702159, 0.62657177, 0.36008775, 0.46901485, 0.61760253, 0.98908836, 0.10372096, 0.69157451, 0.56235802, 0.63620704, 0.54051489, 0.74736416]
sum1:[62.6319, 0, 0, 0]
sum2:[63.7084, 0, 0, 0]
Sum:126.34
```

* In the second example, 2 given pointeses to both inputs arrays,2 matrices are supplied in the `my_planes` array

* `it.size`: indicates the size(# of elements) of each plane

#### Accessing Array Elements by Block

* ` m2 = m.row(3); ` means to create a new array header m2, and to arrange its data pointer, *step* array, and so on, such that it will access the data in row 3 in m. This is very helpful that `if you modify the data in m2, you will be modifying the data in m`, which use can easy to change the original matrix using its blocks such as rows, columns, submatrix(Rect), ...

* In compared to the later mention method `CopyTo()`, the main advantage of a new array that accesses part of an existing array, the **TIME REQUIRED** is `fast` and `independent` of the size of either the old or new array

* Related to `row()`, `col()` are `rowRange()` and `colRange()`, they will extract an array with multiple contigous rows (or columns)

* `diag(offset)`: take diagonal + offset elements of a matrix. offset can be negative (lower half), positive (upper-half) and 0 (main diagonal)

* `operator()`: extract a sub-volumne from a higer-dimensional array. Args: `Range` of rows and cols or `Rect`. Required: a pointer to s-style array of ranges; array must have as many elements as the number of dimensions of the array

#### Matrix Expressions: Algebra and cv::Mat

* `cv::MatExpr`: symbolic representation of the algebra from of the *right- hand- size*. It has some advantages likes clear,...

> [!NOTE]
> vector cross product is only defined for 3-by-1 matrices

* `operator=()` is not assigning a `cv::Mat` to a cv::Mat (as it might appear), but rather a cv::MatExpr (the expression itself) to a cv::Mat.
  * `m2=m1` means m2 would be another reference to the data in m1. m2 change -> m1 change
  * By contrast, `m2=m1+m0` means something different again. Because m1+m0 is a matrix expression, it will be evaluated and a pointer to the results will be assigned in m2. The results will reside in a newly allocated data area, which do not affect to m1 and m0.

> [!EXPERT]
> If you are a real expert, this will not surprise you. Clearly a temporary array must be created to store the result of m1+m0. Then m2 really is just another reference, but it is another reference to that temporary array. When operator+() exits, its reference to the temporary array is discarded, but the reference count is not zero. m2 is left holding the one and only reference to that array.

* `inv()`, `cv::DECOMP_LU`(LU decomposition, works for any nonsingular matrix?), `cv::DECOMP_CHOLESKY`(only works for symmetric, positive definite matrices and faster than `LU`), `CV::DECOMP_SVD`(the only workable option for matrices are singular or not even squares)

`Saturation Casting`:

* Automatically check underflow and overflow, i.e convert to lowest or highest available value

#### Accessing Sparse Array Elements

* four different access mechanisms:`cv::SparseMat::ptr()`, `cv::SparseMat::ref()`, `cv::SparseMat::value()`, and `cv::SparseMat::find()`. For example,

```c
uchar* cv::SparseMat::ptr( int i0, bool createMissing, size_t* hashval=0 )
```

* one-dimensional array with position `i0`, `createMissing` indicates whether the element should be created if it is not already present in the array. `cv::SparseMat` is as a hash table(find key and compute hash value). Normally, hash list will be short, the primary computational cost in a lookup is the computation of hash key. If this key has already been computed then time can be saved by not recomputing it. By default, `size_t* hashval=0`(NULL), the hash key will be computed again. If, however, a key is provided, it will be used.

```c++
#include <opencv2/opencv.hpp>
#include <iostream>
using namespace std;
int main()
{
   // Create a 10x10 sparse matrix with a few nonzero elements
   //
   int size[] = {10, 10};
   cv::SparseMat sm(2, size, CV_32F);
   for (int i = 0; i < 10; i++)
   { // Fill the array
      int idx[2];
      idx[0] = size[0] * rand();
      idx[1] = size[1] * rand();
      sm.ref<float>(idx) += 1.0f;
   }
   // Print out the nonzero elements
   //
   cv::SparseMatConstIterator_<float> it = sm.begin<float>();
   cv::SparseMatConstIterator_<float> it_end = sm.end<float>();
   for (; it != it_end; ++it)
   {
      const cv::SparseMat::Node *node = it.node();
      printf(" (%3d,%3d) %f\n", node->idx[0], node->idx[1], *it);
   }
}
```

* Result:

```dos
C:\Users\trihu\OpenCV\dev\project2\build\Debug>"C:\Users\trihu\OpenCV\dev\project2\build\Debug\cmake-tool.exe"
 (99610,4910) 1.000000
 (232810,168270) 1.000000
 (114780,293580) 1.000000
 (410,184670) 1.000000
 (29950,119420) 1.000000
 (191690,157240) 1.000000
```

#### Functions Unique to Sparse Arrays

page 67

### The Template Structure

* Knowing how to use the templates directly can be of great help in getting things done.

* When you instantiate an object of type `cv::Point`, you are actually instantiating a template object of type `cv::Point_<int>`.

> [!NOTE]
> trailing underscore indicate a template.

* This concept is same to other types such as `cv::Scalar_<>` and `cv::Rect_<>`, as well as `cv::Matx_<>` and `cv::Vec_<>`

#### cv::Mat_<> and cv::SparseMat_<> Are a Little Bit Different

* cv::Mat and cv::Mat_<>, their relationship is not so simple.

* Benefit:
  * Don’t have to use the template forms of their member functions. For example,

   ```cpp
   cv::Mat m( 10, 10, cv::F32C2 );
   m.at< Vec2f >( i0, i1 ) = cv::Vec2f( x, y );
   ```

   is not recommend, instead we use template for simplification and efficiency coding:

   ```cpp
   cv::Mat_<Vec2f> m( 10, 10 );
   m.at( i0, i1 ) = cv::Vec2f( x, y );
   // or…
   m( i0, i1 ) = cv::Vec2f( x, y );
   ```

  * More *correct* because it allows the compiler to detect type mistakes when passing into a function.
  
   ```c++
   cv::Mat m(10, 10, cv::F32C2 );
   ```

   is passed into

   ```cpp
   void foo((cv::Mat_<char> *)myMat);
   ```

   failure would occur during runtime in perhaps nonobvious ways. If you instead used

   ```c++
   cv::Mat_<Vec2f> m( 10, 10 );
   ```

   failure would be detected at compile time.

* Create template functions operating on an array of a particular type.

### Array Operators

* “friend” functions that either take array types as arguments, have array types as return values, or both

* Rules:
  * Saturation
    * Outputs of calculations are saturation casted to the type of the output array

  * Output
    * The output array will be created with cv::Mat::create() if its type and size do not match the inputs.

  * Scalar
    * array->scalar

  * Masks
    * Like filter before computing (such as non-zero filter)

  * dtype
    * changable and automtically match inputs' dtype(default value -1)
  
  * In Place Operation
    * unless otherwise specified, input and output array are of the same size and type for any operation

  * Multichannel
    * process separately

* `cv::addWeighted()` used to implement **alpha blending** *[Smith79; Porter84]*

```cpp
// Example 3-1. Complete program to alpha blend the ROI starting at (0,0) in src2 with the ROI starting at ( x, y) in src1
// alphablend <imageA> <image B> <x> <y> <width> <height> alpha> <beta>
//
cv::Mat src1 = cv::imread(argv[1], 1);
cv::Mat src2 = cv::imread(argv[2], 1);
if (argc == 9 && !src1.empty() && !src2.empty())
{
   int x = atoi(argv[3]);
   int y = atoi(argv[4]);
   int w = atoi(argv[5]);
   int h = atoi(argv[6]);
   double alpha = (double)atof(argv[7]);
   double beta = (double)atof(argv[8]);
   cv::Mat roi1(src1, cv::Rect(x, y, w, h));
   cv::Mat roi2(src2, cv::Rect(0, 0, w, h));
   cv::addWeighted(roi1, alpha, roi2, beta, 0.0, roi1);
   cv::namedWindow("Alpha Blend", 1);
   cv::imshow("Alpha Blend", src2);
   cv::waitKey(0);
}
```

Result : In page 76

* `cv::calcCovarMatrix()`: compute mean and covariance matrix for the Gaussian approximation.

* `cv::Mahalanobis()`

   $$
   r_{mahalonobis} = \sqrt{(\vec{x} - \vec{u})^T\Sigma^{-1}(\vec{x} - \vec{u})}
   $$

`Mahalanobis` distance is defined as the vector distance measured between a point and the center of a Gaussian distribution. This is analogous to the z-score in basic statisitc

>[!WARNING]
> It is good programming practice to use the cv::DECOMP_SVD method for this inversion because someday you will encounter a distribution for which one of the eigenvalues is 0!

* `cv::randu()`: Uniform distribution

* `cv::randn()`: Normal distribution

> [!NOTE]
> Gaussian-distribution random numbers are generated using the Ziggurat algorithm [G. Marsaglia, W. W. Tsang].

* `cv::randShuffle()`: change entries positions randomly

* `cv::reduce()`: same as `tf.reduce_max()`,... in Python.

* `cv::transform()`: linear transformation

### Utility Functions

* `cv::fastAtan2()`: Calculate two-dimensional angle of a vector in degrees

* `cv::getThreadNum()`: Get index of the current thread

* `cvIsInf()`: Check if a floating-point number x is infinity

* `cvIsNaN()`: Check if a floating-point number x is “Not a Number”

* `cv::setNumThreads()`: Set number of threads used by OpenCV

* `cv::setUseOptimized()`: Enables or disables the use of optimized code (SSE2, etc.)

* `cv::useOptimized()` Indicates status of optimized code enabling (see cv::setUseOptimized())

* `cv::getCPUTickCount()`:  reports the number of CPU ticks on those architecture, but difficult to return value that must interpret on many architecture. However, , it is best to use cv::getTickCount() for timing measurements. This function is best for tasks like initializing random number generators

## Objects That Do Stuff

* Called functors or object functions. [What is functors in cpp?](https://www.geeksforgeeks.org/functors-in-cpp/)

* In short, they are too complicated to be associate with a normal function, such as `void SumOfTwoNumber(){};`, they are objects instantiated from their own class, most of them has overloaded `operator()` just to call like a function

### Principal Component Analysis (`cv::PCA`)

* process of analyzing a distribution in many dimensions and extracting from that distribution the particular subset of dimensions that carry the most information.

* not necessarily the basis dimensions

* most important aspects: generate a new basis in which the axes ranked by their importance.

>[!REMIND]
>Covariance Matrix is a square matrix giving that the covariance between each pair.
>$${\displaystyle \mathbf {X} =(X_{1},X_{2},...,X_{n})^{\mathrm {T} }}$$
>$${\displaystyle \Sigma_{i,j} = \operatorname {K}_{X_{i}X_{j}}=\operatorname {cov} [X_{i},X_{j}]=\operatorname {E} [(X_{i}-\operatorname {E} [X_{i}])(X_{j}-\operatorname {E} [X_{j}])]}$$
>$${\displaystyle \Sigma = \operatorname {K}_{\mathbf {X} \mathbf {X} }={\begin{bmatrix}\mathrm {E} [(X_{1}-\operatorname {E} [X_{1}])(X_{1}-\operatorname {E} [X_{1}])]&\mathrm {E} [(X_{1}-\operatorname {E} [X_{1}])(X_{2}-\operatorname {E} [X_{2}])]&\cdots &\mathrm {E} [(X_{1}-\operatorname {E} [X_{1}])(X_{n}-\operatorname {E} [X_{n}])]\\\\\mathrm {E} [(X_{2}-\operatorname {E} [X_{2}])(X_{1}-\operatorname {E} [X_{1}])]&\mathrm {E} [(X_{2}-\operatorname {E} [X_{2}])(X_{2}-\operatorname {E} [X_{2}])]&\cdots &\mathrm {E} [(X_{2}-\operatorname {E} [X_{2}])(X_{n}-\operatorname {E} [X_{n}])]\\\\\vdots &\vdots &\ddots &\vdots \\\\\mathrm {E} [(X_{n}-\operatorname {E} [X_{n}])(X_{1}-\operatorname {E} [X_{1}])]&\mathrm {E} [(X_{n}-\operatorname {E} [X_{n}])(X_{2}-\operatorname {E} [X_{2}])]&\cdots &\mathrm {E} [(X_{n}-\operatorname {E} [X_{n}])(X_{n}-\operatorname {E} [X_{n}])]\end{bmatrix}}}$$

* The big advantage of the new basis is that the basis vectors that correspond to the large eigenvalues carry most of the information.

* Reduce less infomation dimensions using a KLT Transform.(KLT stands for “Karhunen-Loeve Transform,””)

* `cv::PCA::project()`, `cv::PCA::backProject()` in page 118

### Singular Value Decomposition (`cv::SVD`)

* similar to `cv::PCA`, but different purpose. Work with non-square, ill-conditioned, or otherwise poorly-behaved matrices such those encountered when solving underdetermined linear system

* decomposition of matrix m-by-n matrix A in the form:
$$ A = U\times W \times V^T $$
where W is a **diagonal matrix** and U and V are m -by- n and n -by- n (unitary) matrices, W is also m-by-n matrix whose row and column number are not equal is *zero*.

* `cv::SVD::solveZ()` in pages 118-120

### Random Number Generator (`cv::RNG`)

* pseudorandom sequence that generates random numbers.

* The generator uses the Multiply with Carry algorithm (G. Marsaglia) for uniform distributions and the Ziggurat algorithm (G. Marsaglia and W.  W. Tsang) for the generation of numbers from a Gaussian distribution

## Summary

* encounter basic data structures.
* Looked at the most important DS `cv::Mat` which can contain images, matrices and multichannel arrays.

## Exercises

Need to refer to the manual at [http://docs.opencv.org/]

### 1. Open […/opencv/modules/core/include/opencv2/core/core.hpp]. Find helper functions

1. Choose a negative floating-point number. Take its absolute value, round it, and then take its ceiling and floor.
2. Generate some random number
3. Create a floating-point cv::Point2f and convert it to an integer cv::Point2i
4. Convert a cv::Point2i to a CvPoint2f.

Object functions and objects:

* class CV_EXPORTS Exception : public std::exception
* enum ReduceTypes
* enum KmeansFlags
* void add, multiply, subtract, divide, ...
* class CV_EXPORTS PCA
* class CV_EXPORTS RNG

```c++
float n_neg = -3.4f;
float n_abs = cv::abs(n_neg);

int n_round = cvRound(n_abs);
int n_ceil = cvCeil(n_abs);
int n_floor = cvFloor(n_abs);

cout << n_neg << ' ' << n_abs << ' ' << n_round << ' ' << n_ceil << ' ' << n_floor << '\n';

float n_ran = cv::theRNG().uniform(2.f, 5.f);
float n_ran2 = cv::theRNG();
cv::RNG n_ran3 = cv::RNG(4);
cout << "Random:" << n_ran << '\n';
cout << "Cooler random generator: \n";
cout << "A: " << int(n_ran2) << '\n';
cout << "A: " << (int)n_ran2 << '\n';
cout << "a: " << (float)n_ran2 << '\n';
cout << "a: " << float(n_ran2) << '\n';
cout << "A: " << int(n_ran3) << '\n';
cout << "B: " << (int)n_ran3 << '\n';
cout << "C: " << (float)n_ran3 << '\n';
cout << "D: " << float(n_ran3) << '\n';

cv::Point2f p_float = cv::Point2f(1.2f, 2.6f);
cv::Point2i p_int1 = cv::Point2i(p_float); //* constructor
cv::Point2i p_int2 = (cv::Point2i)p_float; //* casting
cout << "Point:" << p_int1.x << ',' << p_int1.y << '\n';
cout << "Point:" << p_int2.x << ',' << p_int2.y << '\n';
return 0;
```

### 2. This exercise will accustom you to the idea of many functions taking matrix types. Create a two dimensional matrix with three channels of type byte with data size 100-by-100. Set all the values to 0

1. Draw a circle in the matrix using the cv::circle() function:

   ```c++
   void circle( 
   cv::Mat& img, // Image to be drawn on
   cv::Point center, // Location of circle center
   int radius, // Radius of circle
   const cv::Scalar& color, // Color, RGB form
   int thickness = 1, // Thickness of line 
   int lineType = 8, // Connectedness, 4 or 8
   int shift = 0 // Bits of radius to treat as fraction
   );
   ```

2. Display this image using methods described in Chapter 2.

   ```c++
   int mat_size[] = {100, 100};

   cv::Mat mat = cv::Mat(2, mat_size, CV_8UC3, cv::Scalar(0,0,0));

   // ? Another way to initialize cv::Mat
   // auto type = cv::traits::Type<cv::Vec<uchar, 3>>::value;
   // cv::Mat mat = cv::Mat(2, mat_size, type);
   // mat.setTo(cv::Scalar(0.0f, 0.0f, 0.0f));

   // 1. Draw a circle in the matrix using the cv::circle() function:
   cv::circle(mat, cv::Point2i(50, 50), 10, cv::Scalar(0, 255, 0), 1, cv::LINE_AA);

   cv::imshow("Mat Circle", mat);
   //2. Display this image using methods described in Chapter 2.
   assert(argc == 2);
   cv::Mat src = cv::imread(argv[1], cv::IMREAD_COLOR);
   if (src.empty())
   {
      printf(" Error opening image\n");
      printf(" Program Arguments: [image_name -- default %s] \n", argv[1]);
      return EXIT_FAILURE;
   }
   cv::circle(src, cv::Point2i(src.cols / 2, src.rows / 2), 10, cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
   cv::imshow("Image Circle", src);

   cv::waitKey();

   return 0;
   ```

### 1. Create a two-dimensional matrix with three channels of type byte with data size 100-by-100, and set all the values to 0. Use the element access member: m.at<cv::Vec3f> to point to the middle (“green”) channel. Draw a green rectangle between (20, 5) and (40, 20)

```c++
int mat_size[] = {100, 100};

cv::Mat m= cv::Mat(2, mat_size, CV_32FC3, cv::Scalar(0.0f, 0.0f, 0.0f));

// * Use for loop to draw individual point
for (int i=20; i<=40; i++){
   for (int j=5; j<=20; j++){
      m.at<cv::Vec3f>(i,j) = 1.0f;
   }
}
// * Use draw function
// cv::rectangle(m,cv::Rect(20,5,20,15),cv::Scalar(0.0,1.0,0.0),1,cv::LINE_AA);
cv::imshow("Rectangle", m);
cv::waitKey();
```

### 2. Create a three-channel RGB image of size 100-by-100. Clear it. Use pointer arithmetic to draw a green square between (20, 5) and (40, 20)

```c++
int mat_size[] = {100, 100};

cv::Mat m = cv::Mat(2, mat_size, CV_32FC3, cv::Scalar(0.0f, 0.0f, 0.0f));

for (int r =5; r <= 20; r++ )
{
   cv::Vec3f* ptr = m.ptr<cv::Vec3f>(r);
   for (int c = 20; c <= 40; c++)
   {
      ptr[c] = cv::Vec3f(0,1,0);
   }
}
cv::imshow("Rectangle", m);
cv::waitKey();
```

### 3. Practice using the block access methods (Table 3-16). Create a 210-by-210 single-channel byte image and zero it. Within the image, build a pyramid of increasing values using the submatrix constructor and the cv::Range object. That is: the outer border should be 0, the next inner border should be 20, the next inner border should be 40, and so on until the final innermost square is set to value 200; all borders should be 10 pixels wide. Display the image

```c++
typedef cv::Vec<uchar, 1> Vec1b;
using namespace std;

int main(int argc, char **argv)
{
   int mat_size[] = {210, 210};
   cv::Mat m = cv::Mat(2, mat_size, CV_8UC1, Vec1b((uchar)0));
   int color = 0;
   int num_border = (m.rows / 2) / 10 - 1;
   for (int i = 10; i < num_border * 10; i += 10)
   {
      color += 20;
      for (int r = i; r <= m.rows - i; r++)
      {
         Vec1b *ptr = m.ptr<Vec1b>(r);
         for (int c = i; c <= m.rows - i; c++)
         {
            ptr[c] = Vec1b(color);
         }
      }
   }
   cv::imshow("Rectangle", m);
   cv::waitKey();
   return 0;
}
```

In above code, I used `Vec1b` but I realized that I complicated, just use `uchar` is enough. And `float` also is better than `Vec<float, 1>` or `Vec1f`.

```c++
int mat_size[] = {210, 210};
cv::Mat m = cv::Mat(2, mat_size, CV_8UC1, cv::Scalar((uchar)0));
int color = 0;
int num_border = (m.rows / 2) / 10 - 1;
for (int i = 10; i < num_border * 10; i += 10)
{
   color += 20;
   cv::Mat sub_m = m(cv::Rect(i,i,210 - 2*i, 210 - 2*i));
   for (int r = 0; r < sub_m.rows; r++)
   {
      uchar *ptr = sub_m.ptr<uchar>(r);
      for (int c = 0; c < sub_m.cols; c++)
      {
         ptr[c] = uchar(color);
      }
   }
}
cv::imshow("Rectangle", m);
cv::waitKey();
return 0;
```

### 4. Use multiple image objects for one image. Load an image that is at least 100-by-100. Create two additional image objects using the first object and the submatrix constructor. Create the new images with width at 20 and the height at 30, but with their origins located at pixel at (5, 10) and (50, 60), respectively. Logically invert the two images using the ‘not’ logical inversion operator. Display the loaded image, which should have two inverted rectangles within the larger image

```c++
cv::Mat m;
if (argc == 1)
{
   int mat_size[] = {100, 100};
   m = cv::Mat(2, mat_size, CV_8UC1, cv::Vec3f(0.0, 0.0, 0.0));
}
else if (argc == 2){
   m = cv::imread(argv[1]);
}
else {
   cout << "Use " << argv[0] " <image path>";
   return -1;
}
cv::Mat invRect1 = m(cv::Rect(5, 10, 20, 30));
cv::Mat invRect2 = m(cv::Rect(50, 60, 20, 30));
cv::bitwise_not(invRect1, invRect1);
cv::bitwise_not(invRect2, invRect2);
cv::imshow("Rectangle", m);
cv::waitKey();
```

### 5. Add an CV_DbgAssert( condition ) to the code of question 4 that will be triggered by a condition in the program. Build the code in debug, run it and see the assert being triggered. Now build it in release mode and see that the condition is not triggered

```c++
// Check condition fail to trigger in just Debug mode 
CV_DbgAssert(argc == 2);
// Check condition fail to trigger in both Debug and Release mode
CV_Assert(argc == 2);
```

### 6. Create a mask using cv::compare(). Load a real image. Use cv::split() to split the image into red, green, and blue images

Some supplement function:

```c++
string type2str(int type)
{
   string r;

   uchar depth = type & CV_MAT_DEPTH_MASK;
   uchar chans = 1 + (type >> CV_CN_SHIFT);

   switch (depth)
   {
   case CV_8U:
      r = "8U";
      break;
   case CV_8S:
      r = "8S";
      break;
   case CV_16U:
      r = "16U";
      break;
   case CV_16S:
      r = "16S";
      break;
   case CV_32S:
      r = "32S";
      break;
   case CV_32F:
      r = "32F";
      break;
   case CV_64F:
      r = "64F";
      break;
   default:
      r = "User";
      break;
   }

   r += "C";
   r += (chans + '0');

   return r;
}

template <typename T>
void cvMax1(cv::Mat_<T> &mat, T &minVal, T &maxVal)
{
   CV_Assert(type2str(mat.type()) == "8UC1" || type2str(mat.type()) == "32FC1");
   CV_Assert(mat.depth()!= sizeof(uchar));
   T max = 0;
   T min = 255;
   uchar* ptr;
   for (int r = 0; r < mat.rows; r++)
   {
      ptr = mat.ptr<uchar>(r);
      for (int c = 0; c < mat.rows; c++)
      {
         if (max < ptr[c])
         {
            max = ptr[c];
         }
         if (min > ptr[c])
         {
            min = ptr[c];
         }
      }
   }
   minVal = min;
   maxVal = max;
};
```

1. Find and display the green image

   ```c++
   CV_Assert(argc == 1);
   cv::Mat_<cv::Vec3b> m = cv::imread("C:/Users/trihu/OpenCV/dev/project2/image/anh4.jpg");
   CV_Assert(m.data);

   imshow("Origin image", m);
   cv::Mat_<uchar> rgbChannel[3];
   cv::split(m, rgbChannel);

   cv::imshow("green", rgbChannel[1]);
   ```

2. Clone this green plane image twice (call these clone1 and clone2).

   ```c++
   cv::Mat clone1, clone2;
   rgbChannel[1].copyTo(clone1);
   rgbChannel[1].copyTo(clone2);
   ```

3. Find the green plane’s minimum and maximum value

   ```c++
   string ty = type2str(m.type());
   printf("Matrix: %s %dx%d \n", ty.c_str(), m.cols, m.rows);
   ty = type2str(rgbChannel[1].type());
   printf("Matrix: %s %dx%d \n", ty.c_str(), rgbChannel[1].cols, rgbChannel[1].rows);

   uchar min, max;
   cvMax1<uchar>(rgbChannel[1], min, max);
   cout << "Min, Max value in green's field:" << (int)min << "," << (int)max << "\n";

   double mi, ma;
   cv::Point pmi, pma;
   cv::minMaxLoc(rgbChannel[1], &mi, &ma, &pmi, &pma);
   cout << "Min, Max value in green's field:" << mi << "," << ma << '\n';
   ```

4. Set clone1’s values to thresh = (unsigned char)((maximum - minimum)/2.0).

   ```c++
   uchar thresh = (uchar)((max - min) / 2.0);

   cv::Mat_<uchar> clone1_new;
   // THRESH_TRUNC:  + dst(x,y) = src(x,y) if src(x,y) < threshold
   //                + dst(x,y) = thresh if src(x,y) > threshold
   cv::threshold(clone1, clone1_new, thresh, -1, cv::THRESH_TRUNC);
   cv::imshow("Threshold", clone1_new);
   ```

5. Set clone2 to 0 and use cv::compare(green_image, clone1, clone2, cv::CMP_GE). Now clone2 will have a mask of where the value exceeds thresh in the green image.

   ```c++
   clone2.setTo((uchar)0);
   cv::compare(rgbChannel[1], clone1_new, clone2, cv::CMP_GT);
   // equivalent to: clone2 = rgbChannel[1] > clone1_new;

   cv::imshow("Exceed Threshould", clone2);
   ```

6. Finally, compute the value: green_image = green_image - thresh/2 and display the results. (Bonus: assign this value to green_image only where clone2 is non-zero.)

   ```c++
   cv::Mat_<uchar> mask = rgbChannel[1] != 0; // every element == 0 will set to 0 (fasle)
   cv::Mat_<uchar> green_image = rgbChannel[1] - thresh/2;

   green_image.copyTo(green_image, mask); // only when mask_i is non-zero
   cv::imshow("6.Green image use non-zero mask", green_image);
   ```
