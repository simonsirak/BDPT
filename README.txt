This README describes the contents of this hand-in, 
where you can find different things, and how you compile
the program.

THE DOCUMENTATION
The project specification is a separate PDF.

The report contains information regarding the techniques 
used to implement the specification. It also contains 
information about future work that could be done on the 
implementation.

The blog (https://bidirectionalpathtracer.blogspot.com/) 
contains a more in-depth discussion of the theory and 
progress that was made throughout the project. The final
post on the blog also contains a retrospective where 
I discuss my experiences in this project and what I took
away from the experience.

THE CODE 
The code is compiled using the exact same compilation 
procedure as in Lab 2 of the rendering track, i.e by 
using CMake and Make.

The code reuses the skeleton used for Lab 2 of the 
rendering track, PURELY for the purpose of using SDL
and the convenient setup. The triangle intersection 
is also reused from Lab 2. The rest of the code was 
written by me, with inspiration from the online 
book PBRT and Eric Veach's PhD thesis.

If you would like to generate another scene to render, 
you can add shapes in the TestModel.h file. Currently,
there is support for Triangles and Spheres.

The number of samples to take can be modified in the
skeleton.cpp file. So can the screen resolution.
