This README describes the contents of this repository, 
where you can find different things, and how you compile
the program.

#####################
# THE DOCUMENTATION #
#####################

PROJECT SPECIFICATION
The project specification is a separate PDF.

PROJECT REPORT
The report contains information regarding the techniques 
used to implement the specification. It also contains 
information about future work that could be done on the 
implementation.

BLOG
The blog (https://bidirectionalpathtracer.blogspot.com/) 
contains a more in-depth discussion of the theory and 
progress that was made throughout the project. The final
post on the blog also contains a retrospective where 
I reflect upon my experiences in this project and what 
I took away from the experience.

############
# THE CODE #
############ 

COMPILATION
The code is compiled using the exact same compilation 
procedure as in Lab 2 of the rendering track, i.e by 
using CMake and Make. This means that the correct SDL
version (the one used in Lab 2, SDL 1.2.15) must be installed 
in order to run this code. The executable provided in this 
zip-file is compiled for Ubuntu 18.04 LTS running on 
an x86 architecture. If you are having trouble compiling
for your system, it might be because the CMake file 
sets a preferred optimization level as well and forces
the code to obey the C++11 standard. Simply remove any 
of these flags if you think they may be stopping you 
from successful compilation.

CODE
The code reuses the skeleton from Lab 2 of the 
rendering track, specifically the SDL wrapper functions
and the provided Cornell Box (excluding lights 
and data structures). The triangle intersection 
is also reused from Lab 2 (but was written by me). 
The rest of the code was written by me, with 
inspiration from the online book PBRT and Eric 
Veach's PhD thesis.

The main algorithms for path generation and connection 
can be found in skeleton.cpp. Utility functions relating 
to the manipulation of path vertices, as well as the vertex 
structures themselves, can be found in utility.h. The 
data structures for different shapes (triangles and spheres)
can be found in TestModel.h.

BUILDING YOUR OWN SCENE
If you would like to generate another scene to render, 
you can add shapes in the TestModel.h file. Currently,
there is support for Triangles and Spheres. The easiest
thing to add is a sphere, which requires only a radius,
color, emission (which is 0 for non-light sources) and 
position. The type of the material should always be set 
to 1 since support for other materials has not been 
implemented yet.

You can also change the number of samples to take and the
image resolution. This can be modified in the skeleton.cpp 
file, near the beginning of the file.

Rendering the scene is as simple as running the executable.
