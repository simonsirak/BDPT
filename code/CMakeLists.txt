# Top-Level CMakeList.txt

cmake_minimum_required (VERSION 2.6)
project ( BDPT )

SET(GCC_COVERAGE_COMPILE_FLAGS "-std=c++11 -O3")
add_definitions(${GCC_COVERAGE_COMPILE_FLAGS})

add_executable( BDPT skeleton.cpp)

find_package (SDL)

if ( NOT SDL_FOUND )
   message ( FATAL_ERROR "SDL not found!" )
endif ( NOT SDL_FOUND )

if(SDL_FOUND)
	include_directories(
		${SDL_INCLUDE_DIR}
		${PROJECT_SOURCE_DIR}/glm
	)
	#link_libraries(${SDL_LIBRARY})
	target_link_libraries(BDPT ${SDL_LIBRARY})
endif(SDL_FOUND)

