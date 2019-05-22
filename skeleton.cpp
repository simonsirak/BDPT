#include <iostream>
#include <glm/glm.hpp>
#include <SDL.h>
#include <cmath>
#include "SDLauxiliary.h"
#include "TestModel.h"
#include "utility.h"

using namespace std;
using glm::vec3;
using glm::mat3;

// STRUCTS 
struct Intersection{
	vec3 position;
	vec3 normal;
	float t;
	int triangleIndex;
};

// ----------------------------------------------------------------------------
// GLOBAL VARIABLES

/* Screen variables */
const int SCREEN_WIDTH = 300;
const int SCREEN_HEIGHT = 300;
SDL_Surface* screen;

/* Time */
int t;

/* Camera state */ 
float focalLength = SCREEN_HEIGHT;
float yaw = 0;
float pitch = 0;

/* Setters for the pitch and yaw given mouse coordinates relative to the center of screen */
#define PITCH(x, dt) (pitch += (SCREEN_WIDTH / 2.0f - x) * PI * 0.001f * dt / (SCREEN_WIDTH))
#define YAW(y, dt) (yaw += (y - SCREEN_HEIGHT / 2.0f) * PI * 0.001f * dt / (SCREEN_HEIGHT))

vec3 cameraPos( 0, 0, -3 );

mat3 R; // Y * P
mat3 Y; // Yaw rotation matrix (around y axis)
mat3 P; // Pitch rotation matrix (around x axis)

/* Directions extracted from given mat3 */

#define FORWARD(R) (R[2])
#define RIGHT(R) (R[0])
#define UP(R) (R[1])

/* Model */
vector<Obj*> triangles;

/* Light source */
vec3 lightPos( 0, -0.5, -0.7 );
vec3 lightColor = 14.f * vec3( 1, 1, 1 );
vec3 indirectLight = 0.5f*vec3( 1, 1, 1 );

// ----------------------------------------------------------------------------
// FUNCTIONS

void Update();
void Draw();
bool ClosestIntersection(
	vec3 start, 
	vec3 dir,
	const vector<Obj*>& triangles, 
	Intersection& closestIntersection 
);

vec3 DirectLight( const Intersection& i );

int main( int argc, char* argv[] )
{
	// load model
	LoadTestModel(triangles);

	screen = InitializeSDL( SCREEN_WIDTH, SCREEN_HEIGHT );
	t = SDL_GetTicks();	// Set start value for timer.

	while( NoQuitMessageSDL() )
	{
		Update();
		Draw();
	}

	SDL_SaveBMP( screen, "screenshot.bmp" );
	return 0;
}

void Update()
{
	// Compute frame time:
	int t2 = SDL_GetTicks();
	float dt = float(t2-t);
	t = t2;
	cout << "Render time: " << dt << " ms." << endl;

	int x, y;
	if(SDL_GetMouseState(&x, &y) & SDL_BUTTON(SDL_BUTTON_LEFT)) {
		YAW(x, dt);
		PITCH(y, dt);
	}

	Uint8* keystate = SDL_GetKeyState( 0 );

	if( keystate[SDLK_UP] )
		lightPos += FORWARD(R) * 0.007f * dt;

	if( keystate[SDLK_DOWN] )
		lightPos -= FORWARD(R) * 0.007f * dt;

	if( keystate[SDLK_RIGHT] )
		lightPos += RIGHT(R) * 0.007f * dt;

	if( keystate[SDLK_LEFT] )
		lightPos -= RIGHT(R) * 0.007f * dt;

	if(keystate[SDLK_w]){
		cameraPos += FORWARD(R) * 0.007f * dt; // camera Z
	} 
	
	if(keystate[SDLK_s]){
		cameraPos -= FORWARD(R) * 0.007f * dt; // camera Z
	} 
	
	if(keystate[SDLK_a]){
		cameraPos -= RIGHT(R) * 0.007f * dt; // camera X
	} 
	
	if(keystate[SDLK_d]){
		cameraPos += RIGHT(R) * 0.007f * dt; // camera X
	} 
	
	if(keystate[SDLK_q]){
		cameraPos -= UP(R) * 0.007f * dt; // camera Y
	} 

	if(keystate[SDLK_e]){
		cameraPos += UP(R) * 0.007f * dt; // camera Y
	} 

	Y[0][0] = cos(yaw);
	Y[0][2] = -sin(yaw);
	Y[2][0] = sin(yaw);
	Y[2][2] = cos(yaw);

	P[1][1] = cos(pitch);
	P[1][2] = sin(pitch);
	P[2][1] = -sin(pitch);
	P[2][2] = cos(pitch);

	R = Y * P;
}

// TODO: Perhaps make normal vectors always face camera so light cannot be seen on both sides of a face
// i.e no refraction
void Draw()
{
	if( SDL_MUSTLOCK(screen) )
		SDL_LockSurface(screen);

	for( int y=0; y<SCREEN_HEIGHT; ++y )
	{
		for( int x=0; x<SCREEN_WIDTH; ++x )
		{
			vec3 dir(
						x-SCREEN_WIDTH/2.0f, 
						y-SCREEN_HEIGHT/2.0f, 
						focalLength
					); 

			// WRONG. This should NEVER be DONE.
			// Reason: dir is supposed to store the 
			// direction FROM the cameraPosition. 
			// From the cameraPosition, the direction
			// is then obtained by a constant delta 
			// that is "focalLength" units forward, 
			// and then x and y decided by the pixels.
			// The image plane is ALWAYS offset by
			// "focalLength" units forward from the 
			// cameraPosition. The objects themselves 
			// are not rotated. It is the camera that 
			// rotates, and translation of the camera
			// then depends on this rotational information.
			// The rotational information is also applied
			// to each direction calculated, so that the 
			// direction is in the camera frame.

			// I found this by thinking about what vector 
			// we actually want. One can practically see this 
			// issue by positioning yourself to the right 
			// of the Cornell box, centering it in the image 
			// plane. Then, backing away from the box shows 
			// how dir should never be modified from the 
			// above calculation, since it ALREADY IS relative
			// to the camera origin.


			//if(toggle)
			//	dir -= cameraPos; // detta borde väl alltid göras?? NEJ, se kommentar ovan

			dir = dir;
			dir = R * dir; // direction is rotated by camera rotation matrix

			vec3 color( 0, 0, 0 );
			Intersection inter;
			inter.triangleIndex = -1;
			if(ClosestIntersection(cameraPos, dir, triangles, inter)){
				PutPixelSDL( screen, x, y,  DirectLight(inter) + triangles[inter.triangleIndex]->color * indirectLight);
			} else {
				PutPixelSDL( screen, x, y,  color);
			}
		}
	}

	if( SDL_MUSTLOCK(screen) )
		SDL_UnlockSurface(screen);

	SDL_UpdateRect( screen, 0, 0, 0, 0 );
}

bool ClosestIntersection(
	vec3 start, 
	vec3 dir,
	const vector<Obj*>& triangles, 
	Intersection& closestIntersection 
){
	closestIntersection.t = std::numeric_limits<float>::max();
	dir = glm::normalize(dir); // does not matter what you do here, the t aka x.x will adjust the length. No need to normalize
    Ray r(start, dir);

	int originalTriangle = closestIntersection.triangleIndex;
	
    for(int i = 0; i < triangles.size(); ++i){
		if(i == originalTriangle)
			continue;

		const Obj* triangle = triangles[i];
		double t = triangle->intersect(r);
        if(t > 0 && t < closestIntersection.t){ // 0.001 is small epsilon to prevent self intersection
            closestIntersection.t = t;
            closestIntersection.triangleIndex = i;
        }
	}

	if(closestIntersection.t == std::numeric_limits<float>::max()){
		return false;
	} else {
        closestIntersection.position = r.o + r.d * closestIntersection.t;
        closestIntersection.normal = triangles[closestIntersection.triangleIndex]->normal(closestIntersection.position);
		return true;
	}
}

vec3 DirectLight( const Intersection& i ){
	
	// get radius from sphere defined by light position 
	// and the point of intersection.
	vec3 radius = (lightPos - i.position);
	float r2 = glm::length(radius)*glm::length(radius); // r^2
	radius = glm::normalize(radius); // normalize for future calculations

	vec3 light = lightColor/float(4.0f*PI*r2); // calculate irradiance of light

	/* 
		calculate normal in direction based on source of ray
		for the "first ray", it is the camera position that 
		is the source. However, for multiple bounces you would 
		need to embed the ray source into the Intersection 
		struct so that the correct normal can be calculated
		even when the camera is not the source.

		This makes the model only take the viewable surface into
		consideration. Note however that for any given ray source, only 
		one side of the surface is ever viewable , so this all works 
		out in the end. Multiple bounces would have different sources,
		so it is still possible that the other side of a surface can 
		receive light rays. Just not from light hitting the other side 
		of the surface.
	*/

	vec3 sourceToLight = cameraPos - i.position;
	vec3 normal = projectAOntoB(sourceToLight, i.normal);
	normal = glm::normalize(normal);

	/*
		Direction needs to either always or never be normalised. 
		Because it is not normalised in the draw function, I 
		will not normalize here.

		Also, I use a shadow bias (tiny offset in normal direction)
		to avoid "shadow acne" which is caused by self-intersection.
	*/

	Intersection blocker;
	blocker.triangleIndex = i.triangleIndex;
	if(ClosestIntersection(i.position, (lightPos - i.position), triangles, blocker) && blocker.t < glm::length(lightPos-i.position)){
		return vec3(0, 0, 0);
	} else {
		return triangles[i.triangleIndex]->color * light * (glm::dot(radius, normal) > 0.0f ? glm::dot(radius, normal) : 0.0f);
	}
}