#include <iostream>
#include <random>
#include <cassert>
#include <glm/glm.hpp>
#include <SDL.h>
#include <cmath>
#include "SDLauxiliary.h"
#include "TestModel.h"

#include <cstdlib>     /* srand, rand */
#include <ctime>       /* time */


using namespace std;
using glm::vec3;
using glm::mat3;

#define PI 3.141592653589793238462643383279502884

std::random_device rd;  //Will be used to obtain a seed for the random number engine
std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()

// STRUCTS 
struct Intersection{
	vec3 position;
	vec3 normal;
	float t;
	int triangleIndex;
};

struct BDPTpath{
	vec3 startP;
	vec3 endP;
	vector<Intersection> intersections;
};

// ----------------------------------------------------------------------------
// GLOBAL VARIABLES

/* Screen variables */
const int SCREEN_WIDTH = 50;
const int SCREEN_HEIGHT = 50;
SDL_Surface* screen;

/* Time */
int t;

/* Camera state */ 
float focalLength = SCREEN_HEIGHT;
float yaw = 0;
float pitch = 0;

/* Setters for the pitch and yaw given mouse coordinates relative to the center of screen */
#define PITCH(x, dt) (pitch += (SCREEN_WIDTH / 2.0f - x) * float(PI) * 0.001f * dt / (SCREEN_WIDTH))
#define YAW(y, dt) (yaw += (y - SCREEN_HEIGHT / 2.0f) * float(PI) * 0.001f * dt / (SCREEN_HEIGHT))

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

int numSamples = 500;

/* Light source */
vec3 lightPos( 0, -0.5, -0.7 );
vec3 lightColor = 800.f * vec3( 1, 1, 1 );
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

void FindPath(vec3 start, vec3 direction, int maxDepth, vector<Intersection>& intersections);
void FindPathHelper(vec3 point, vec3 normalToPoint, vec3 directionToPoint, int currentDepth, int maxDepth, vector<Intersection>& intersections);

vec3 calcRadianceToPoint(const BDPTpath& path, unsigned int i);

vec3 TracePath(vec3 start, vec3 dir, int depth);

int main( int argc, char* argv[] )
{
    srand((unsigned int)time(NULL));
	// load model
	LoadTestModel(triangles);

	screen = InitializeSDL( SCREEN_WIDTH, SCREEN_HEIGHT );
	t = SDL_GetTicks();	// Set start value for timer.

	Update();
	Draw();

	//otherwise visual studio closes window
	while(1){}

	// Compute frame time:
	int t2 = SDL_GetTicks();
	float dt = float(t2-t);
	t = t2;
	cout << "Render time: " << dt << " ms." << endl;

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

			dir = dir;
			dir = R * dir; // direction is rotated by camera rotation matrix
            dir = glm::normalize(dir); // normalize direction, crucial!!

			vec3 color( 0, 0, 0 );

			for (int i = 0; i < numSamples; ++i) {
				color += TracePath(cameraPos, dir, 0);
			}

			color /= numSamples;  // Average samples.

			/*vec3 lightDir = glm::normalize(R*vec3(0.5, 0.5, 0.5));
			for(int i = 0 ; i < numSamples ; ++i){
				BDPTpath path;
				path.startP = cameraPos;
				path.endP = lightPos;
				vector<Intersection> cameraIntersections;
				FindPath(path.startP, dir, 4, cameraIntersections);
				vector<Intersection> lightIntersections;
				FindPath(path.endP, lightDir, 4, lightIntersections);
				for(unsigned int i = 0 ; i < cameraIntersections.size() ; ++i){
					path.intersections.push_back(cameraIntersections[i]);
					cout << i << endl;
				}
				for(unsigned int i = lightIntersections.size() - 1 ; i >= 0 ; --i){
					path.intersections.push_back(lightIntersections[i]);
				}
				color += calcRadianceToPoint(path, 0);
			}
			color /= numSamples;*/

			PutPixelSDL( screen, x, y,  color);
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
	
    for(unsigned int i = 0; i < triangles.size(); ++i){
		const Obj* triangle = triangles[i];
		double t = triangle->intersect(r);
        if(t > 0.001f && t < closestIntersection.t){ // 0.001 is small epsilon to prevent self intersection
            closestIntersection.t = float(t);
            closestIntersection.triangleIndex = i;
        }
	}

	if(closestIntersection.t == std::numeric_limits<float>::max()){
		return false;
	} else {
        closestIntersection.position = r.o + r.d * closestIntersection.t;
		return true;
	}
}

vec3 TracePath(vec3 start, vec3 dir, int depth) {

	if (depth >= 4000) {
		return vec3(0, 0, 0) ;  // Bounced enough times
	}

	Intersection i;
	if (!ClosestIntersection(start, dir, triangles, i)) {
		return vec3(0, 0, 0);  // Nothing was hit.
	}

	// Pick a random direction from here and keep going.

	vec3 newstart = i.position;

	// This is NOT a cosine-weighted distribution!

	// RNG vector in hemisphere
	std::uniform_real_distribution<float> dis(0, 1.0);

	float theta0 = 2*float(PI)*dis(rd);
	float theta1 = acos(1 - 2*dis(rd));

	vec3 newdir = vec3(sin(theta1)*sin(theta0), sin(theta1)*cos(theta0), cos(theta1)); // http://corysimon.github.io/articles/uniformdistn-on-sphere/ THIS WAS A BIG ISSUE, WE FOLLOWED THE STACK OVERFLOW PIECES OF SHIEET. This fixed the "cross light on the ceiling"
	vec3 surfaceToLight = glm::normalize(start - newstart);

	newdir = glm::normalize(newdir);
	// enforce direction to be in correct hemisphere, aka project normal onto random vector and normalize
	vec3 normal = glm::dot(surfaceToLight, triangles[i.triangleIndex]->normal(newstart)) * triangles[i.triangleIndex]->normal(newstart) / glm::dot(triangles[i.triangleIndex]->normal(newstart), triangles[i.triangleIndex]->normal(newstart));
	normal = glm::normalize(normal); // HAS to be normalized after projection onto another vector.
	newdir = glm::dot(newdir, normal) * newdir / glm::dot(newdir, newdir);
	newdir = glm::normalize(newdir);


	// Compute the BRDF for this ray (assuming Lambertian reflection)

	float cos_theta = glm::dot(newdir, normal); // note that the cosine factor is part of the Rendering Equation integrand/differential, NOT the BRDF.
	vec3 BRDF = triangles[i.triangleIndex]->color / float(PI); // just the reflectance of the material / PI for Lambertian Reflection: http://www.oceanopticsbook.info/view/surfaces/lambertian_brdfs

	// Probability distribution function for lambertian reflection BRDF
	// const vec3 p = glm::abs(glm::dot(normal, -dir)) * BRDF; // dot between normal and origin ray (ONLY FOR EYE PATH) (THIS DOES NOT WORK WITH LAMBERTIAN REFLECTION SINCE THE BRDF EVALUATIONS WILL CANCEL EACH OTHER OUT AND YIELD A GRAY SCALE IMAGE)
	const float p = 1 /(2.f*float(PI)); // dot between normal and origin ray (ONLY FOR EYE PATH) (THIS ONE IS USED ONLY FOR LAMBERTIAN REFLECTION SINCE OTHERWISE THE BRDF CANCEL EACH OTHER OUT)

	// Recursively trace reflected light sources.

	vec3 incoming = TracePath(newstart, newdir, depth + 1);


	// Apply the Rendering Equation here. Let ceiling triangle emit light
	vec3 emittance = vec3(triangles[i.triangleIndex]->emission, triangles[i.triangleIndex]->emission, triangles[i.triangleIndex]->emission);

	// light distance thingy is NOT needed because we only regard the light ocming from a point on an emitting surface, whereas the labs calculated some sort of 
	// projected area onto the point light to deduce how much of the point light shines on the vertex. Further away, there are "less light" in a particular direction.

	//vec3 light = emittance/float(4.0f*PI*glm::length(start - newstart)*glm::length(start - newstart)); // calculate irradiance of light
	// emittance = emittance * (glm::dot(surfaceToLight, normal) > 0.0f ? glm::dot(surfaceToLight, normal) : 0.f);

	// assert(vec3(2, 3, 10) / vec3(2, 1, 5) == vec3(1, 3, 2));

	vec3 res =  (BRDF / p * incoming * cos_theta);
	// if(res.x > 0)
	// 	cout  << res.x << " " << res.y << " " << res.z << endl;

	return emittance + res;

}

//assume startP and endP already assigned
void FindPath(vec3 start, vec3 direction, int maxDepth, vector<Intersection>& intersections){
	//get first intersection, assume that first intersection is always possible
	Intersection i;
	ClosestIntersection(start, direction, triangles, i);
	//add next point
	intersections.push_back(i);
	//helper function to calculate rest of the points
	FindPathHelper(i.position, i.normal, direction, 1, maxDepth, intersections);
}

void FindPathHelper(vec3 point, vec3 normalToPoint, vec3 directionToPoint, int currentDepth, int maxDepth, vector<Intersection>& intersections){
	if(currentDepth <= maxDepth){
		Intersection i;
		std::uniform_real_distribution<float> dis(0, 1.0);
		float theta0 = 2*float(PI)*dis(rd);
		float theta1 = acos(1 - 2*dis(rd));
		vec3 newDirection = glm::normalize(vec3(sin(theta1)*sin(theta0), sin(theta1)*cos(theta0), cos(theta1)));
		vec3 normalDirection = glm::normalize(-directionToPoint);
		vec3 normal = glm::dot(normalDirection, normalToPoint) * normalToPoint / glm::dot(normalToPoint, normalToPoint);
		normal = glm::normalize(normal);
		newDirection = glm::dot(newDirection, normal) * newDirection / glm::dot(newDirection, newDirection);
		newDirection = glm::normalize(newDirection);
		if(ClosestIntersection(point, newDirection, triangles, i)){
			intersections.push_back(i);
			FindPathHelper(i.position, i.normal, newDirection, currentDepth+1, maxDepth, intersections);
		}else{
			//FindPathHelper(point, normalToPoint, directionToPoint, currentDepth, maxDepth, intersections);
		}
	}
}

vec3 calcRadianceToPoint(const BDPTpath& path, unsigned int i) {
	if(i >= path.intersections.size()){
		return vec3(0, 0, 0);
	}else{
		vec3 p0;
		if(i == 0){
			p0 = path.startP;
		}else{
			p0 = path.intersections[i-1].position;
		}
		vec3 p1 = path.intersections[i].position;
		vec3 p2;
		if(i == path.intersections.size() - 1){
			p2 = path.endP;
		}else{
			p2 = path.intersections[i+1].position;
		}
		vec3 n1 = path.intersections[i].normal;
		vec3 oldDirection = glm::normalize(p0 - p1);
		vec3 newDirection = glm::normalize(p2 - p1);
		vec3 normal = glm::normalize(glm::dot(oldDirection, n1) * n1 / glm::dot(n1, n1));
		float cos_theta = glm::dot(newDirection, normal);
		vec3 BRDF = triangles[path.intersections[i].triangleIndex]->color / float(PI);
		const float p = 1 /(2.f*float(PI));
		vec3 incoming = calcRadianceToPoint(path, i+1);
		vec3 emittance = vec3(triangles[path.intersections[i].triangleIndex]->emission, triangles[path.intersections[i].triangleIndex]->emission, triangles[path.intersections[i].triangleIndex]->emission);
		vec3 res = (BRDF / p * incoming * cos_theta);
		return emittance + res;
	}

}