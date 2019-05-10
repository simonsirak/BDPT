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

struct Vertex {
	float p; // probability of reaching this particular vertex
	float c; // contribution from a path that crosses to the other sub-path from this vertex
};

// ----------------------------------------------------------------------------
// GLOBAL VARIABLES

/* Screen variables */
const int SCREEN_WIDTH = 200;
const int SCREEN_HEIGHT = 200;
SDL_Surface* screen;
vec3 buffer[SCREEN_WIDTH][SCREEN_HEIGHT];

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

//vec3 TracePath(vec3 start, vec3 dir, int depth);
void TracePath(Ray r, vector<Vertex>& subPath, int i, int t);

vec3 DirectLight( const Intersection& i );

int main( int argc, char* argv[] )
{
    srand(time(NULL));
	// load model
	LoadTestModel(triangles);

	screen = InitializeSDL( SCREEN_WIDTH, SCREEN_HEIGHT );
	t = SDL_GetTicks();	// Set start value for timer.

	Update();
	Draw();

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
	for(int i = 0 ; i < numSamples; ++i){
        if( SDL_MUSTLOCK(screen) )
		    SDL_LockSurface(screen);

        for( int y=0; y<SCREEN_HEIGHT; ++y ){
            for( int x=0; x<SCREEN_WIDTH; ++x ){
                if(!NoQuitMessageSDL())
                    return;

                vec3 dir(
                            x-SCREEN_WIDTH/2.0f, 
                            y-SCREEN_HEIGHT/2.0f, 
                            focalLength
                        ); 

                // dir = dir;
                // dir = R * dir; // direction is rotated by camera rotation matrix
                // dir = glm::normalize(dir); // normalize direction, crucial!!

				vector<Vertex> lightPath;
				vector<Vertex> eyePath;
				lightPath.push_back({1, 1}); // i = 0
				eyePath.push_back({1, 1}); // i = 0
				
				// i = 1

				// P1, C1 for Light
				Sphere * light = dynamic_cast<Sphere*>(triangles[triangles.size()-1]);
				float p1 = float(1/(light->r*light->r*4*PI*2*PI)); // probability is 1/(AreaOfLight*2PI)
				lightPath.push_back({p1, light->emission/p1});

				// P1, C1 for Eye
				p1 = 1; // for now, we assume area of a pixel is perfectly covered by one ray.
				// We calculation: http://rendering-memo.blogspot.com/2016/03/bidirectional-path-tracing-3-importance.html
				float cosTheta = glm::normalize(dir).z; // think about the triangle, we want the angle.
				float G = cosTheta*cosTheta / (dir.length() * dir.length());
				float We = float(1 / (1 * G)); // 1 / (AreaOfFilmAkaPixel * G), vi använder inte lens, vi kör 0-size aperture
				eyePath.push_back({p1, We/p1});

				// Trace eye path
				TracePath(Ray(cameraPos, dir), eyePath, 2, 4);

				// Trace light path

				std::uniform_real_distribution<float> dis(0, 1.0);

				float theta0 = 2*PI*dis(rd);
				float theta1 = acos(1 - 2*dis(rd));

				vec3 newdir = glm::normalize(vec3(sin(theta1)*sin(theta0), sin(theta1)*cos(theta0), cos(theta1)));

				// for now, direction of ray = direction from center to point. Should change to random hemisphere direction later.
				TracePath(Ray(light->c + light->r*newdir, newdir), lightPath, 2, 4);

                // buffer[x][y] += (TracePath(Ray(cameraPos, dir), 0) / float(numSamples));

                PutPixelSDL( screen, x, y,  buffer[x][y]);
            }
        }   

        if( SDL_MUSTLOCK(screen) )
            SDL_UnlockSurface(screen);

        SDL_UpdateRect( screen, 0, 0, 0, 0 );
    }
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
	
    for(int i = 0; i < triangles.size(); ++i){
		const Obj* triangle = triangles[i];
		double t = triangle->intersect(r);
        if(t > 0.001f && t < closestIntersection.t){ // 0.001 is small epsilon to prevent self intersection
            closestIntersection.t = t;
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
	vec3 normal = glm::dot(sourceToLight, triangles[i.triangleIndex]->normal(i.position)) * triangles[i.triangleIndex]->normal(i.position) / glm::dot(triangles[i.triangleIndex]->normal(i.position), triangles[i.triangleIndex]->normal(i.position));
	normal = glm::normalize(normal);

	/*
		Direction needs to either always or never be normalised. 
		Because it is not normalised in the draw function, I 
		will not normalize here.
		Also, I use a shadow bias (tiny offset in normal direction)
		to avoid "shadow acne" which is caused by self-intersection.
	*/

	Intersection blocker;
	if(ClosestIntersection(i.position + normal * 0.0001f, (lightPos - i.position), triangles, blocker) && glm::length(blocker.position - i.position) <= glm::length(lightPos-i.position)){
		return vec3(0, 0, 0);
	} else {
		return triangles[i.triangleIndex]->color * light * (glm::dot(radius, normal) > 0.0f ? glm::dot(radius, normal) : 0.0f);
	}
}

vec3 TracePath(vec3 start, vec3 dir, int depth) {

	if (depth >= 5) {
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

	float theta0 = 2*PI*dis(rd);
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
	const float p = 1 /(2.f*PI); // dot between normal and origin ray (ONLY FOR EYE PATH) (THIS ONE IS USED ONLY FOR LAMBERTIAN REFLECTION SINCE OTHERWISE THE BRDF CANCEL EACH OTHER OUT)

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

void TracePath(Ray r, vector<Vertex>& subPath, int i, int t) {

	if (depth >= t) {
		return;  // Bounced enough times
	}

	Intersection i;
	if (!ClosestIntersection(r.o, r.d, triangles, i)) {
		return;  // Nothing was hit.
	}

	/*
		r.d is incoming direction to our current position, i.e the intersection point.
		
		newdir is the outgoing direction.

		These can be used to calculate the cosThetaPrime and cosTheta for the probability
		and contribution of this point.
	*/

	// Pick a random direction from here and keep going.

	vec3 newstart = i.position;

	// RNG vector in hemisphere
	std::uniform_real_distribution<float> dis(0, 1.0);

	float theta0 = 2*PI*dis(rd);
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
	const float p = 1 /(2.f*PI); // dot between normal and origin ray (ONLY FOR EYE PATH) (THIS ONE IS USED ONLY FOR LAMBERTIAN REFLECTION SINCE OTHERWISE THE BRDF CANCEL EACH OTHER OUT)

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