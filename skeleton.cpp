#include <iostream>
#include <glm/glm.hpp>
#include <SDL.h>
#include <cmath>
#include <random>
#include "SDLauxiliary.h"
#include "TestModel.h"
#include "utility.h"

using namespace std;
using glm::vec3;
using glm::mat3;

int GenerateEyePath(int x, int y, vector<Vertex>& eyePath, int maxDepth);
int GenerateLightPath(vector<Vertex>& lightPath, int maxDepth);
int TracePath(Ray r, vector<Vertex>& subPath, int maxDepth, bool isRadiance, vec3 beta);
vec3 connect(vector<Vertex>& lightPath, vector<Vertex>& eyePath);
// ----------------------------------------------------------------------------
// GLOBAL VARIABLES

/* Screen variables */
const int SCREEN_WIDTH = 200;
const int SCREEN_HEIGHT = 200;
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
vector<Obj*> lights;

/* Light source */
vec3 lightPos( 0, -0.5, -0.7 );
vec3 lightColor = 14.f * vec3( 1, 1, 1 );
vec3 indirectLight = 0.5f*vec3( 1, 1, 1 );

/* Other BDPT stuff */
vec3 buffer[SCREEN_WIDTH][SCREEN_HEIGHT];
int samplesDone[SCREEN_WIDTH][SCREEN_HEIGHT];
int numSamples = 50;
int maxDepth = 10;
int curX, curY;

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
    srand(NULL);
	// load model
	LoadTestModel(triangles, lights);

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

void Draw()
{
	for(int i = 0 ; i < numSamples; ++i){
        if( SDL_MUSTLOCK(screen) )
		    SDL_LockSurface(screen);

		cout << "Sample " << (i+1) << "/" << numSamples << endl; 

        for( int y=0; y<SCREEN_HEIGHT; ++y ){
            for( int x=0; x<SCREEN_WIDTH; ++x ){
                if(!NoQuitMessageSDL())
                    return;

                // if(samplesDone[x][y] >= numSamples)
                //     continue;

                // curX = x;
                // curY = y;

				vector<Vertex> lightPath;
				vector<Vertex> eyePath;

				// Trace eye path
				GenerateEyePath(x, y, eyePath, maxDepth);

				// Trace light path
				// for now, direction of ray = direction from center to point. Should change to random hemisphere direction later.
				GenerateLightPath(lightPath, maxDepth);

                vec3 old = buffer[x][y];

				// The result is divided by numSamples, aka 
				// sum of the filter function at all evaluated points on the pixel 
				// film, which were all from 1 point on the pixel, which had an importance of 1 

                // This is the sequential form of division by numSamples.
				buffer[x][y] = (old * float(i) + connect(lightPath, eyePath))/float(i+1);
                //buffer[x][y] += connect(lightPath, eyePath);

                PutPixelSDL( screen, x, y,  buffer[x][y]);

                // Regular ray tracing
                // vec3 dir(
				// 		x-SCREEN_WIDTH/2.0f, 
				// 		y-SCREEN_HEIGHT/2.0f, 
				// 		focalLength
				// 	); 

                // dir = dir;
                // dir = R * dir; // direction is rotated by camera rotation matrix

                // vec3 color( 0, 0, 0 );
                // Intersection inter;
                // inter.triangleIndex = -1;
                // if(ClosestIntersection(cameraPos, dir, triangles, inter)){
                //     PutPixelSDL( screen, x, y,  DirectLight(inter) + triangles[inter.triangleIndex]->color * indirectLight);
                // } else {
                //     PutPixelSDL( screen, x, y,  color);
                // }
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

	int originalTriangle = closestIntersection.triangleIndex;
	
    for(int i = 0; i < triangles.size(); ++i){
		if(i == originalTriangle)
			continue;

		const Obj* triangle = triangles[i];
		double t = triangle->intersect(r);
        if(t > 0 && t < closestIntersection.t){ // 0.0001f is small epsilon to prevent self intersection
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

int GenerateEyePath(int x, int y, vector<Vertex>& eyePath, int maxDepth){

	if(maxDepth == 0)
		return 0;

	vec3 normal(
		0, 
		0, 
		focalLength
	);

	vec3 dir(
		x-SCREEN_WIDTH/2.0f, 
		y-SCREEN_HEIGHT/2.0f, 
		focalLength
	); 

	normal = glm::normalize(R * normal);
	dir = glm::normalize(R * dir); // direction is rotated by camera rotation matrix

	// KEEP AN EYE OUT FOR THIS BETA, should be correct tho
	vec3 We = vec3(1,1,1), beta = vec3(1,1,1); // for simplicity, one pixel is perfectly covered by one ray, so importance is 1.
	eyePath.push_back({1, 0, We, -1, normal, cameraPos, dir}); // the probability up to this point is 1

	return TracePath(Ray(cameraPos, dir), eyePath, maxDepth - 1, true, beta) + 1;
}

int GenerateLightPath(vector<Vertex>& lightPath, int maxDepth){

	if(maxDepth == 0)
		return 0;

    int index = rand() % lights.size();
    int triangleIndex = -1;

	Sphere * light = dynamic_cast<Sphere*>(lights[index]);   

    for(int i = 0; i < triangles.size(); ++i){
        if(light == triangles[i]){
            triangleIndex = i;
            break;
        }
    } 

	vec3 offset = uniformSphereSample(light->r);
    vec3 dir = cosWeightedUniformHemisphereSample(offset);

	float lightChoiceProb = 1 / float(lights.size());
	float lightPosProb = uniformSphereSamplePDF(light->r); // 1 / Area
	float lightDirProb = uniformHemisphereSamplePDF(1); //cosWeightedUniformHemisphereSamplePDF(dir, offset); // hemisphere uniform, not cosine weighted
	float pointProb = lightChoiceProb * lightPosProb * lightDirProb;

	vec3 Le = vec3(light->emission, light->emission, light->emission);
	lightPath.push_back({lightChoiceProb * lightPosProb, 0, Le, triangleIndex, offset, light->c + offset, dir}); 

	// the result is built up from this, which describes the "light" 
	// that is carried over to the "next" ray in the path.
	vec3 beta = Le * glm::dot(glm::normalize(offset), dir) / pointProb;

	//// note, not cosine weighted, we just calculate the cosine term of the LTE.
	// vec3 Le = vec3(light->emission, light->emission, light->emission) * glm::dot(offset, dir) / pointProb;
	return TracePath(Ray(light->c + offset, dir), lightPath, maxDepth - 1, false, beta) + 1;
}

/*
	wi = the direction from next point to the intersection/current point.
	wo = the direction from current/intersection point to previous point.
	The reason for these to be seemingly "flipped" is because we are evaluating
	a path backwards from some origin. So the "backwards/forward" is in relation
	to the direction traveling in.
	If we start in the eye, we want to transport radiance to the eye. 
	So wi is correct, since that is the general direction to travel towards 
	the eye. 
	If we start in the light, we want to transport importance to the light. 
	So wi is correct, since that is the general direction to travel towards 
	the eye.
*/
int TracePath(Ray r, vector<Vertex>& subPath, int maxDepth, bool isRadiance, vec3 beta) {

	if (maxDepth == 0) {
		return 0;  // Bounced enough times
	}

	int bounces = 0;

	float pdfFwd = isRadiance ? uniformHemisphereSamplePDF(1) : uniformHemisphereSamplePDF(1) /*cosWeightedUniformHemisphereSamplePDF(subPath[0].dir, subPath[0].normal)*/, pdfRev = 0;
	while(true){
		Intersection point;
		point.triangleIndex = subPath[bounces].surfaceIndex; // previous vertex

		// Nothing was hit.
		if (!ClosestIntersection(r.o, r.d, triangles, point)) {
			break; 
		}

		// Vertex &vert = subPath[bounces+1], &prev = subPath[bounces]; // other code had offset vector by 1 forward, that's why its not bounces and bounces-1

		if(++bounces >= maxDepth)
			break;

		Vertex vertex;
		Vertex *prev = &subPath[bounces-1];

		/* Process intersection data */
		Ray r1;

        r1.o = point.position;

        vec3 intersectionToOrigin = glm::normalize(r.o - r1.o);
        vec3 gnormal = glm::normalize(triangles[point.triangleIndex]->normal(r1.o));
		vec3 snormal = glm::normalize(projectAOntoB(intersectionToOrigin, gnormal)); // HAS to be normalized after projection onto another vector.

        r1.d = uniformHemisphereSample(snormal, 1);

		/* Construct vertex */

		vertex.position = point.position;
		vertex.normal = snormal;
		vertex.surfaceIndex = point.triangleIndex;

		vec3 w = vertex.position - prev->position;
		float invDist2 = 1 / glm::dot(w, w);

		// convert to area based density by applying solidangle-to-area conversion
		vertex.pdfFwd = pdfFwd * invDist2 * glm::abs(glm::dot(gnormal, glm::normalize(w)));

		// give old beta as the incoming at this intersection
		vertex.c = beta;

		// insert vertex
		subPath.push_back(vertex);
        prev = &subPath[bounces-1]; // NEED TO DO THIS IN CASE RESIZING OF VECTOR OCCURS

		// don't include anything after a light source
		if(triangles[point.triangleIndex]->emission > 0){
			break;
		}

		// reverse is simulated as if the ray came 
		pdfFwd = uniformHemisphereSamplePDF(1);
		pdfRev = uniformHemisphereSamplePDF(1);
		prev->pdfRev = pdfRev * invDist2 * glm::abs(glm::dot(prev->normal, glm::normalize(-w)));

		// append the contribution to the beta from the current intersection point 
		// onto the future intersection points

        // DOUBLE CHECK THAT CORRECT WO AND WI ARE USED
        // (TRY DRAWING A PICTURE)
		vec3 brdf = BRDF(vertex, r1.d, r.d, triangles, isRadiance); 
		vec3 wi = -glm::normalize(w);
		// one of the many nested surface integral samples
		beta *= (brdf * glm::abs(glm::dot(r1.d, snormal) / pdfFwd)); // THIS was the reason i got white lines, it's because i used the wrong direction (the outgoing as opposed to the incoming from the next point)
		
		// It should be allowed to be > 10, this was just for debugging
		// if(glm::length(beta) > 10)
		// 	cout << "Depth " << bounces+1 << ", beta: " << beta.x << " " << beta.y << " " << beta.z << endl;

		// CHANGE THE RAY OBVIOUSLY
		r = r1;
    }
    
    return bounces;
}

/* connects all possible paths */
vec3 connect(vector<Vertex>& lightPath, vector<Vertex>& eyePath){
	//cout << "Doin shit" << endl;
	int s = lightPath.size();
	int t = eyePath.size();

	float basicScale = 1; //1 / (s + t + 2.);
	vec3 color;

	// // // s == 0
	vec3 L;
	for(int i = 2; i <= t; ++i){
        Vertex &last = eyePath[i-1];
		if(triangles[last.surfaceIndex]->emission > 0){
			// get light emitted from last light to second last element of eyePAth
			vec3 lightToPoint = glm::normalize(eyePath[i-2].position - last.position); // maybe i should cosine weight the emittance using this?
			vec3 Le = vec3(triangles[last.surfaceIndex]->emission, triangles[last.surfaceIndex]->emission, triangles[last.surfaceIndex]->emission);
			L = Le * last.c; // MAYBE LOOK HERE, DON'T USE LAST?
			color += L * MIS(lightPath, eyePath, 0, i);
		}
		
	}

	// t == 1: Skipped (pretty sure its pointless since we only have 1 direction and 1 point on the pixel to go from)

	// s == 1: Also kinda skipped, but is done in the other for loops by not recalculating the light sample but instead trying to connect it.

	// s, t > 1
	for(int i = 1; i <= s; ++i){
		for(int j = 2; j <= t; ++j){ // eye is not really a surface so im not counting it?
			// if(i == 1 && j == 1) // already done by DirectLight, so don't add
			// 	continue;

			//cout << i << " " << j << endl;
            Intersection otherObj;
			otherObj.triangleIndex = lightPath[i-1].surfaceIndex; // previous vertex
			if(!ClosestIntersection(lightPath[i-1].position, (eyePath[j-1].position - lightPath[i-1].position), triangles, otherObj)){
				continue;
			} else {
				if(otherObj.triangleIndex != eyePath[j-1].surfaceIndex && otherObj.t > 0.01f){
					continue;
				} else {
					// // Assume lambertian surface
					// if(eyePath[j-1].c.x < -10000 || eyePath[j-1].c.x > 10000 || eyePath[j-1].c.y < -10000 || eyePath[j-1].c.y > 10000 || eyePath[j-1].c.z < -10000 || eyePath[j-1].c.z > 10000){
					// 	cout << eyePath[j-1].c.x << " " << eyePath[j-1].c.y << " " << eyePath[j-1].c.z << endl;
					// 	continue;
					// }
					// if(lightPath[i-1].c.x < -10000 || lightPath[i-1].c.x > 10000 || lightPath[i-1].c.y < -10000 || lightPath[i-1].c.y > 10000 || lightPath[i-1].c.z < -10000 || lightPath[i-1].c.z > 10000){
					// 	cout << lightPath[i-1].c.x << " " << lightPath[i-1].c.y << " " << lightPath[i-1].c.z << endl;
					// 	continue;
					// }
					color += lightPath[i-1].c * eyePath[j-1].c * triangles[lightPath[i-1].surfaceIndex]->color / float(PI)
						*  G(lightPath[i-1].normal, eyePath[j-1].normal, eyePath[j-1].position - lightPath[i-1].position)
						*  triangles[eyePath[j-1].surfaceIndex]->color / float(PI)
						*  MIS(lightPath, eyePath, i, j);
					
					/* This samamamich line is supposed to have that last commented out factor */ 
					//color += /*triangles[lightPath[i].surfaceIndex]->color / float(PI) * G(lightPath[i].normal, eyePath[j].normal, eyePath[j].position - lightPath[i].position);*/ triangles[eyePath[j].surfaceIndex]->color / float(PI);
				}
			} 
			
		}
	}

	// color is now the result of the multi-sample estimation, aka 
	// is a multi-sample estimator. This is because we have taken 
	// a sample of a number of strategies p(s,t), weighted by their 
	// probability of occuring.
	//
	// I should now remove the samplesDone buffer since it is unnecessary.

	return color; 
}

/*
TODO: GÖR OM CONNECT SÅ ATT DEN GÖR;
- FOR EACH CONNECTION OF TWO SUBPATHS
	- RUN ALL THE CASES 0, 1, WHATEVER
	- COUNT THAT AS ONE SAMPLE AND DIVIDE BY NUMSAMPLES

Right now, I am only calculating one strategy for each sample, i.e the 
strategy you get from letting the subpaths be exactly what they actually are.

When I fix this, I am 100 % done with the code.
*/