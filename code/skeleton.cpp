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

// ----------------------------------------------------------------------------
// GLOBAL VARIABLES

/* Screen variables */
const int SCREEN_WIDTH = 400;
const int SCREEN_HEIGHT = 400;
SDL_Surface* screen;

/* Camera state (Camera looks into z direction) */ 
float focalLength = SCREEN_HEIGHT;
vec3 cameraPos( 0, 0, -3 );

/* Model */
/*
	Meshes to be rendered are added to
	triangles. Lights to be used are
	added to triangles and lights.
*/
vector<Obj*> triangles;
vector<Obj*> lights;

/* BDPT parameters */
vec3 buffer[SCREEN_WIDTH][SCREEN_HEIGHT];
int numSamples = 75;
int maxDepth = 10;

// ----------------------------------------------------------------------------
// FUNCTIONS

void Draw();
bool ClosestIntersection(
	vec3 start, 
	vec3 dir,
	const vector<Obj*>& triangles, 
	Intersection& closestIntersection 
);

int GenerateEyePath(int x, int y, vector<Vertex>& eyePath, int maxDepth);
int GenerateLightPath(vector<Vertex>& lightPath, int maxDepth);
int TracePath(Ray r, vector<Vertex>& subPath, int maxDepth, bool isCameraPath, vec3 beta);
vec3 connect(vector<Vertex>& lightPath, vector<Vertex>& eyePath);

int main( int argc, char* argv[] )
{
    srand(NULL);
    // load model
    LoadTestModel(triangles, lights);

    screen = InitializeSDL( SCREEN_WIDTH, SCREEN_HEIGHT );
    int t1 = SDL_GetTicks();	// Set start value for timer.

    Draw();

    // Compute frame time:
    int t2 = SDL_GetTicks();
    float dt = float(t2-t1);
    cout << "Render time: " << dt << " ms." << endl;

    SDL_SaveBMP( screen, "screenshot.bmp" );
    return 0;
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

                vector<Vertex> lightPath;
                vector<Vertex> eyePath;

                // Generate eye path
                GenerateEyePath(x, y, eyePath, maxDepth);

                // Generate light path
                GenerateLightPath(lightPath, maxDepth);

                vec3 old = buffer[x][y];

                // This is the sequential form of division by numSamples.
                // connect() calculates a multi-sample estimator from 
                // the two paths using multiple importance sampling 
                // with the balance heuristic.
                buffer[x][y] = (old * float(i) + connect(lightPath, eyePath))/float(i+1);

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

int GenerateEyePath(int x, int y, vector<Vertex>& eyePath, int maxDepth){

    if(maxDepth == 0)
        return 0;

    vec3 normal(0, 0, focalLength);
    vec3 dir(x-SCREEN_WIDTH/2.0f, y-SCREEN_HEIGHT/2.0f, focalLength); 

    normal = glm::normalize(normal);
    dir = glm::normalize(dir);

    /*
        Calculation of the initial sample contribution from 
        camera endpoint.

        Because my camera model is a pinhole camera and I only
        shoot rays through one point of each pixel, We = 1. I.e
        I model the camera so that one pixel is perfectly covered
        by one ray.
    */
    vec3 We = vec3(1,1,1);
    vec3 beta = We;
    eyePath.push_back({1, 0, We, -1, normal, cameraPos, dir}); // the probability up to this point is 1

    return TracePath(Ray(cameraPos, dir), eyePath, maxDepth - 1, true, beta) + 1;
}

int GenerateLightPath(vector<Vertex>& lightPath, int maxDepth){

    if(maxDepth == 0)
        return 0;

    // choose random light and find its index int triangles[],
    // the vector containing all shapes (yes I know it's a dumb
    // name since we have spheres now...)
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
    vec3 dir = uniformHemisphereSample(offset, 1);

    float lightChoiceProb = 1 / float(lights.size());
    float lightPosProb = uniformSphereSamplePDF(light->r);
    float lightDirProb = uniformHemisphereSamplePDF(1);
    float pointProb = lightChoiceProb * lightPosProb * lightDirProb;

    vec3 Le = vec3(light->emission, light->emission, light->emission);
    lightPath.push_back({lightChoiceProb * lightPosProb, 0, Le, triangleIndex, offset, light->c + offset, dir}); 

    /*			
        The light endpoint base case of the measurement 
        contribution function.

        Formally according to the measurement contribution 
        function, this calculation should be:

        = Le * G(current, next) / (lightChoiceProb * lightPosProb * lightDirProb * DirectionToAreaConversion(current, next))

        However certain factors in G and DirectionToAreaConversion
        require knowledge about the next point which we do not have.
        Luckily, these factors cancel out, and we only have to worry
        about the factor remaining in the calculation below.
    */

    vec3 beta = Le * glm::dot(glm::normalize(offset), dir) / pointProb;

    return TracePath(Ray(light->c + offset, dir), lightPath, maxDepth - 1, false, beta) + 1;
}

/*
    Generates a path from a certain starting 
    vertex (camera or light source) specified
    in the subPath[0]. The path generates at 
    most maxDepth additional vertices.

    The path starts off using the Ray r. The 
    base case sample contribution (from either) is passed 
    through from the beta vector.
*/
int TracePath(Ray r, vector<Vertex>& subPath, int maxDepth, bool isCameraPath, vec3 beta) {

    if (maxDepth == 0) {
        return 0;  
    }

    int bounces = 0;
    float pdfFwd = uniformHemisphereSamplePDF(1);
    float pdfRev = 0;

    while(true){
        Intersection point;
        point.triangleIndex = subPath[bounces].surfaceIndex; // avoid self intersection

        if (!ClosestIntersection(r.o, r.d, triangles, point)) { // Nothing was hit.
            break; 
        }

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
        r1.d = uniformHemisphereSample(snormal, 1); // regardless of which path, sample uniformly

        /* Construct vertex */
        vertex.position = point.position;
        vertex.normal = snormal;
        vertex.surfaceIndex = point.triangleIndex;

        // convert to area based density for by applying solidangle-to-area conversion
        vertex.pdfFwd = pdfFwd * DirectionToAreaConversion(*prev, vertex);

        // give currently constructed sample contribution to the 
        // current vertex
        vertex.c = beta;

        // add vertex to path
        subPath.push_back(vertex);

        // Refetch previous node in case of vector resizing
        prev = &subPath[bounces-1];

        // terminate if a light source was reached
        if(triangles[point.triangleIndex]->emission > 0){
            break;
        }

        // pdfFwd: Probability of sampling the next direction from current.
        // pdfRev: Probability of current sampling a direction towards previous. 
        // so pdfRev is calculated in reversed direction compared to direction of path generation.

        // Regardless of path, sample uniformly
        pdfFwd = uniformHemisphereSamplePDF(1);
        pdfRev = uniformHemisphereSamplePDF(1);

        prev->pdfRev = pdfRev * DirectionToAreaConversion(vertex, *prev);

        // append the contribution from the current intersection point 
        // onto the total sample contribution "beta"

        vec3 brdf = BRDF(vertex, r1.d, r.d, triangles, isCameraPath); 

        /*
            One of the many nested surface integral samples.

            Formally according to the measurement contribution 
            function, this calculation should be:

            *= (brdf * G(current, next) / (pdfFwd * DirectionToAreaConversion(current, next)))

            However certain factors in G and DirectionToAreaConversion
            require knowledge about the next point which we do not have.
            Luckily, these factors cancel out, and we only have to worry
            about the factor remaining in the calculation below.
        */

        // Using the old direction (r.d) is incorrect and 
        // will produce artefacts.
        beta *= (brdf * glm::abs(glm::dot(r1.d, snormal) / pdfFwd));

        // Don't forget to update the ray for the next iteration!
        r = r1;
    }

    return bounces;
}

/*
    Calculate multi-sample estimator using MIS. The light and 
    eye path are connected in different ways to form the many
    path sampling strategies specified in the report. The 
    contributions from these are then weighted with the balance 
    heuristic and added together, forming the multi-sample 
    estimator.
*/
vec3 connect(vector<Vertex>& lightPath, vector<Vertex>& eyePath){
    int t = lightPath.size();
    int s = eyePath.size();

    float basicScale = 1; //1 / (s + t + 2.);
    vec3 F;

    // t == 0, s >= 2:
    for(int i = 2; i <= s; ++i){
        Vertex &last = eyePath[i-1];
        if(triangles[last.surfaceIndex]->emission > 0){
            // get light emitted from last light to second last element of eyePath
            vec3 lightToPoint = glm::normalize(eyePath[i-2].position - last.position); // maybe i should cosine weight the emittance using this?
            vec3 Le = vec3(triangles[last.surfaceIndex]->emission, triangles[last.surfaceIndex]->emission, triangles[last.surfaceIndex]->emission);
            F += Le * last.c * MIS(lightPath, eyePath, 0, i);
        }
    }

    // t >= 1, s >= 2:
    for(int i = 1; i <= t; ++i){
        for(int j = 2; j <= s; ++j){

            // perform visibility test that is related to the geometry 
            // term G. Only calculate contribution from this path 
            // sampling strategy if connection is possible.
            Intersection otherObj;
            otherObj.triangleIndex = lightPath[i-1].surfaceIndex; // previous vertex
            if(!ClosestIntersection(lightPath[i-1].position, (eyePath[j-1].position - lightPath[i-1].position), triangles, otherObj)){
                continue;
            } else {
                if(otherObj.triangleIndex != eyePath[j-1].surfaceIndex && otherObj.t > 0.01f){
                    continue;
                } else {
                
                /*
                    special case of light BRDF is to avoid 
                    index out of bounds. It essentially 
                    simulates "not using the light BRDF" if 
                    we are connecting directly to the light 
                    source. 

                    This is correct behavior, since there is 
                    no real meaning to the BRDF at the light 
                    source (There is no "incoming direction")
                */

                F += lightPath[i-1].c * eyePath[j-1].c 
                        * (i > 1 ? BRDF(lightPath[i-1], (lightPath[i-2].position - lightPath[i-1].position), (eyePath[j-1].position - lightPath[i-1].position), triangles, true) : vec3(1,1,1))
                        * G(lightPath[i-1], eyePath[j-1])
                        * BRDF(eyePath[j-1], (lightPath[i-1].position - eyePath[j-1].position), (eyePath[j-2].position - eyePath[j-1].position), triangles, true)
                        * triangles[eyePath[j-1].surfaceIndex]->color / float(PI)
                        * MIS(lightPath, eyePath, i, j);
                }
            } 
        }
    }

    // F is now the result of the multi-sample estimation, aka 
    // is a multi-sample estimator F. This is because we have taken 
    // a sample of a number of strategies defined by their pdf:s 
    // p(s,t). Note that the pdf division was baked into the 
    // calculation of the pre-computed sample contribution.

    return F; 
}