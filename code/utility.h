#include <glm/glm.hpp>
#include <SDL.h>
#include <cmath>
#include "SDLauxiliary.h"
#include "TestModel.h"
#include <random>

#define PI 3.141592653589793238462643383279502884

/*
    More data structures.

    Intersection structure is largely the same as 
    Lab 2 of rendering track.

    Vertex structure is used for storing information
    about each vertex in a subpath.
*/
struct Intersection{
    vec3 position;
    vec3 normal;
    float t;
    int triangleIndex;
};

struct Vertex {
    float pdfFwd; // probability of reaching this particular vertex regular way
    float pdfRev; // probability of reaching this particular vertex from the reversed direction, e.g through the other path
    vec3 c; // contribution from a path that crosses to the other sub-path from this vertex
    int surfaceIndex; // triangles[]-index of surface collided with

    vec3 normal; // correctly oriented normal of surface collided with
    vec3 position;
    vec3 dir; // outgoing direction
};

std::random_device rd;  //Will be used to obtain a seed for the random number engine

using namespace std;
using glm::vec3;
using glm::mat3;

/*
    Calculates and returns the orthogonal 
    projection of vector a onto vector b.
*/
vec3 projectAOntoB(const vec3 & a, const vec3 & b){
    vec3 c = glm::dot(a, b) * b / glm::dot(b, b);
    return c;
}

/*
    Returns a uniform sphere sample.
*/
vec3 uniformSphereSample(float r){

    std::uniform_real_distribution<float> dis(0, 1.0);

    float theta0 = 2*PI*dis(rd);
    float theta1 = acos(1 - 2*dis(rd));

    vec3 dir = vec3(sin(theta1)*sin(theta0), sin(theta1)*cos(theta0), cos(theta1)); 

    dir = r * glm::normalize(dir);

    return dir;
}

/*
    PDF for uniform sphere sample.
*/
float uniformSphereSamplePDF(float r){
    return 1/(r*r*4*PI);
}

/*
    Sample hemisphere uniformly around an axis.
*/
vec3 uniformHemisphereSample(const vec3 & axis, float r){

    std::uniform_real_distribution<float> dis(0, 1.0);

    float theta0 = 2*PI*dis(rd);
    float theta1 = acos(1 - 2*dis(rd));

    vec3 dir = vec3(sin(theta1)*sin(theta0), sin(theta1)*cos(theta0), cos(theta1)); 

    dir = glm::normalize(projectAOntoB(axis, dir));

    return dir;
}

/*
    Get PDF for a uniform hemisphere sample.
*/
float uniformHemisphereSamplePDF(float r){
    return 1 / (r*r*2*PI);
}

/*
    Calculates the BRDF. Currently only supports 
    lambertian reflection (shape type == 1).
*/
vec3 BRDF(Vertex &vert, vec3 wo, vec3 wi, vector<Obj*> &shapes, bool keepDirections){
    if(!keepDirections){ // if we are on light path
        // Path is generated in other direction, so 
        // flip incident and outgoing for correct 
        // BRDF (only matters for non-diffuse 
        // reflection).
        vec3 tmp = wo;
        wo = wi;
        wi = tmp;
    }

    Obj* shape = shapes[vert.surfaceIndex];

    if(shape->type == 1){ // lambertian
        return shape->color / float(PI);
    } else if(shape->type == 2){ // phong
        // FUTURE EXTENSION: Do Phong reflection.
    }
}

/*
    Returns the geometry term between two vertices.

    This geometry term does not include visibility
    checks. It is done explicitly before using G,
    so it only has to be done only when we don't 
    know whether there is visibility.
*/
float G(const Vertex& a, const Vertex& b){
    vec3 ab = b.position - a.position;
    float cosTheta = glm::dot(glm::normalize(a.normal), glm::normalize(ab)); // angle between outgoing and normal at a
    float cosThetaPrime = glm::dot(glm::normalize(b.normal), glm::normalize(-ab)); // angle between ingoing and normal at b
    return abs(cosTheta*cosThetaPrime) / glm::dot(ab, ab);
}

/*
    Calculates the conversion factor used when 
    converting a direction-based probability 
    to an area-based probability. Follows the 
    description of conversion provided by 
    chapter 8.2.2.2 of Veach's PhD thesis.

    Input: Two vertices on a subpath.
*/
float DirectionToAreaConversion(const Vertex& a, const Vertex& b){
    vec3 w = b.position - a.position;
    float invDist2 = 1 / glm::dot(w, w);
    return invDist2 * glm::abs(glm::dot(b.normal, glm::normalize(w)));
}

/*
    Calculates the weighting function
    for a certain connection of the light
    and eye subpaths, specified by the 
    indices s and t.

    The weighting function implemented here
    is the balance heuristic

    The method of computation was inspired 
    by the implementation in chapter 16.3
    of the book PBRT (3rd ed). The loops 
    were modified to consider only the path 
    sampling strategies specified in the 
    report. This was crucial in order for
    the weighting function to be correct.
*/
float MIS(
    vector<Vertex>& lightVertices, 
    vector<Vertex>& eyeVertices, 
    int t,
    int s
) {
    if (s + t == 2) return 1;
    float sumRi = 0;

    // Define helper function _remap0_ that deals with Dirac delta functions
    // e.g the beginning positional probability of the camera, which is 1 for 
    // one point of the pixel, and 0 everywhere else. The camera vertex is 
    // never reached here though, but it's just an example of a delta function.
    auto remap0 = [](float f) -> float { return f != 0 ? f : 1; };

    // Consider hypothetical path sampling strategies along the camera subpath
    float ri = 1;
    for (int i = s - 1; i > 1; --i) {
        ri *= (remap0(eyeVertices[i].pdfRev) / remap0(eyeVertices[i].pdfFwd));
        sumRi += ri;
    }

    // Consider hypothetical path sampling strategies along the light subpath
    ri = 1;
    for (int i = t - 1; i >= 0; --i) {
        ri *= remap0(lightVertices[i].pdfRev) / remap0(lightVertices[i].pdfFwd);
        sumRi += ri;
    }

    return 1 / (1 + sumRi);
}