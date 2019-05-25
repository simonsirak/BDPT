#include <glm/glm.hpp>
#include <SDL.h>
#include <cmath>
#include "SDLauxiliary.h"
#include "TestModel.h"
#include <random>

#define PI 3.141592653589793238462643383279502884

// STRUCTS 
struct Intersection{
	vec3 position;
	vec3 normal;
	float t;
	int triangleIndex;
};

struct Vertex {
	float pdfFwd; // probability of reaching this particular vertex regular way
	float pdfRev; // probability of reaching this particular vertex via reversed path (through the other kind of path)
	vec3 c; // contribution from a path that crosses to the other sub-path from this vertex
	int surfaceIndex; // index of surface collided with
	vec3 normal; // correctly oriented normal of surface collided with
	vec3 position;
	vec3 dir; // rly only relevant to eye endpoint
};

std::random_device rd;  //Will be used to obtain a seed for the random number engine

using namespace std;
using glm::vec3;
using glm::mat3;

vec3 projectAOntoB(const vec3 & a, const vec3 & b){
    vec3 c = glm::dot(a, b) * b / glm::dot(b, b);
    return c;
}

vec3 uniformSphereSample(float r){

    std::uniform_real_distribution<float> dis(0, 1.0);

	float theta0 = 2*PI*dis(rd);
	float theta1 = acos(1 - 2*dis(rd));

	vec3 dir = 
        vec3(sin(theta1)*sin(theta0), sin(theta1)*cos(theta0), cos(theta1)); 
    
    dir = r * glm::normalize(dir);
    
    return dir;
}

float uniformSphereSamplePDF(float r){
    return 1/(r*r*4*PI);
}

vec3 uniformHemisphereSample(const vec3 & axis, float r){

    std::uniform_real_distribution<float> dis(0, 1.0);

	float theta0 = 2*PI*dis(rd);
	float theta1 = acos(1 - 2*dis(rd));

	vec3 dir = 
        vec3(sin(theta1)*sin(theta0), sin(theta1)*cos(theta0), cos(theta1)); 
    
    dir = glm::normalize(projectAOntoB(axis, dir));
    
    return dir;
}

float uniformHemisphereSamplePDF(float r){
    return 1 / (r*r*2*PI);
}

// frame calculation taken from:
// https://github.com/embree/embree-renderer/blob/master/common/math/linearspace3.h
mat3 frame(const vec3 &N){
    vec3 dx0 = glm::cross(vec3(1,0,0),N);
    vec3 dx1 = glm::cross(vec3(0,1,0),N);
    vec3 dx = glm::normalize(glm::dot(dx0,dx0) > glm::dot(dx1,dx1) ? dx0 : dx1);
    vec3 dy = glm::normalize(glm::cross(N,dx));
    return mat3(dx,dy,N);
}

// cosine weighted taken from: 
// https://github.com/embree/embree-renderer/blob/master/devices/device_singleray/samplers/shapesampler.h
// only works for directions as of now because of normalization, not e.g point samples
vec3 cosWeightedHemisphereSample(const vec3 & axis){

    std::uniform_real_distribution<float> dis(0, 1.0);

    const float phi = float(2*PI) * dis(rd);
    const float vv = 2.0f*(dis(rd) - 0.5f);
    const float cosTheta = (vv < 0 ? -1 : 1)*sqrt(abs(vv)), sinTheta = sqrt(glm::max(0.f, 1 - cosTheta*cosTheta));
    return glm::normalize(frame(glm::normalize(axis)) * vec3(cos(phi) * sinTheta, sin(phi) * sinTheta, cosTheta)); //Sample3f(Vector3f(cos(phi) * sinTheta, sin(phi) * sinTheta, cosTheta), 2.0f*cosTheta*float(one_over_pi));
}


// only works for directions as of now, because I deliberately normalize
// PDF corresponds to cosine weighted that was taken from:
// https://github.com/embree/embree-renderer/blob/master/devices/device_singleray/samplers/shapesampler.h
float cosWeightedHemisphereSamplePDF(const vec3 &sampled, const vec3 &axis){
    return 2.0f*abs(glm::dot(glm::normalize(sampled),glm::normalize(axis)))*float(1/PI);
}

float uniformAreaLightSample(const Obj * light){
    // TODO: 1 / Area
    return 0;
}

float uniformAreaLightSamplePDF(const Obj * light){
    // TODO
    return 0;
}

// dir is direction to the top left of pixel
vec3 uniformFilmSample(const vec3 & dir){
    // TODO
    return dir;
}

float uniformFilmSamplingPDF(){
    // TODO
    return 1;
}

vec3 BRDF(Vertex &vert, vec3 wo, vec3 wi, vector<Obj*> &shapes, bool isRadiance){
    if(isRadiance){ // if radiance transport, i.e path starting from eye
        // do nothing
    } else { // Importance transport
        // Path is generated backwards, wo flip incident and outgoing
        // for correct BRDF
        vec3 tmp = wo;
        wo = wi;
        wi = tmp;
    }

    Obj* shape = shapes[vert.surfaceIndex];

    if(shape->type == 1){ // lambertian
        return shape->color / float(PI);
    } else if(shape->type == 2){ // phong
        // calculate reflected direction using Snell's Law
        // calculate Fresnel Reflectance
        // calculate BRDF according to PBR-BOOK
    }
}

float G(vec3 na, vec3 nb, vec3 ab){
	float cosTheta = glm::dot(glm::normalize(na), glm::normalize(ab)); // angle between outgoing and normal at a
	float cosThetaPrime = glm::dot(glm::normalize(nb), glm::normalize(-ab)); // angle between ingoing and normal at b
	return abs(cosTheta*cosThetaPrime) / glm::dot(ab, ab);
}

float MIS(
	vector<Vertex>& lightVertices, 
	vector<Vertex>& eyeVertices, 
	int s,
	int t
) {
	if (s + t == 2) return 1;
    float sumRi = 0;

    // Define helper function _remap0_ that deals with Dirac delta functions
	// e.g the beginning positional probability of the camera, which is 1 for 
	// one point of the pixel, and 0 everywhere else. This is never reached 
	// here tho, but it's just an example of a delta function.
    auto remap0 = [](float f) -> float { return f != 0 ? f : 1; };

	// Consider hypothetical connection strategies along the camera subpath
    float ri = 1;
    for (int i = t - 1; i > 1; --i) {
        ri *=
            remap0(eyeVertices[i].pdfRev) / remap0(eyeVertices[i].pdfFwd);
        sumRi += ri;
    }

    // Consider hypothetical connection strategies along the light subpath
    ri = 1;
    for (int i = s - 1; i >= 0; --i) {
        ri *= remap0(lightVertices[i].pdfRev) / remap0(lightVertices[i].pdfFwd);
        sumRi += ri;
    }
    return 1 / (1 + sumRi);
}