#include <glm/glm.hpp>
#include <SDL.h>
#include <cmath>
#include "SDLauxiliary.h"
#include "TestModel.h"
#include <random>

#define PI 3.141592653589793238462643383279502884

std::random_device rd;  //Will be used to obtain a seed for the random number engine

using namespace std;
using glm::vec3;
using glm::mat3;

vec3 projectAOntoB(const vec3 & a, const vec3 & b){
    vec3 c = glm::dot(a, b) * b / glm::dot(b, b);
    return c;
}

vec3 uniformHemisphereSample(const vec3 & axis){

    std::uniform_real_distribution<float> dis(0, 1.0);

	float theta0 = 2*PI*dis(rd);
	float theta1 = acos(1 - 2*dis(rd));

	vec3 dir = 
        vec3(sin(theta1)*sin(theta0), sin(theta1)*cos(theta0), cos(theta1)); 
    
    dir = glm::normalize(projectAOntoB(axis, dir));
    
    return dir;
}

float uniformHemisphereSamplingPDF(){
    return 1 / 2*PI;
}

float uniformAreaLightSample(const Obj * light){
    // TODO: 1 / Area
    return 0;
}

float uniformAreaLightSamplingPDF(const Obj * light){
    // TODO
    return 0;
}

// dir is direction to the top left of pixel
vec3 uniformFilmSample(const vec3 * dir){
    // TODO
    return dir;
}

float uniformFilmSamplingPDF(){
    // TODO
    return 1;
}

