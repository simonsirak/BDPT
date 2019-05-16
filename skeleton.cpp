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
	vec3 c; // contribution from a path that crosses to the other sub-path from this vertex
	int surfaceIndex; // index of surface collided with
	vec3 normal; // correctly oriented normal of surface collided with
	vec3 position;
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

int numSamples = 25;

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

float G(vec3 na, vec3 nb, vec3 ab);
vec3 connect(vector<Vertex>& lightPath, vector<Vertex>& eyePath);
float pst(float ps, float pt);

//vec3 TracePath(vec3 start, vec3 dir, int depth);
int TracePath(Ray r, vector<Vertex>& subPath, int maxDepth, bool isRadiance);
int GenerateEyePath(int x, int y, vector<Vertex>& subPath, int maxDepth);
int GenerateLightPath(vector<Vertex>& subPath, int maxDepth);

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

				vec3 normal(
							0, 
                            0, 
                            focalLength
						);

                // dir = dir;
                dir = R * dir; // direction is rotated by camera rotation matrix
				normal = R * normal;
                // dir = glm::normalize(dir); // normalize direction, crucial!!

				vector<Vertex> lightPath;
				vector<Vertex> eyePath;
				lightPath.push_back({1, vec3(1,1,1), -1, vec3(), vec3()}); // i = 0, so collided surface is nothing
				eyePath.push_back({1, vec3(1,1,1), -1, vec3(), vec3()}); // i = 0, so collided surface is nothing
				
				// i = 1

				std::uniform_real_distribution<float> dis(0, 1.0);

				float theta0 = 2*PI*dis(rd);
				float theta1 = acos(1 - 2*dis(rd));

				vec3 newdir = glm::normalize(vec3(sin(theta1)*sin(theta0), sin(theta1)*cos(theta0), cos(theta1)));

				theta0 = 2*PI*dis(rd);
				theta1 = acos(1 - 2*dis(rd));

				vec3 newnewdir = glm::normalize(vec3(sin(theta1)*sin(theta0), sin(theta1)*cos(theta0), cos(theta1)));
				newnewdir = glm::dot(newnewdir, newdir) * newdir / glm::dot(newdir, newdir);

				// P1, C1 for Light
				Sphere * light = dynamic_cast<Sphere*>(triangles[triangles.size()-1]);
				float p1 = float(1/(light->r*light->r*4*PI*2*PI*2*PI)); // probability is 1/(AreaOfLight*2PI), then another 2PI for choosing arbitrary direction
				lightPath.push_back({p1, vec3(light->emission/p1,light->emission/p1,light->emission/p1), int(triangles.size()-1), newdir, light->c + float(light->r)*newdir}); // light is on last index

				// P1, C1 for Eye
				p1 = 1; // for now, we assume area of a pixel is perfectly covered by one ray.
				// We calculation: http://rendering-memo.blogspot.com/2016/03/bidirectional-path-tracing-3-importance.html
				float cosTheta = glm::normalize(dir).z; // think about the focalLength triangle, we want the angle.
				float G = cosTheta*cosTheta / (dir.length() * dir.length());
				float We = 1; // float(1 / (1 * G)); // 1 / (AreaOfFilmAkaPixel * G), vi använder inte lens, vi kör 0-size aperture
				eyePath.push_back({p1, vec3(We/p1,We/p1,We/p1), -1, glm::normalize(normal), vec3()}); // surface of a pixel is not really a collided surface

				// Trace eye path
				TracePath(Ray(cameraPos, dir), eyePath, 2, 5);

				// Trace light path

				// for now, direction of ray = direction from center to point. Should change to random hemisphere direction later.
				TracePath(Ray(light->c + float(light->r + 0.001f)*newdir, newnewdir), lightPath, 2, 5);

				buffer[x][y] += connect(lightPath, eyePath) / float(numSamples);

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
	eyePath.push_back({1, vec3(1,1,1), -1, vec3(), cameraPos}); // the probability up to this point is 1

	return TracePath(Ray(cameraPos, dir), eyePath, maxDepth - 1, true) + 1;


}

int GenerateLightPath(vector<Vertex>& lightPath, int maxDepth){

	if(maxDepth == 0)
		return 0;

	float lightChoiceProb = 1; // only one light atm
	float lightPosProb = float(1/(light->r*light->r*4*PI)); // 1 / Area
	float lightDirProb = float(1/(2*PI)); // hemisphere uniform, not cosine weighted
	float pointProb = lightChoiceProb * lightPosProb * lightDirProb;

	std::uniform_real_distribution<float> dis(0, 1.0);

	float theta0 = 2*PI*dis(rd);
	float theta1 = acos(1 - 2*dis(rd));

	// direction from center of sphere to go from
	vec3 offset = glm::normalize(vec3(sin(theta1)*sin(theta0), sin(theta1)*cos(theta0), cos(theta1)));

	theta0 = 2*PI*dis(rd);
	theta1 = acos(1 - 2*dis(rd));

	// direction of ray from point 
	vec3 dir = glm::normalize(vec3(sin(theta1)*sin(theta0), sin(theta1)*cos(theta0), cos(theta1)));
	dir = glm::dot(dir, offset) * offset / glm::dot(offset, offset);

	Sphere * light = dynamic_cast<Sphere*>(triangles[triangles.size()-1]);
	
	vec3 Le = vec3(light->emission, light->emission, light->emission);
	lightPath.push_back({1, vec3(1,1,1), -1, vec3(), vec3()}); 

	//// note, not cosine weighted, we just calculate the cosine term of the LTE.
	// vec3 Le = vec3(light->emission, light->emission, light->emission) * glm::dot(offset, dir) / pointProb;
	return TracePath(Ray(light->c + float(light->r + 0.001f)*offset), dir, lightPath, maxDepth - 1) + 1;
}

int TracePath(Ray r, vector<Vertex>& subPath, int maxDepth, bool isRadiance) {

	if (maxDepth == 0) {
		return 0;  // Bounced enough times
	}

	int bounces = 0;

	/*
	
	IN LACK OF A BETTER PLACE TO COMMENT THIS:

	The measurement equation is EZ. Look, the throughput function is incrementally calculated.
	The rest of the integrand is then sampled one by one. In the beginning of the light path,
	we calculate one "factor" of the throughput function. The beta will provide that factor, 
	i.e "how much importance/radiance we have at this point". This is only for light path however.

	For eye path, you begin on the other end of the measurement equation, i.e you calculate 
	We. Then you incrementally multiply in the throughput function from this direction instead.

	The connection procedure then attempts to "close" each sub path so that they have a light endpoint 
	and a camera endpoint. 

	The only thing that is not quite clear to me is why we use a PDFRev, or how it is calculated.

	However it is important to note that the measurement equation is a sum of the contributions of all paths
	in the scene onto a given pixel (if you think of the Path Integral form of LTE). A sample of the measured
	radiance Ij in the measurement equation is calculated by summing the contributions from a few different strategies
	achieved by connecting a light path and eye path. We then calculate many such samples and get a monte carlo
	estimate of the correct measured radiance Ij, since at the end of the day, the above calculations were just
	one sample of the Ij.
	
	*/

	Intersection point;
	if (!ClosestIntersection(r.o, r.d, triangles, point)) {
		return 0;  // Nothing was hit.
	}

	/*
		r.d is incoming direction to our current position, i.e the intersection point.
		
		newdir is the outgoing direction.

		These can be used to calculate the cosThetaPrime and cosTheta for the probability
		and contribution of this point.
	*/

	// Pick a random direction from here and keep going.

	vec3 newstart = point.position;

	// RNG vector in hemisphere
	std::uniform_real_distribution<float> dis(0, 1.0);

	float theta0 = 2*PI*dis(rd);
	float theta1 = acos(1 - 2*dis(rd));

	vec3 newdir = vec3(sin(theta1)*sin(theta0), sin(theta1)*cos(theta0), cos(theta1)); // http://corysimon.github.io/articles/uniformdistn-on-sphere/ THIS WAS A BIG ISSUE, WE FOLLOWED THE STACK OVERFLOW PIECES OF SHIEET. This fixed the "cross light on the ceiling"
	vec3 surfaceToLight = glm::normalize(r.o - newstart);

	newdir = glm::normalize(newdir);
	// enforce direction to be in correct hemisphere, aka project normal onto random vector and normalize
	vec3 normal = glm::dot(surfaceToLight, triangles[point.triangleIndex]->normal(newstart)) * triangles[point.triangleIndex]->normal(newstart) / glm::dot(triangles[point.triangleIndex]->normal(newstart), triangles[point.triangleIndex]->normal(newstart));
	normal = glm::normalize(normal); // HAS to be normalized after projection onto another vector.
	newdir = glm::dot(newdir, normal) * newdir / glm::dot(newdir, newdir);
	newdir = glm::normalize(newdir);


	// Compute the BRDF for this ray (assuming Lambertian reflection)
	// if i = 2 for the eye, then we have the normal being the focalLength-vector rotated according to camera. But we 
	// won't rotate camera for now, so just focalLength vector

	/* calculating p_i */
	float g = G(subPath[i-1].normal, normal, point.position - r.o);
	const float p = 1 /(2.f*PI); // dot between normal and origin ray (ONLY FOR EYE PATH) (THIS ONE IS USED ONLY FOR LAMBERTIAN REFLECTION SINCE OTHERWISE THE BRDF CANCEL EACH OTHER OUT)
	
	float p_i = p * g * subPath[i-1].p;

	/* calculating C_i */

	vec3 C_i;
	if(i == 2){
		// (subPath[i-1].c * subPath[i-1].p) -> L0 o W0
		// ska egentligen vara W1 och L1, men använder W0 och L0 
		// och antar att spatial och directional light/measurement e samma
		// för våra ljus
		C_i = (subPath[i-1].c * subPath[i-1].p) * subPath[i-1].c / p;
	} else {
		// Note: BRDF of last node, gotten from second last node, in direction of current node.
		vec3 BRDF = triangles[subPath[i-1].surfaceIndex]->color / float(PI); // just the reflectance of the material / PI for Lambertian Reflection: http://www.oceanopticsbook.info/view/surfaces/lambertian_brdfs
		C_i = BRDF * subPath[i-1].c / p;
	}

	subPath.push_back({p_i, C_i, point.triangleIndex, normal, point.position});

	float cos_theta = glm::dot(newdir, normal); // note that the cosine factor is part of the Rendering Equation integrand/differential, NOT the BRDF.

	// don't continue if you are on light
	int bounces = 0;
	if(triangles[point.triangleIndex]->emission == 0)
		bounces = TracePath(Ray(newstart + 0.001f*normal, newdir), subPath, t - 1);

	return bounces + 1;
}

/* connects all possible paths */
/* right now, it is unweighted */
vec3 connect(vector<Vertex>& lightPath, vector<Vertex>& eyePath){
	int s = lightPath.size() - 1;
	int t = eyePath.size() - 1;
	vector<Vertex> path(lightPath);
	path.insert(path.end(), eyePath.begin(), eyePath.end());
	vec3 color;

				//cout << "hi " << endl;
	// s = 0

	float p2sumInv = 1/((s+t+1)*1/((2*PI)*(2*PI)));

	int k = s + t - 1;
	vector<float> p(k + 2, 0.f);
	p[s] = 1;

	//cout << "Max size is: " << path.size() << endl;

	for(int i = s; i <= k; ++i){
		p[i+1] = path[i].p / path[s + t - i].p * p[i]; 
		//cout << i << endl;
	}

	for(int i = s - 1; i >= 0; --i){
		p[i] = path[s + t - i].p / path[i].p * p[i+1]; 
		//cout << i << endl;
	}

	float p2sum = 0;
	for(int i = 0; i <= k+1; ++i){
		p2sum += p[i]*p[i];
	}

	if(s == 1){
		for(int i = 2; i <= t; ++i){ // z_{0} is sorta defined, so we want this
			float emission = triangles[eyePath[i].surfaceIndex]->emission;
			color += vec3(emission, emission, emission) * eyePath[i].c * lightPath[1].c / p2sum; // our light is a sphere light which shoots same in every direction, this may not scale to other kinds of lights.
		}
					cout << "s == 0 " << endl;

		return color;
	}

	// t = 0

	if(t == 1){
		// I'm supposed to use We but I cannot understand what it is; the only thing i dont grasp at all.
		for(int i = 2; i <= s; ++i){ // z_{0} is sorta defined, so we want this
			// vec3 emission = eyePath[1].c * eyePath[1].p * p2sumInv * lightPath[0].p * lightPath[0].p; // triangles[eyePath[1].surfaceIndex]->emission;
			// color += emission; // our light is a sphere light which shoots same in every direction, this may not scale to other kinds of lights.
		}	
					cout << "t == 0 " << endl;

		return color;
	}
	
			//cout << "s, t > 0 " << endl;

	// s, t > 0

	for(int i = 2; i <= s; ++i){
		for(int j = 2; j <= t; ++j){ // eye is not really a surface so im not counting it?
			//cout << i << " " << j << endl;
			Intersection otherObj;
			if(!ClosestIntersection(lightPath[i].position + 0.001f*lightPath[i].normal, (eyePath[j].position - lightPath[i].position), triangles, otherObj)){
				continue;
			} else {
				if(otherObj.t < glm::length(lightPath[i].position - eyePath[j].position)){
					continue;
				} else {
					// Assume lambertian surface
					color += lightPath[i].c * eyePath[j].c * triangles[lightPath[i].surfaceIndex]->color / float(PI)
						  *  G(lightPath[i].normal, eyePath[j].normal, eyePath[j].position - lightPath[i].position)
						  *  (j == 1 ? eyePath[j].c * eyePath[j].p : triangles[eyePath[j].surfaceIndex]->color) / float(PI)
						  / p2sum;
					/* This samamamich line is supposed to have that last commented out factor */ 
					//color += /*triangles[lightPath[i].surfaceIndex]->color / float(PI) * G(lightPath[i].normal, eyePath[j].normal, eyePath[j].position - lightPath[i].position);*/ triangles[eyePath[j].surfaceIndex]->color / float(PI);
				}
			}

		}
	}

	return color;
}

// na = normal at a, nb = normal at b, ab = vector from a to b.
float G(vec3 na, vec3 nb, vec3 ab){
	float cosTheta = glm::dot(glm::normalize(na), glm::normalize(ab)); // angle between outgoing and normal at a
	float cosThetaPrime = glm::dot(glm::normalize(na), glm::normalize(-ab)); // angle between ingoing and normal at b
	return abs(cosTheta*cosThetaPrime) / glm::dot(ab, ab);
}

float pst(float ps, float pt){
	return ps * pt;
}