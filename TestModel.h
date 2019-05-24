#ifndef TEST_MODEL_CORNEL_BOX_H
#define TEST_MODEL_CORNEL_BOX_H

// Defines a simple test model: The Cornel Box

#include <glm/glm.hpp>
#include <vector>

using glm::vec3;
using glm::mat3;

// Rays have origin and direction.
// The direction vector should always be normalized.
struct Ray {
	vec3 o, d;
	Ray(vec3 o0 = vec3(0, 0, 0), vec3 d0 = vec3(1, 0, 0)): o(o0), d(glm::normalize(d0)) {}
};

// Objects have color, emission, type (diffuse, specular, refractive)
// All object should be intersectable and should be able to compute their surface normals.
class Obj {
	public:
	vec3 color;
	double emission;
	int type;
	void setMat(vec3 cl_ = vec3(0, 0, 0), double emission_ = 0, int type_ = 0) { color = cl_; emission = emission_ ; type = type_; }
	virtual double intersect(const Ray&) const = 0;
	virtual vec3 normal(const vec3&) const = 0;
};

class Triangle : public Obj {
	public:
	vec3 v0, v1, v2, n;
	Triangle(vec3 v1, vec3 v2, vec3 v3): v0(v1), v1(v2), v2(v3) {
		ComputeNormal();
	}
	double intersect(const Ray& ray) const {
		using glm::vec3;
		using glm::mat3;
		vec3 e1 = v1 - v0;
		vec3 e2 = v2 - v0;
		vec3 b = ray.o - v0;
		mat3 A( -ray.d, e1, e2 );
		vec3 x = glm::inverse( A ) * b;

		if(x.x >= 0 && x.y >= 0 && x.z >= 0 && x.y <= 1 && x.z <= 1 && x.y + x.z <= 1){
            return x.x; // return t
		} else {
            return 0;
        }
	}
	vec3 normal(const vec3& p0) const { return n; }
    void ComputeNormal(){
        glm::vec3 e1 = v1-v0;
		glm::vec3 e2 = v2-v0;
		n = glm::normalize( glm::cross( e2, e1 ) );
    }
};

class Sphere : public Obj {
	public:
	vec3 c;
	double r;

	Sphere(double r_= 0, vec3 c_= vec3(0, 0, 0)) { c=c_; r=r_; }
	double intersect(const Ray& ray) const {
		double b = glm::dot((ray.o-c)*2.f,ray.d);
		double c_ = glm::dot(ray.o-c, ray.o-c) - (r*r);
		double disc = b*b - 4*c_;
		if (disc<=0) return 0;
		else disc = sqrt(disc);
		double sol1 = -b + disc;
		double sol2 = -b - disc;
		return (sol2>=0) ? sol2/2 : ((sol1>=0) ? sol1/2 : 0); // 0.01f is some epsilon to avoid self intersection
	}

	vec3 normal(const vec3& p0) const {
		return glm::normalize(p0 - c);
	}
};

// Loads the Cornell Box. It is scaled to fill the volume:
// -1 <= x <= +1
// -1 <= y <= +1
// -1 <= z <= +1
void LoadTestModel( std::vector<Obj*>& triangles, std::vector<Obj*>& lights )
{
	using glm::vec3;

	// Defines colors:
	vec3 red(    0.75f, 0.15f, 0.15f );
	vec3 yellow( 0.75f, 0.75f, 0.15f );
	vec3 green(  0.15f, 0.75f, 0.15f );
	vec3 cyan(   0.15f, 0.75f, 0.75f );
	vec3 blue(   0.15f, 0.15f, 0.75f );
	vec3 purple( 0.75f, 0.15f, 0.75f );
	vec3 white(  0.75f, 0.75f, 0.75f );

	triangles.clear();
	triangles.reserve( 5*2*3 );

	// ---------------------------------------------------------------------------
	// Room

	float L = 555;			// Length of Cornell Box side.

	vec3 A(L,0,0);
	vec3 B(0,0,0);
	vec3 C(L,0,L);
	vec3 D(0,0,L);

	vec3 E(L,L,0);
	vec3 F(0,L,0);
	vec3 G(L,L,L);
	vec3 H(0,L,L);

	// Floor:
	triangles.push_back( new Triangle( C, B, A ) );
	triangles.push_back( new Triangle( C, D, B ) );
    triangles[triangles.size()-2]->setMat(green, 0, 1);
    triangles[triangles.size()-1]->setMat(green, 0, 1);

	// Left wall
	triangles.push_back( new Triangle( A, E, C ) );
	triangles.push_back( new Triangle( C, E, G ) );
    triangles[triangles.size()-2]->setMat(purple, 0, 1);
    triangles[triangles.size()-1]->setMat(purple, 0, 1);

	// Right wall
	triangles.push_back( new Triangle( F, B, D ) );
	triangles.push_back( new Triangle( H, F, D ) );
    triangles[triangles.size()-2]->setMat(yellow, 0, 1);
    triangles[triangles.size()-1]->setMat(yellow, 0, 1);

	// Ceiling
	triangles.push_back( new Triangle( E, F, G ) );
	triangles.push_back( new Triangle( F, H, G ) );
    triangles[triangles.size()-2]->setMat(cyan, 0, 1);
    triangles[triangles.size()-1]->setMat(cyan, 0, 1);

	// Back wall
	triangles.push_back( new Triangle( G, D, C ) );
	triangles.push_back( new Triangle( G, H, D ) ); // index 9
    triangles[triangles.size()-2]->setMat(white, 0, 1);
    triangles[triangles.size()-1]->setMat(white, 0, 1);

	// ---------------------------------------------------------------------------
	// Short block

	A = vec3(290,0,114);
	B = vec3(130,0, 65);
	C = vec3(240,0,272);
	D = vec3( 82,0,225);

	E = vec3(290,165,114);
	F = vec3(130,165, 65);
	G = vec3(240,165,272);
	H = vec3( 82,165,225);

	// Front
	triangles.push_back( new Triangle(E,B,A) );
	triangles.push_back( new Triangle(E,F,B) );
    triangles[triangles.size()-2]->setMat(red, 0, 1);
    triangles[triangles.size()-1]->setMat(red, 0, 1);

	// Front
	triangles.push_back( new Triangle(F,D,B) );
	triangles.push_back( new Triangle(F,H,D) );
    triangles[triangles.size()-2]->setMat(red, 0, 1);
    triangles[triangles.size()-1]->setMat(red, 0, 1);

	// BACK
	triangles.push_back( new Triangle(H,C,D) );
	triangles.push_back( new Triangle(H,G,C) );
    triangles[triangles.size()-2]->setMat(red, 0, 1);
    triangles[triangles.size()-1]->setMat(red, 0, 1);

	// LEFT
	triangles.push_back( new Triangle(G,E,C) );
	triangles.push_back( new Triangle(E,A,C) );
    triangles[triangles.size()-2]->setMat(red, 0, 1);
    triangles[triangles.size()-1]->setMat(red, 0, 1);

	// TOP
	triangles.push_back( new Triangle(G,F,E) );
	triangles.push_back( new Triangle(G,H,F) ); // index 19
    triangles[triangles.size()-2]->setMat(red, 0, 1);
    triangles[triangles.size()-1]->setMat(red, 0, 1);

	// ---------------------------------------------------------------------------
	// Tall block

	A = vec3(423,0,247);
	B = vec3(265,0,296);
	C = vec3(472,0,406);
	D = vec3(314,0,456);

	E = vec3(423,330,247);
	F = vec3(265,330,296);
	G = vec3(472,330,406);
	H = vec3(314,330,456);

	// Front
	triangles.push_back( new Triangle(E,B,A) );
	triangles.push_back( new Triangle(E,F,B) );
    triangles[triangles.size()-2]->setMat(blue, 0, 1);
    triangles[triangles.size()-1]->setMat(blue, 0, 1);

	// Front
	triangles.push_back( new Triangle(F,D,B) );
	triangles.push_back( new Triangle(F,H,D) );
    triangles[triangles.size()-2]->setMat(blue, 0, 1);
    triangles[triangles.size()-1]->setMat(blue, 0, 1);

	// BACK
	triangles.push_back( new Triangle(H,C,D) );
	triangles.push_back( new Triangle(H,G,C) );
    triangles[triangles.size()-2]->setMat(blue, 0, 1);
    triangles[triangles.size()-1]->setMat(blue, 0, 1);

	// LEFT
	triangles.push_back( new Triangle(G,E,C) );
	triangles.push_back( new Triangle(E,A,C) );
    triangles[triangles.size()-2]->setMat(blue, 0, 1);
    triangles[triangles.size()-1]->setMat(blue, 0, 1);

	// TOP
	triangles.push_back( new Triangle(G,F,E) ); // index 28 & 29
	triangles.push_back( new Triangle(G,H,F) );
    triangles[triangles.size()-2]->setMat(blue, 0, 1);
    triangles[triangles.size()-1]->setMat(blue, 0, 1);


	// ----------------------------------------------
	// Scale to the volume [-1,1]^3

	for( size_t i=0; i<triangles.size(); ++i )
	{
        if(Triangle* t = dynamic_cast<Triangle*>(triangles[i])){
            t->v0 *= 2/L;
            t->v1 *= 2/L;
            t->v2 *= 2/L;

            t->v0 -= vec3(1,1,1);
            t->v1 -= vec3(1,1,1);
            t->v2 -= vec3(1,1,1);

            t->v0.x *= -1;
            t->v1.x *= -1;
            t->v2.x *= -1;

            t->v0.y *= -1;
            t->v1.y *= -1;
            t->v2.y *= -1;

            t->ComputeNormal();
        }
	}

    // triangles.push_back(new Sphere(0.05, vec3(-0.7,  -0.7, -0.5)));
    // triangles[triangles.size()-1]->setMat(white, 60, 1);
    // lights.push_back(triangles[triangles.size()-1]);

    triangles.push_back(new Sphere(0.15, vec3(0,  -0.7, -0.5)));
    triangles[triangles.size()-1]->setMat(white, 60, 1);
    lights.push_back(triangles[triangles.size()-1]);

    // triangles.push_back(new Sphere(0.05, vec3(0.7,  -0.7, -0.5)));
    // triangles[triangles.size()-1]->setMat(white, 60, 1);
    // lights.push_back(triangles[triangles.size()-1]);

    // triangles.push_back(new Sphere(0.05, vec3(-0.7,  -0.7, 0)));
    // triangles[triangles.size()-1]->setMat(white, 60, 1);
    // lights.push_back(triangles[triangles.size()-1]);

    // triangles.push_back(new Sphere(0.05, vec3(0,  -0.7, 0)));
    // triangles[triangles.size()-1]->setMat(white, 60, 1);
    // lights.push_back(triangles[triangles.size()-1]);

    // triangles.push_back(new Sphere(0.05, vec3(0.7,  -0.7, 0)));
    // triangles[triangles.size()-1]->setMat(white, 60, 1);
    // lights.push_back(triangles[triangles.size()-1]);

    // triangles.push_back(new Sphere(0.05, vec3(-0.7,  -0.7, 0.5)));
    // triangles[triangles.size()-1]->setMat(white, 60, 1);
    // lights.push_back(triangles[triangles.size()-1]);

    // triangles.push_back(new Sphere(0.05, vec3(0,  -0.7, 0.5)));
    // triangles[triangles.size()-1]->setMat(white, 60, 1);
    // lights.push_back(triangles[triangles.size()-1]);

    // triangles.push_back(new Sphere(0.05, vec3(0.7,  -0.7, 0.5)));
    // triangles[triangles.size()-1]->setMat(white, 60, 1);
    // lights.push_back(triangles[triangles.size()-1]);
}

#endif