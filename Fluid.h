#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "error.h"
#include "xoshiro.h"
#include "color.h"

#include <cstdint>
#include <cmath>
#include <chrono>
#include <SFML/Graphics.hpp>

using namespace sf;
using namespace std;

struct Vec2f {
	float x = 0;
	float y = 0;

	__host__ __device__ constexpr Vec2f() = default;
	__host__ __device__ constexpr Vec2f(float _x, float _y) {
		x = _x;
		y = _y;
	}

	__host__ __device__ friend Vec2f operator*(Vec2f rhs, float lhs) {
		return { rhs.x * lhs, rhs.y * lhs };
	}
	__host__ __device__ friend Vec2f operator*(float rhs, Vec2f lhs) {
		return lhs * rhs;
	}

	__host__ __device__ Vec2f operator+(Vec2f other) {
		return { x + other.x, y + other.y };
	}

	__host__ __device__ float angle() {
		constexpr float PI = 3.1415926f;
		float a = atanf(y / (fabsf(x) + 0.001f));
		if (x < 0) a = PI - a;
		return fmodf(fmodf(a, 2 * PI) + 2 * PI, 2 * PI);
	}

	__host__ __device__ float magnitude() {
		return sqrtf(x * x + y * y);
	}
};
struct RGBA {
	uint8_t r = 0;
	uint8_t g = 0;
	uint8_t b = 0;
	uint8_t a = 255;

	__host__ __device__ constexpr RGBA() {
		r = 0;
		g = 0;
		b = 0;
		a = 255;
	}

	__host__ __device__ constexpr RGBA(uint8_t w) {
		r = w;
		g = w;
		b = w;
		a = 255;
	}

	__host__ __device__ constexpr RGBA(uint8_t _r, uint8_t _g, uint8_t _b) {
		r = _r;
		g = _g;
		b = _b;
		a = 255;
	}

	__device__ bool operator==(const RGBA right) {
		return r == right.r && g == right.g && b == right.b && a == right.a;
	}

	__device__ bool operator!=(const RGBA right) {
		return !(*this == right);
	}

	__device__ bool isEmpty() {
		return r == 0 && g == 0 && b == 0;
	}
};
struct PixelData {
	float density = 0;
	float pressure = 0;
	float ink = 0;
	Vec2f velocity = { 0, 0 };

	float nextDensity = 0;
	float nextInk = 0;
	Vec2f nextVelocity = { 0, 0 };
};
struct MouseInput {
	bool lmbDown;
	bool rmbDown;
	Vec2f position;
	Vec2f velocity;
};

enum class BoundaryCondition {
	Loop,
	Fixed,
	Reflect,
};

__device__ __constant__ int w;
__device__ __constant__ int h;
__device__ int dIters = 0;
int hIters = 0;

Texture tex;
RGBA* hPixels;
RGBA* dPixels;
PixelData* dPixelData;
__device__ __constant__ RGBA* texture;
__device__ __constant__ PixelData* pixelData;

constexpr auto NUMBLOCKS = 240;
constexpr auto BLOCKSIZE = 128;

constexpr auto DOWNSCALE = 1;
constexpr auto LOOPCOUNT = 2;

constexpr auto INPUT_RADIUS = 5;

constexpr BoundaryCondition BOUNDARY_NX = BoundaryCondition::Loop;
constexpr BoundaryCondition BOUNDARY_PX = BoundaryCondition::Loop;
constexpr BoundaryCondition BOUNDARY_NY = BoundaryCondition::Fixed;
constexpr BoundaryCondition BOUNDARY_PY = BoundaryCondition::Fixed;

__device__ static constexpr inline bool SetBoundary(int& c, int m, bool write, BoundaryCondition neg, BoundaryCondition pos) {
	if (c <= 0) {
		if (c == 0) {
			return neg == BoundaryCondition::Fixed && write;
		}
		else {
			if (neg == BoundaryCondition::Loop) c = m - 1;
			return neg != BoundaryCondition::Loop;
		}
	}
	else if (c >= m - 1) {
		if (c == m - 1) {
			return pos == BoundaryCondition::Fixed && write;
		}
		else {
			if (pos == BoundaryCondition::Loop) c = 0;
			return pos != BoundaryCondition::Loop;
		}
	}
	else return false;
}

__device__ PixelData* GetPixelData(int x, int y) {
	return pixelData + (y * w + x);
}

__device__ RGBA GetPixelColor(int x, int y) {
	return texture[y * w + x];
}
__device__ void SetPixelColor(RGBA c, int x, int y) {
	texture[y * w + x] = c;
}
__device__ void SetPixelData(PixelData data, int x, int y) {
	pixelData[y * w + x] = data;
}

#define GetVal(fieldname, _x, _y, default) [&default](int x, int y) {\
	if (SetBoundary(x, w, false, BOUNDARY_NX, BOUNDARY_PX)) return default;\
	if (SetBoundary(y, h, false, BOUNDARY_NY, BOUNDARY_PY)) return default;\
	return GetPixelData(x, y)->fieldname;\
}(_x, _y)

#define SetVal(fieldname, _x, _y, val) [&val](int x, int y) {\
	if (SetBoundary(x, w, true, BOUNDARY_NX, BOUNDARY_PX)) return;\
	if (SetBoundary(y, h, true, BOUNDARY_NY, BOUNDARY_PY)) return;\
	GetPixelData(x, y)->fieldname = val;\
}(_x, _y)

#define GradVal(fieldname, _x, _y) [](int x, int y) {\
	auto implicit_default = decltype(PixelData::fieldname){};\
	auto default = GetVal(fieldname, x, y, implicit_default);\
	Vec2f gradient = {\
		(GetVal(fieldname, x + 1, y, default) - GetVal(fieldname, x - 1, y, default)) * 0.5f,\
		(GetVal(fieldname, x, y + 1, default) - GetVal(fieldname, x, y - 1, default)) * 0.5f\
	};\
	return gradient;\
}(_x, _y)

#define AvgVal(fieldname, _x, _y) [](int x, int y) {\
	auto implicit_default = decltype(PixelData::fieldname){};\
	auto default = GetVal(fieldname, x, y, implicit_default);\
	auto average = (GetVal(fieldname, x + 1, y, default) + GetVal(fieldname, x - 1, y, default) + \
					GetVal(fieldname, x, y + 1, default) + GetVal(fieldname, x, y - 1, default)) * 0.25f;\
	return average;\
}(_x, _y)

//#define PATTERNED_UPDATE
#ifdef PATTERNED_UPDATE
#define DefineTextureFunction(devFn)\
__global__ void Texture_##devFn(bool odd, double t) {\
	uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;\
	uint32_t stride = blockDim.x * gridDim.x;\
	for (int pix = index; pix < w * h / 2; pix += stride) {\
		int y = (pix * 2) / w;\
		int x = pix % (w / 2);\
		x = 2 * x + (y & 1) ^ odd;\
		devFn(x, y, t);\
	}\
}
#define CallTextureFunction(devFn, t)\
Texture_##devFn<<<NUMBLOCKS, BLOCKSIZE>>>(false, t);\
checkCudaErrors(cudaDeviceSynchronize());\
Texture_##devFn<<<NUMBLOCKS, BLOCKSIZE>>>(true, t);\
checkCudaErrors(cudaDeviceSynchronize());
#else
#define DefineTextureFunction(devFn)\
__global__ void Texture_##devFn(double t) {\
	uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;\
	uint32_t stride = blockDim.x * gridDim.x;\
	for (int pix = index; pix < w * h; pix += stride) {\
		int y = pix / w;\
		int x = pix % w;\
		devFn(x, y, t);\
	}\
}
#define CallTextureFunction(devFn, t)\
Texture_##devFn<<<NUMBLOCKS, BLOCKSIZE>>>(t);\
checkCudaErrors(cudaDeviceSynchronize());
#endif

/*
__global__ void HandleInput(MouseInput input) {
	if (input.lmbDown)
		for (int xOffset = -INPUT_RADIUS; xOffset <= INPUT_RADIUS; xOffset++) {
			for (int yOffset = -INPUT_RADIUS; yOffset <= INPUT_RADIUS; yOffset++) {
				if (InvalidCoords(input.position.x + xOffset, input.position.y + yOffset)) continue;
				Vec2f vel = GetPixelData(input.position.x + xOffset, input.position.y + yOffset)->velocity[0] + input.velocity * 10;
				GetPixelData(input.position.x + xOffset, input.position.y + yOffset)->velocity[0] = vel;
				GetPixelData(input.position.x + xOffset, input.position.y + yOffset)->velocity[1] = vel;
			}
		}
	else if (input.rmbDown)
		for (int xOffset = -INPUT_RADIUS; xOffset <= INPUT_RADIUS; xOffset++)
		{
			for (int yOffset = -INPUT_RADIUS; yOffset <= INPUT_RADIUS; yOffset++)
			{
				if (InvalidCoords(input.position.x + xOffset, input.position.y + yOffset)) continue;
				GetPixelData(input.position.x + xOffset, input.position.y + yOffset)->pressure[0] += 10;
			}
		}
}

*/

__device__ void Advect(int x, int y, double t) {
	PixelData* data = GetPixelData(x, y);
	float lastX = x - data->velocity.x * t;
	float lastY = y - data->velocity.y * t;

	int lastXInt = floorf(lastX);
	int lastYInt = floorf(lastY);

	float lastXFrac = lastX - lastXInt;
	float lastYFrac = lastY - lastYInt;

	float TL = (1 - lastXFrac) * (1 - lastYFrac);
	float TR = lastXFrac * (1 - lastYFrac);
	float BL = (1 - lastXFrac) * lastYFrac;
	float BR = lastXFrac * lastYFrac;

	float dD = data->density;
	float nD = TL * GetVal(density, lastXInt, lastYInt, dD) +
		TR * GetVal(density, lastXInt + 1, lastYInt, dD) +
		BL * GetVal(density, lastXInt, lastYInt + 1, dD) +
		BR * GetVal(density, lastXInt + 1, lastYInt + 1, dD);

	Vec2f dV = data->velocity * -1;
	Vec2f nV = TL * GetVal(velocity, lastXInt, lastYInt, dV) +
		TR * GetVal(velocity, lastXInt + 1, lastYInt, dV) +
		BL * GetVal(velocity, lastXInt, lastYInt + 1, dV) +
		BR * GetVal(velocity, lastXInt + 1, lastYInt + 1, dV);


	float dI = data->ink;
	float nI = TL * GetVal(ink, lastXInt, lastYInt, dI) +
		TR * GetVal(ink, lastXInt + 1, lastYInt, dI) +
		BL * GetVal(ink, lastXInt, lastYInt + 1, dI) +
		BR * GetVal(ink, lastXInt + 1, lastYInt + 1, dI);

	SetVal(nextDensity, x, y, nD);
	SetVal(nextVelocity, x, y, nV);
	SetVal(nextInk, x, y, nI);
}

__device__ void AdvectCopy(int x, int y, double t) {
	GetPixelData(x, y)->density = GetPixelData(x, y)->nextDensity;
	GetPixelData(x, y)->velocity = GetPixelData(x, y)->nextVelocity;
	GetPixelData(x, y)->ink = GetPixelData(x, y)->nextInk;
}

__device__ void Flux(int x, int y, double t) {

}

__device__ void Divergence(int x, int y, double t) {
	constexpr float z = 0;
	float dP = GetVal(density, x, y, z);

	float gradVX = GradVal(velocity.x, x, y).x;
	float gradVY = GradVal(velocity.y, x, y).y;

	float gradient = (gradVX + gradVY);
	float avgP = AvgVal(pressure, x, y);

	float nP = dP - gradient;
	SetVal(density, x, y, nP);
}

__device__ void UpdateVel(int x, int y, double t) {
	constexpr Vec2f nulVel = { 0, 0 };
	Vec2f dV = GetVal(velocity, x, y, nulVel);

	Vec2f gradP = GradVal(density, x, y);
	Vec2f avgV = AvgVal(velocity, x, y);

	Vec2f nV = dV + (gradP * -1);
	SetVal(velocity, x, y, nV);
}

DefineTextureFunction(Advect)
DefineTextureFunction(AdvectCopy)
DefineTextureFunction(Divergence)
DefineTextureFunction(UpdateVel)

__device__ float sigmoid(float x) {
	return 1 / (1 + powf(2.71828f, -x));
}

// 0: ink, 1: greyscale velmagn, 2: colorized velmagn, 3: greyscale pressure gradient
__global__ void DrawFluid(int scheme) {
	uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
	uint32_t stride = blockDim.x * gridDim.x;
	constexpr float z = 0;
	constexpr Vec2f nulVel = { 0, 0 };
	for (int pix = index; pix < w * h; pix += stride) {
		int x = pix % w;
		int y = pix / w;

		float ink = GetVal(ink, x, y, z);
		float velAngle = GetVal(velocity, x, y, nulVel).angle() / (2 * 3.1415926f);
		float velMagn = fminf(1, GetVal(velocity, x, y, nulVel).magnitude() / 120.f);

		UniversalColor rgb;
		if (scheme == 0) {
			rgb = { ink, ink, 255 - ink };
		}
		else if (scheme == 1) {
			rgb = hsl2rgb({ velAngle, 1, 0 });
		}
		else if (scheme == 2) {
			rgb = hsl2rgb({ 0, 0, velMagn });
		}
		else if (scheme == 3) {
			float dP = GetVal(density, x, y, z);
			Vec2f gradP = { GetVal(density, x + 1, y, dP) - GetVal(density, x - 1, y, dP),
				GetVal(density, x, y + 1, dP) - GetVal(density, x, y - 1, dP) };
			float gpMagn = fminf(1, gradP.magnitude() / 6.f);
			rgb = hsl2rgb({ 0, 0, gpMagn });
		}

		RGBA color(rgb.r, rgb.g, rgb.b);
		SetPixelColor(color, x, y);
	}
}

__global__ void InitTexture(uint64_t seed) {
	uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
	uint32_t stride = blockDim.x * gridDim.x;

	for (int pix = index * 8; pix < w * h; pix += stride * 8) {
		int x = pix % w;
		int y = pix / w;
		Xoshiro256ssc rnd(seed);
		rnd.SetRandomSeed(x, y);

		for (int i = 0; i < 8; i++) {
			int x2 = (pix + i) % w;
			int y2 = (pix + i) / w;
			PixelData* data = GetPixelData(x2, y2);

			//data->velocity = data->nextVelocity = { 0, 0 };
			//data->density = data->nextDensity = 0;
			//if (800 < x2 && x2 < 850 && 650 < y2 && y2 < 700) data->density = data->nextDensity = 1000;

			data->density = data->nextDensity = rnd.Next();
			if (abs(y - h / 2) < h / 3) {
				data->ink = 255;
				data->velocity = data->nextVelocity = { 100, rnd.Next() };
			}
			else {
				data->velocity = data->nextVelocity = { 0, rnd.Next() };
			}
		}
	}
}

void texInit(const int _width, const int _height) {
	int width = _width / DOWNSCALE;
	int height = _height / DOWNSCALE;
	hIters = 0;

	hPixels = (RGBA*)malloc(width * height * 4);
	checkCudaErrors(cudaMalloc(&dPixels, width * height * 4));
	checkCudaErrors(cudaMalloc(&dPixelData, width * height * sizeof(PixelData)));
	checkCudaErrors(cudaMemcpyToSymbol(texture, &dPixels, sizeof(RGBA*)));
	checkCudaErrors(cudaMemcpyToSymbol(pixelData, &dPixelData, sizeof(PixelData*)));

	checkCudaErrors(cudaMemcpyToSymbol(w, &width, sizeof(int)));
	checkCudaErrors(cudaMemcpyToSymbol(h, &height, sizeof(int)));
	checkCudaErrors(cudaMemcpyToSymbol(dIters, &hIters, sizeof(int)));

	InitTexture << <NUMBLOCKS, BLOCKSIZE >> > (1234);
	checkCudaErrors(cudaDeviceSynchronize());

	tex = Texture();
	tex.create(width, height);
}

Vec2f lastMPos = { 0, 0 };


void texRender(RenderWindow& window, const int _width, const int _height, float deltaT) {
	int width = _width / DOWNSCALE;
	int height = _height / DOWNSCALE;
	deltaT = 0.01f;

	Vec2f currentMPos(Mouse::getPosition().x / DOWNSCALE, Mouse::getPosition().y / DOWNSCALE);
	Vec2f diff = currentMPos + (lastMPos * -1);
	lastMPos = currentMPos;
	MouseInput mouseInput = { Mouse::isButtonPressed(Mouse::Left), Mouse::isButtonPressed(Mouse::Right), currentMPos, diff };
	//HandleInput << <1, 1 >> > (mouseInput);

	bool paused = Keyboard::isKeyPressed(Keyboard::Space);

	if (!paused) {
		for (int gLoops = 0; gLoops < LOOPCOUNT; gLoops++) {
			CallTextureFunction(Advect, deltaT);
			CallTextureFunction(AdvectCopy, deltaT);
			CallTextureFunction(Divergence, deltaT);
			CallTextureFunction(UpdateVel, deltaT);
		}
	}
	DrawFluid << <NUMBLOCKS, BLOCKSIZE >> > (0);
	checkCudaErrors(cudaDeviceSynchronize());

	checkCudaErrors(cudaMemcpy(hPixels, dPixels, 4 * width * height, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaDeviceSynchronize());

	//Drawing
	hIters++;
	if (hIters % 4 == 0) tex.update((uint8_t*)hPixels);
	Sprite sprite = Sprite(tex);
	sprite.setScale(Vector2f(DOWNSCALE, DOWNSCALE));
	window.draw(sprite);
}

void texCleanup() {
	cudaFree(dPixels);
	cudaFree(dPixelData);
	free(hPixels);
}