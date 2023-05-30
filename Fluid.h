#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "error.h"
#include "xoshiro.h"

#include <stdint.h>
#include <cmath>
#include <chrono>
#include <SFML/Graphics.hpp>

using namespace sf;
using namespace std;

struct Vec2f
{
	float x = 0;
	float y = 0;

	__host__ __device__ constexpr Vec2f(float _x, float _y)
	{
		x = _x;
		y = _y;
	}

	__host__ __device__ Vec2f operator*(float scalar)
	{
		return { x * scalar, y * scalar };
	}

	__host__ __device__ Vec2f operator+(Vec2f other)
	{
		return { x + other.x, y + other.y};
	}
};

struct PixelData
{
	float density = 0;
	float nextDensity = 0;
	float nextDensityGS = 0;

	Vec2f velocity = { 0, 0 };
	Vec2f nextVelocity = { 0, 0 };
	Vec2f nextVelocityGS = { 0, 0 };

	float p = 0;
	float pGS = 0;

	__host__ __device__ constexpr PixelData()
	{

	}
};

struct RGBA
{
	uint8_t r = 0;
	uint8_t g = 0;
	uint8_t b = 0;
	uint8_t a = 255;

	__host__ __device__ constexpr RGBA()
	{
		r = 0;
		g = 0;
		b = 0;
		a = 255;
	}

	__host__ __device__ constexpr RGBA(uint8_t w)
	{
		r = w;
		g = w;
		b = w;
		a = 255;
	}

	__host__ __device__ constexpr RGBA(uint8_t _r, uint8_t _g, uint8_t _b)
	{
		r = _r;
		g = _g;
		b = _b;
		a = 255;
	}

	__device__ bool operator==(const RGBA right)
	{
		return r == right.r && g == right.g && b == right.b && a == right.a;
	}

	__device__ bool operator!=(const RGBA right)
	{
		return !(*this == right);
	}

	__device__ bool isEmpty()
	{
		return r == 0 && g == 0 && b == 0;
	}
};

struct MouseInput
{
	bool lmbDown;
	bool rmbDown;
	Vec2f position;
	Vec2f velocity;
};

__device__ __constant__ RGBA* texture;
__device__ __constant__ PixelData* pixelData;

__device__ __constant__ int w;
__device__ __constant__ int h;
__device__ int dIters = 0;
int hIters = 0;

RGBA* hPixels;
RGBA* dPixels;

PixelData* dPixelData;

constexpr auto NUMBLOCKS = 256;
constexpr auto BLOCKSIZE = 128;

constexpr auto DOWNSCALE = 2;
constexpr auto LOOPCOUNT = 4;

constexpr auto INPUT_RADIUS = 5;

Texture tex;

__device__ bool InvalidCoords(int x, int y)
{
	return x < 0 || x >= w || y < 0 || y >= h;
}

__device__ PixelData* GetPixelData(int x, int y)
{
	return pixelData + (y * w + x);
}
__device__ RGBA GetPixel(int x, int y)
{
	return texture[y * w + x];
}
__device__ void SetPixel(RGBA c, int x, int y)
{
	texture[y * w + x] = c;
}
__device__ void SetPixelData(PixelData data, int x, int y)
{
	pixelData[y * w + x] = data;
}

__device__ float GetDensity(int x, int y, float defaultDensity, bool next)
{
	if (InvalidCoords(x, y)) return defaultDensity;
	if (next) return GetPixelData(x, y)->nextDensity;
	return GetPixelData(x, y)->density;
}
__device__ Vec2f GetVelocity(int x, int y, Vec2f defaultVelocity, bool next, int axis)
{
	if (InvalidCoords(x, y))
	{
		if (axis == 0) return { -defaultVelocity.x, defaultVelocity.y };
		else if(axis == 1) return { defaultVelocity.x, -defaultVelocity.y };
		else return { -defaultVelocity.x, -defaultVelocity.y };
	}
	if (next) return GetPixelData(x, y)->nextVelocity;
	return GetPixelData(x, y)->velocity;
}
__device__ float GetP(int x, int y, float defaultP)
{
	if (InvalidCoords(x, y)) return defaultP;
	return GetPixelData(x, y)->p;
}

__global__ void HandleInput(MouseInput input)
{
	if (input.lmbDown)
		for (int xOffset = -INPUT_RADIUS; xOffset <= INPUT_RADIUS; xOffset++)
		{
			for (int yOffset = -INPUT_RADIUS; yOffset <= INPUT_RADIUS; yOffset++)
			{
				if (InvalidCoords(input.position.x + xOffset, input.position.y + yOffset)) continue;
				Vec2f vel = GetPixelData(input.position.x + xOffset, input.position.y + yOffset)->velocity + input.velocity * 10;
				GetPixelData(input.position.x + xOffset, input.position.y + yOffset)->velocity = vel;
				GetPixelData(input.position.x + xOffset, input.position.y + yOffset)->nextVelocity = vel;
				GetPixelData(input.position.x + xOffset, input.position.y + yOffset)->nextVelocityGS = vel;
			}
		}
	else if (input.rmbDown)
		for (int xOffset = -INPUT_RADIUS; xOffset <= INPUT_RADIUS; xOffset++)
		{
			for (int yOffset = -INPUT_RADIUS; yOffset <= INPUT_RADIUS; yOffset++)
			{
				if (InvalidCoords(input.position.x + xOffset, input.position.y + yOffset)) continue;
				GetPixelData(input.position.x + xOffset, input.position.y + yOffset)->density = 10;
			}
		}
}

__device__ void PixelDiffusePass(int x, int y, double t)
{
	float defaultDensity = GetDensity(x, y, 0, true);
	Vec2f defaultVelocity = GetVelocity(x, y, { 0, 0 }, true, 0);

	double k = t * 0.1;
	float s_n = GetDensity(x - 1, y, defaultDensity, true) * 0.25f + GetDensity(x + 1, y, defaultDensity, true) * 0.25f +
		GetDensity(x, y - 1, defaultDensity, true) * 0.25f + GetDensity(x, y + 1, defaultDensity, true) * 0.25f;

	float d_c = GetDensity(x, y, 0, false);
	float d_n = (d_c + k * s_n) / (1 + k);
	GetPixelData(x, y)->nextDensityGS = d_n;

	Vec2f vels_n = GetVelocity(x - 1, y, defaultVelocity, true, 0) * 0.25f + GetVelocity(x + 1, y, defaultVelocity, true, 0) * 0.25f +
		GetVelocity(x, y - 1, defaultVelocity, true, 1) * 0.25f + GetVelocity(x, y + 1, defaultVelocity, true, 1) * 0.25f;
	Vec2f veld_c = GetVelocity(x, y, { 0, 0 }, false, 0);
	Vec2f veld_n = (veld_c + vels_n * k) * (1.0f / (1 + k));
	GetPixelData(x, y)->nextVelocityGS = veld_n;

}
__global__ void TextureDiffusePass(double t)
{
	uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
	uint32_t stride = blockDim.x * gridDim.x;

	for (int pix = index; pix < w * h; pix += stride)
	{
		int x = pix % w;
		int y = pix / w;
		PixelDiffusePass(x, y, t);
	}
}

__device__ void PixelAdvectPass(int x, int y, double t)
{
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

	data->nextDensityGS = TL * GetDensity(lastXInt, lastYInt, data->density, false) +
		TR * GetDensity(lastXInt + 1, lastYInt, data->density, false) +
		BL * GetDensity(lastXInt, lastYInt + 1, data->density, false) +
		BR * GetDensity(lastXInt + 1, lastYInt + 1, data->density, false);

	data->nextVelocityGS = GetVelocity(lastXInt, lastYInt, data->velocity, false, 0) * TL +
		GetVelocity(lastXInt + 1, lastYInt, data->velocity, false, 0) * TR +
		GetVelocity(lastXInt, lastYInt + 1, data->velocity, false, 1) * BL +
		GetVelocity(lastXInt + 1, lastYInt + 1, data->velocity, false, 2) * BR;
}
__global__ void TextureAdvectPass(double t)
{
	uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
	uint32_t stride = blockDim.x * gridDim.x;

	for (int pix = index; pix < w * h; pix += stride)
	{
		int x = pix % w;
		int y = pix / w;
		PixelAdvectPass(x, y, t);
	}
}

__device__ void PixelPValPass(int x, int y, double t)
{
	Vec2f defaultVelocity = GetVelocity(x, y, { 0, 0 }, true, 0);

	float velsum = ((GetVelocity(x + 1, y, defaultVelocity, true, 0).x - GetVelocity(x - 1, y, defaultVelocity, true, 0).x) +
		(GetVelocity(x, y + 1, defaultVelocity, true, 1).y - GetVelocity(x, y - 1, defaultVelocity, true, 1).y)) * 0.5f;

	GetPixelData(x, y)->pGS = (GetP(x - 1, y, 0) + GetP(x + 1, y, 0) + GetP(x, y - 1, 0) + GetP(x, y + 1, 0) - velsum) * 0.25f;
}
__global__ void TexturePValPass(double t)
{
	uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
	uint32_t stride = blockDim.x * gridDim.x;

	for (int pix = index; pix < w * h; pix += stride)
	{
		int x = pix % w;
		int y = pix / w;
		PixelPValPass(x, y, t);
	}
}

__device__ void PixelDivergencePass(int x, int y, double t)
{
	Vec2f gradP((GetP(x + 1, y, 0) - GetP(x - 1, y, 0)) / 2, (GetP(x, y + 1, 0) - GetP(x, y - 1, 0)) / 2);
	GetPixelData(x, y)->nextVelocityGS = GetPixelData(x, y)->velocity + (gradP * -1);
}
__global__ void TextureDivergencePass(double t)
{
	uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
	uint32_t stride = blockDim.x * gridDim.x;

	for (int pix = index; pix < w * h; pix += stride)
	{
		int x = pix % w;
		int y = pix / w;
		PixelDivergencePass(x, y, t);
	}
}

__global__ void CopyValues(bool finalCopy)
{
	uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
	uint32_t stride = blockDim.x * gridDim.x;

	for (int pix = index; pix < w * h; pix += stride)
	{
		int x = pix % w;
		int y = pix / w;
		GetPixelData(x, y)->nextDensity = GetPixelData(x, y)->nextDensityGS;
		if (finalCopy)
			GetPixelData(x, y)->density = GetPixelData(x, y)->nextDensity;
		GetPixelData(x, y)->nextVelocity = GetPixelData(x, y)->nextVelocityGS;
		if (finalCopy)
			GetPixelData(x, y)->velocity = GetPixelData(x, y)->nextVelocity;
	}
}
__global__ void CopyPVals()
{
	uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
	uint32_t stride = blockDim.x * gridDim.x;

	for (int pix = index; pix < w * h; pix += stride)
	{
		int x = pix % w;
		int y = pix / w;
		GetPixelData(x, y)->p = GetPixelData(x, y)->pGS;
	}
}

__device__ float sigmoid(float x)
{
	return 1 / (1 + powf(2.71828f, -x));
}
__global__ void DrawFluid()
{
	uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
	uint32_t stride = blockDim.x * gridDim.x;

	for (int pix = index; pix < w * h; pix += stride)
	{
		int x = pix % w;
		int y = pix / w;

		float sigDensity = sigmoid(GetDensity(x, y, 0, false) * 2);
		float sigVelX = sigmoid(GetVelocity(x, y, { 0,0 }, false, 0).x / 10);
		float sigVelY = sigmoid(GetVelocity(x, y, { 0,0 }, false, 0).y / 10);

		RGBA color(sigDensity * 255, sigVelX * 255, sigVelY * 255);
		SetPixel(color, x, y);
	}
}

__global__ void InitTexture(uint64_t seed)
{
	uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
	uint32_t stride = blockDim.x * gridDim.x;

	for (int pix = index * 8; pix < w * h; pix += stride * 8)
	{
		int x = pix % w;
		int y = pix / w;
		Xoshiro256ssc rnd(seed);
		rnd.SetRandomSeed(x, y);

		for (int i = 0; i < 8; i++)
		{
			int x2 = (pix + i) % w;
			int y2 = (pix + i) / w;
			PixelData* data = GetPixelData(x2, y2);
			float density = rnd.Next() / 10;
			data->density = density;
			data->nextDensity = density;
			data->velocity = { rnd.Next() * 10 - 5, rnd.Next() * 10 - 5};
		}
	}
}

void texInit(const int _width, const int _height)
{
	int width = _width / DOWNSCALE;
	int height = _height / DOWNSCALE;

	hPixels = (RGBA*)malloc(width * height * 4);
	checkCudaErrors(cudaMalloc(&dPixels, width * height * 4));
	checkCudaErrors(cudaMalloc(&dPixelData, width * height * sizeof(PixelData)));
	checkCudaErrors(cudaMemcpyToSymbol(texture, &dPixels, sizeof(RGBA*)));
	checkCudaErrors(cudaMemcpyToSymbol(pixelData, &dPixelData, sizeof(PixelData*)));

	checkCudaErrors(cudaMemcpyToSymbol(w, &width, sizeof(int)));
	checkCudaErrors(cudaMemcpyToSymbol(h, &height, sizeof(int)));

	InitTexture << <NUMBLOCKS, BLOCKSIZE >> > (1234);
	checkCudaErrors(cudaDeviceSynchronize());

	tex = Texture();
	tex.create(width, height);
}

Vec2f lastMPos = { 0, 0 };

void texRender(RenderWindow& window, const int _width, const int _height, const float deltaT)
{
	int width = _width / DOWNSCALE;
	int height = _height / DOWNSCALE;

	Vec2f currentMPos(Mouse::getPosition().x / DOWNSCALE, Mouse::getPosition().y / DOWNSCALE);
	Vec2f diff = currentMPos + (lastMPos * -1);
	lastMPos = currentMPos;
	MouseInput mouseInput = { Mouse::isButtonPressed(Mouse::Left), Mouse::isButtonPressed(Mouse::Right), currentMPos, diff };
	HandleInput << <1, 1 >> > (mouseInput);

	bool paused = Keyboard::isKeyPressed(Keyboard::Space);

	if (!paused)
	{
		for (int gLoops = 0; gLoops < LOOPCOUNT; gLoops++)
		{
			hIters++;
			for (int loops = 0; loops < 4; loops++)
			{
				TextureDiffusePass << <NUMBLOCKS, BLOCKSIZE >> > (deltaT);
				checkCudaErrors(cudaDeviceSynchronize());
				CopyValues << <NUMBLOCKS, BLOCKSIZE >> > (false);
				checkCudaErrors(cudaDeviceSynchronize());
			}
			CopyValues << <NUMBLOCKS, BLOCKSIZE >> > (true);
			checkCudaErrors(cudaDeviceSynchronize());

			TextureAdvectPass << <NUMBLOCKS, BLOCKSIZE >> > (deltaT);
			checkCudaErrors(cudaDeviceSynchronize());
			CopyValues << <NUMBLOCKS, BLOCKSIZE >> > (true);
			checkCudaErrors(cudaDeviceSynchronize());
			
			for (int loops = 0; loops < 4; loops++)
			{
				TexturePValPass << <NUMBLOCKS, BLOCKSIZE >> > (deltaT);
				checkCudaErrors(cudaDeviceSynchronize());
				CopyPVals << <NUMBLOCKS, BLOCKSIZE >> > ();
				checkCudaErrors(cudaDeviceSynchronize());
			}

			TextureDivergencePass << <NUMBLOCKS, BLOCKSIZE >> > (deltaT);
			checkCudaErrors(cudaDeviceSynchronize());
			CopyValues << <NUMBLOCKS, BLOCKSIZE >> > (true);
			checkCudaErrors(cudaDeviceSynchronize());
		}
	}
	DrawFluid << <NUMBLOCKS, BLOCKSIZE >> > ();
	checkCudaErrors(cudaDeviceSynchronize());

	checkCudaErrors(cudaMemcpy(hPixels, dPixels, 4 * width * height, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaDeviceSynchronize());

	//Drawing
	tex.update((uint8_t*)hPixels);
	Sprite sprite = Sprite(tex);
	sprite.setScale(Vector2f(DOWNSCALE, DOWNSCALE));
	window.draw(sprite);

	//Chunk Updater 9000
	/*if (Keyboard::isKeyPressed(Keyboard::Tab))
	{
		int loadedChunks = 0;
		checkCudaErrors(cudaMemcpy(hChunkData, dChunkData, (width / CHUNKDIM + 1) * (height / CHUNKDIM + 1) * sizeof(ChunkData), cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaDeviceSynchronize());
		for (int chunkIdx = 0; chunkIdx < (width / CHUNKDIM + 1) * (height / CHUNKDIM + 1); chunkIdx++)
		{
			ChunkData dat = hChunkData[chunkIdx];

			if (dat.updated)
				loadedChunks++;

			int cx = chunkIdx % (width / CHUNKDIM + 1);
			int cy = chunkIdx / (width / CHUNKDIM + 1);

			RectangleShape shape(Vector2f(CHUNKDIM * DOWNSCALE, CHUNKDIM * DOWNSCALE));
			shape.setFillColor(Color(0, 0, 0, 0));
			shape.setOutlineThickness(-1);

			shape.setOutlineColor(Color(255, 255, 0));

			shape.setPosition(Vector2f(cx * CHUNKDIM * DOWNSCALE, cy * CHUNKDIM * DOWNSCALE));
			window.draw(shape);
		}

		RectangleShape bgRect(Vector2f(300, 200));
		bgRect.setFillColor(Color(0, 0, 0, 128));
		window.draw(bgRect);

		Font arial;
		arial.loadFromFile("arial.ttf");
		Text chunkCounter;
		chunkCounter.setFont(arial);
		chunkCounter.setCharacterSize(18);
		chunkCounter.setFillColor(Color::White);

		char buffer[30];
		sprintf(buffer, "%i/%i chunks loaded", loadedChunks, (width / CHUNKDIM + 1) * (height / CHUNKDIM + 1));
		chunkCounter.setString(buffer);
		chunkCounter.setPosition(Vector2f(0, 30));
		window.draw(chunkCounter);

	}*/
}

void texCleanup()
{
	cudaFree(dPixels);
	cudaFree(dPixelData);
	free(hPixels);
}