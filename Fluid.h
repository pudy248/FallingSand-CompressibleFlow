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

struct PixelData
{
	float density[2] = { 0, 0 };
	float pressure[2] = { 0, 0 };
	Vec2f velocity[2] = { {0, 0}, {0, 0} };

	__host__ __device__ constexpr PixelData()
	{

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

constexpr auto DOWNSCALE = 1;
constexpr auto LOOPCOUNT = 2;

constexpr auto INPUT_RADIUS = 5;

constexpr bool LOOP_X = true;
constexpr bool LOOP_Y = false;

Texture tex;

__device__ bool InvalidCoords(int x, int y)
{
	return (!LOOP_X && (x < 0 || x >= w)) || (!LOOP_Y && (y < 0 || y >= h));
}
__device__ void ModularizeCoords(int& x, int& y) {
	if (LOOP_X) x = ((x % w) + w) % w;
	if (LOOP_Y) y = ((y % h) + h) % h;
}

__device__ PixelData* GetPixelData(int x, int y)
{
	return pixelData + (y * w + x);
}

__device__ RGBA GetPixelColor(int x, int y)
{
	return texture[y * w + x];
}
__device__ void SetPixelColor(RGBA c, int x, int y)
{
	texture[y * w + x] = c;
}
__device__ void SetPixelData(PixelData data, int x, int y)
{
	pixelData[y * w + x] = data;
}

__device__ float GetDensity(int x, int y, float oob)
{
	ModularizeCoords(x, y);
	if (InvalidCoords(x, y)) return oob;
	return GetPixelData(x, y)->density[0];
}
__device__ Vec2f GetVelocity(int x, int y, Vec2f oob, int axis)
{
	ModularizeCoords(x, y);
	if (InvalidCoords(x, y))
	{
		if (axis == 0) return { -oob.x, oob.y };
		else if(axis == 1) return { oob.x, -oob.y };
		else return { -oob.x, -oob.y };
	}
	return GetPixelData(x, y)->velocity[0];
}
__device__ float GetPressure(int x, int y, float oob)
{
	ModularizeCoords(x, y);
	if (InvalidCoords(x, y)) return oob;
	return GetPixelData(x, y)->pressure[0];
}

__global__ void HandleInput(MouseInput input)
{
	if (input.lmbDown)
		for (int xOffset = -INPUT_RADIUS; xOffset <= INPUT_RADIUS; xOffset++)
		{
			for (int yOffset = -INPUT_RADIUS; yOffset <= INPUT_RADIUS; yOffset++)
			{
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
				GetPixelData(input.position.x + xOffset, input.position.y + yOffset)->pressure[0] += 1;
			}
		}
}

/*
__device__ void PixelDiffusePass(int x, int y, double t)
{
	float defaultDensity = GetDensity(x, y, 0);
	Vec2f defaultVelocity = GetVelocity(x, y, { 0, 0 }, 0);

	double k = 0.1;
	float s_n = (GetDensity(x - 1, y, defaultDensity) + GetDensity(x + 1, y, defaultDensity) +
				 GetDensity(x, y - 1, defaultDensity) + GetDensity(x, y + 1, defaultDensity)) * 0.25f;

	float d_c = GetDensity(x, y, 0);
	float d_n = (d_c + k * s_n) / (1 + k);
	GetPixelData(x, y)->density[1] = d_n;

	Vec2f vels_n = (GetVelocity(x - 1, y, defaultVelocity, 0) + GetVelocity(x + 1, y, defaultVelocity, 0) +
					GetVelocity(x, y - 1, defaultVelocity, 1) + GetVelocity(x, y + 1, defaultVelocity, 1)) * 0.25f;
	Vec2f veld_c = GetVelocity(x, y, { 0, 0 }, 0);
	Vec2f veld_n = (veld_c + vels_n * k) * (1.0f / (1 + k));
	GetPixelData(x, y)->velocity[1] = veld_n;

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
*/

__device__ void PixelAdvectPass(int x, int y, double t)
{
	PixelData* data = GetPixelData(x, y);
	float lastX = x - data->velocity[0].x * t;
	float lastY = y - data->velocity[0].y * t;

	int lastXInt = floorf(lastX);
	int lastYInt = floorf(lastY);

	float lastXFrac = lastX - lastXInt;
	float lastYFrac = lastY - lastYInt;

	float TL = (1 - lastXFrac) * (1 - lastYFrac);
	float TR = lastXFrac * (1 - lastYFrac);
	float BL = (1 - lastXFrac) * lastYFrac;
	float BR = lastXFrac * lastYFrac;

	data->density[1] = TL * GetDensity(lastXInt, lastYInt, data->density[0]) +
		TR * GetDensity(lastXInt + 1, lastYInt, data->density[0]) +
		BL * GetDensity(lastXInt, lastYInt + 1, data->density[0]) +
		BR * GetDensity(lastXInt + 1, lastYInt + 1, data->density[0]);

	data->velocity[1] = GetVelocity(lastXInt, lastYInt, data->velocity[0], 0) * TL +
		GetVelocity(lastXInt + 1, lastYInt, data->velocity[0], 0) * TR +
		GetVelocity(lastXInt, lastYInt + 1, data->velocity[0], 1) * BL +
		GetVelocity(lastXInt + 1, lastYInt + 1, data->velocity[0], 2) * BR;

	data->pressure[1] = data->density[1];
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

__device__ void PixelDivergencePass(int x, int y, double t)
{
	Vec2f defaultVelocity = GetVelocity(x, y, { 0, 0 }, 0);

	float velGradient = ((GetVelocity(x + 1, y, defaultVelocity, 0).x - GetVelocity(x - 1, y, defaultVelocity, 0).x) +
		(GetVelocity(x, y + 1, defaultVelocity, 1).y - GetVelocity(x, y - 1, defaultVelocity, 1).y)) * 0.5f;

	GetPixelData(x, y)->pressure[1] = (GetPressure(x - 1, y, 0) + GetPressure(x + 1, y, 0) + GetPressure(x, y - 1, 0) + GetPressure(x, y + 1, 0) - velGradient) * 0.25f;
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

__device__ void PixelUpdateVelocity(int x, int y, double t)
{
	Vec2f gradP((GetPressure(x + 1, y, 0) - GetPressure(x - 1, y, 0)) / 2, (GetPressure(x, y + 1, 0) - GetPressure(x, y - 1, 0)) / 2);
	GetPixelData(x, y)->velocity[1] = GetPixelData(x, y)->velocity[0] + (gradP * -1);
}
__global__ void TextureUpdateVelocity(double t)
{
	uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
	uint32_t stride = blockDim.x * gridDim.x;

	for (int pix = index; pix < w * h; pix += stride)
	{
		int x = pix % w;
		int y = pix / w;
		PixelUpdateVelocity(x, y, t);
	}
}

__global__ void CopyValues()
{
	uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
	uint32_t stride = blockDim.x * gridDim.x;

	for (int pix = index; pix < w * h; pix += stride)
	{
		int x = pix % w;
		int y = pix / w;
		GetPixelData(x, y)->density[0] = GetPixelData(x, y)->density[1];
		GetPixelData(x, y)->pressure[0] = GetPixelData(x, y)->pressure[1];
		GetPixelData(x, y)->velocity[0] = GetPixelData(x, y)->velocity[1];
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

		float velAngle = GetVelocity(x, y, { 0,0 }, 0).angle() / (2 * 3.1415926f);
		float velMagn = fminf(1, GetVelocity(x, y, { 0,0 }, 0).magnitude() / 120.f);
		UniversalColor tmp = { velAngle, 1, 0 };
		UniversalColor rgb = hsl2rgb(tmp);
		rgb.r *= velMagn;
		rgb.g *= velMagn;
		rgb.b *= velMagn;

		float sigDensity = sigmoid(GetDensity(x, y, 0) * 2);

		RGBA color(rgb.r, rgb.g, rgb.b);
		SetPixelColor(color, x, y);
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
			data->density[0] = data->density[1] = rnd.Next();
			data->pressure[0] = data->pressure[1] = rnd.Next();
			if (abs(y - h / 2) < h / 3)
				data->velocity[0] = {100, rnd.Next() / 10 - 0.5f};
			else
				data->velocity[0] = {-100, rnd.Next() / 10 - 0.5f};
			//data->velocity = { rnd.Next() * 10 - 5, rnd.Next() * 10 - 5};
		}
	}
}

void texInit(const int _width, const int _height)
{
	int width = _width / DOWNSCALE;
	int height = _height / DOWNSCALE;
	int z = 0;

	hPixels = (RGBA*)malloc(width * height * 4);
	checkCudaErrors(cudaMalloc(&dPixels, width * height * 4));
	checkCudaErrors(cudaMalloc(&dPixelData, width * height * sizeof(PixelData)));
	checkCudaErrors(cudaMemcpyToSymbol(texture, &dPixels, sizeof(RGBA*)));
	checkCudaErrors(cudaMemcpyToSymbol(pixelData, &dPixelData, sizeof(PixelData*)));

	checkCudaErrors(cudaMemcpyToSymbol(w, &width, sizeof(int)));
	checkCudaErrors(cudaMemcpyToSymbol(h, &height, sizeof(int)));
	checkCudaErrors(cudaMemcpyToSymbol(dIters, &z, sizeof(int)));
	hIters = 0;

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
			//TextureDiffusePass << <NUMBLOCKS, BLOCKSIZE >> > (deltaT);
			//checkCudaErrors(cudaDeviceSynchronize());
			//CopyValues << <NUMBLOCKS, BLOCKSIZE >> > ();
			//checkCudaErrors(cudaDeviceSynchronize());
			TextureAdvectPass << <NUMBLOCKS, BLOCKSIZE >> > (deltaT);
			checkCudaErrors(cudaDeviceSynchronize());
			CopyValues << <NUMBLOCKS, BLOCKSIZE >> > ();
			checkCudaErrors(cudaDeviceSynchronize());
			TextureDivergencePass << <NUMBLOCKS, BLOCKSIZE >> > (deltaT);
			checkCudaErrors(cudaDeviceSynchronize());
			CopyValues << <NUMBLOCKS, BLOCKSIZE >> > ();
			checkCudaErrors(cudaDeviceSynchronize());
			TextureUpdateVelocity << <NUMBLOCKS, BLOCKSIZE >> > (deltaT);
			checkCudaErrors(cudaDeviceSynchronize());
			CopyValues << <NUMBLOCKS, BLOCKSIZE >> > ();
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