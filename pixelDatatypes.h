#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "Vector.h"

#include <stdint.h>

enum Material : uint8_t
{
	AIR,
	SAND,
	WALL
};

struct RGBA
{
	uint8_t r = 0;
	uint8_t g = 0;
	uint8_t b = 0;
	uint8_t a = 255;

	__device__ constexpr RGBA()
	{
		r = 0;
		g = 0;
		b = 0;
		a = 255;
	}

	__device__ constexpr RGBA(uint8_t _r, uint8_t _g, uint8_t _b)
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

struct PixelState
{
	RGBA color = RGBA();

	Vec2f exactPos = { 0, 0 };
	Vec2f velocity = { 0, 0 };

	__device__ PixelState(const PixelState& copy)
	{
		memcpy(this, &copy, sizeof(PixelState));
	}
	__device__ constexpr PixelState() {}
	__device__ constexpr PixelState(RGBA _c)
	{
		color = _c;
	}
};

struct ChunkData
{
	bool lastUpdated;
	bool updated;
	bool reactionUpdated;
};

struct MaterialInfo
{
	RGBA meanColor;
};

__device__ __constant__ RGBA* pixelColors;
__device__ __constant__ Material* mats;
__device__ __constant__ PixelState* pixelStates;
__device__ __constant__ ChunkData* chunkData;
__device__ __constant__ bool* pixelsUpdated;

__device__ __constant__ int w;
__device__ __constant__ int h;

__device__ int dIters = 0;
int hIters = 0;

RGBA* hPixels;
RGBA* dPixels;

Material* dMaterials;
PixelState* dPixelData;
ChunkData* dChunkData;
bool* dPixelsUpdated;

ChunkData* hChunkData;

constexpr auto NUMBLOCKS = 256;
constexpr auto BLOCKSIZE = 128;

constexpr auto CHUNKDIM = 8;
constexpr auto REACTION_RATE = 8;

constexpr auto DOWNSCALE = 1.0f;
constexpr auto LOOPCOUNT = 1;

constexpr auto INPUT_RADIUS = 0;