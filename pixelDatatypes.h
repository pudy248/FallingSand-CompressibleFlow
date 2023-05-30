#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

struct PixelData
{
	uint8_t id = 0;
	bool updated = false;
	uint8_t density = 0;

	int lifetime = -1;

	bool flammable = false;

	__host__ __device__ constexpr PixelData()
	{
		id = 0;
		updated = false;
		density = 0;
	}

	__host__ __device__ constexpr PixelData(const uint8_t _id)
	{
		id = _id;
		updated = false;
		density = 0;
	}

	__host__ __device__ constexpr PixelData(const uint8_t _id, const uint8_t _density)
	{
		id = _id;
		updated = false;
		density = _density;
	}


	__host__ __device__ constexpr PixelData(const uint8_t _id, const uint8_t _density, const bool _flammable)
	{
		id = _id;
		updated = false;
		density = _density;
		flammable = _flammable;
	}

	__host__ __device__ constexpr PixelData(const uint8_t _id, const uint8_t _density, const bool _flammable, const int _lifetime)
	{
		id = _id;
		updated = false;
		density = _density;
		flammable = _flammable;
		lifetime = _lifetime;
	}
};

struct ChunkData
{
	bool lastUpdated;
	bool updated;
	bool reactionUpdated;
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

struct PixelPrefab
{
	PixelData defaultData;
	RGBA defaultColor;
};

__device__ PixelPrefab AIR = { PixelData(0), RGBA(0, 0, 0) };
__device__ PixelPrefab SAND = { PixelData(1, 10), RGBA(0xFF, 0xDF, 0x7F) };
__device__ PixelPrefab WATER = { PixelData(2, 5), RGBA(0x00, 0x00, 0xFF) };
__device__ PixelPrefab SNOW = { PixelData(3, 7), RGBA(0xFF, 0xFF, 0xFF) };
__device__ PixelPrefab SMOKE = { PixelData(4, 1, false, 200), RGBA(0x5F, 0x5F, 0x5F) };
__device__ PixelPrefab OIL = { PixelData(5, 2, true), RGBA(0x7F, 0x6F, 0x00) };
__device__ PixelPrefab FIRE = { PixelData(6, 3, false, 10), RGBA(0xDF, 0x7F, 0x3F) };

__device__ __constant__ RGBA* texture;
__device__ __constant__ PixelData* pixelData;
__device__ __constant__ ChunkData* chunkData;

__device__ __constant__ int w;
__device__ __constant__ int h;

__device__ int dIters = 0;
int hIters = 0;

RGBA* hPixels;
RGBA* dPixels;

PixelData* dPixelData;
ChunkData* dChunkData;

ChunkData* hChunkData;

constexpr auto NUMBLOCKS = 256;
constexpr auto BLOCKSIZE = 64;

constexpr auto CHUNKDIM = 16;
constexpr auto REACTION_RATE = 8;

constexpr auto DOWNSCALE = 1;
constexpr auto LOOPCOUNT = 5;

constexpr auto INPUT_RADIUS = 3;