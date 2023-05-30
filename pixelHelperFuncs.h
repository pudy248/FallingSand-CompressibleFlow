#pragma once

#include "pixelDatatypes.h"
#include "xoshiro.h"

#include <cmath>

__device__ bool InvalidCoords(int x, int y)
{
	return x < 0 || x >= w || y < 0 || y >= h;
}

__device__ void TouchChunk(int x, int y)
{
	auto TouchChunkHelper = [](int _x, int _y)
	{
		int _chunkIdx = (_y / CHUNKDIM) * (w / CHUNKDIM + 1) + (_x / CHUNKDIM);
		chunkData[_chunkIdx].updated = true;
		return _chunkIdx;
	};

	int chunkIdx = TouchChunkHelper(x, y);

	int cx = chunkIdx % (w / CHUNKDIM + 1) * CHUNKDIM;
	int cy = chunkIdx / (w / CHUNKDIM + 1) * CHUNKDIM;

	int xDiff = x - cx;
	int yDiff = y - cy;

	if (xDiff == 0) TouchChunkHelper(x - 1, y);
	else if (xDiff == CHUNKDIM - 1) TouchChunkHelper(x + 1, y);
	if (yDiff == 0) TouchChunkHelper(x, y - 1);
	else if (yDiff == CHUNKDIM - 1) TouchChunkHelper(x, y + 1);
}

__device__ void TouchReactionChunk(int x, int y)
{
	auto TouchChunkHelper = [](int _x, int _y)
	{
		int _chunkIdx = (_y / CHUNKDIM) * (w / CHUNKDIM + 1) + (_x / CHUNKDIM);
		chunkData[_chunkIdx].reactionUpdated = true;
		return _chunkIdx;
	};

	int chunkIdx = TouchChunkHelper(x, y);

	int cx = chunkIdx % (w / CHUNKDIM + 1) * CHUNKDIM;
	int cy = chunkIdx / (w / CHUNKDIM + 1) * CHUNKDIM;

	int xDiff = x - cx;
	int yDiff = y - cy;

	if (xDiff == 0) TouchChunkHelper(x - 1, y);
	else if (xDiff == CHUNKDIM - 1) TouchChunkHelper(x + 1, y);
	if (yDiff == 0) TouchChunkHelper(x, y - 1);
	else if (yDiff == CHUNKDIM - 1) TouchChunkHelper(x, y + 1);
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
	TouchChunk(x, y);
}

__device__ void SetPixelData(PixelData data, int x, int y)
{
	pixelData[y * w + x] = data;
	TouchChunk(x, y);
}

__device__ bool PixelUpdated(int x, int y)
{
	return GetPixelData(x, y)->updated;
}

__device__ void SwapPixels(int x1, int y1, int x2, int y2)
{
	if (InvalidCoords(x1, y1) || InvalidCoords(x2, y2)) return;
	RGBA colorTemp = GetPixel(x1, y1);
	SetPixel(GetPixel(x2, y2), x1, y1);
	SetPixel(colorTemp, x2, y2);

	PixelData dataTemp = *GetPixelData(x1, y1);
	*GetPixelData(x1, y1) = *GetPixelData(x2, y2);
	*GetPixelData(x2, y2) = dataTemp;
}

__device__ RGBA randomizeColor(RGBA base, const int variation, uint seed)
{
	uint PCG_state = seed;
	PCG_state = PCG32_hash(PCG_state);
	int r = base.r + (PCG_state % (2 * variation)) - variation;
	PCG_state = PCG32_hash(PCG_state);
	int g = base.g + (PCG_state % (2 * variation)) - variation;
	PCG_state = PCG32_hash(PCG_state);
	int b = base.b + (PCG_state % (2 * variation)) - variation;
	r = min(max(0, r), 255);
	g = min(max(0, g), 255);
	b = min(max(0, b), 255);
	return RGBA(r, g, b);
}

__device__ void SetPixelFromPrefab(PixelPrefab prefab, int x, int y)
{
	uint pixelIdx = y * w + x;
	uint PCG_state = PCG32_hash(PCG32_hash(pixelIdx) * PCG32_hash(dIters) + 37);
	RGBA color = randomizeColor(prefab.defaultColor, 16, PCG_state);
	SetPixel(color, x, y);
	SetPixelData(prefab.defaultData, x, y);
	TouchReactionChunk(x, y);
}



__device__ bool TrySwapDown(int x, int y, PixelData data)
{
	bool bottomOpen = false;
	if (!InvalidCoords(x, y + 1)) bottomOpen = GetPixelData(x, y + 1)->density < data.density;

	if (bottomOpen)
	{
		SwapPixels(x, y, x, y + 1);
		return true;
	}
	return false;
}

__device__ bool TrySwapDownDiag(int x, int y, PixelData data)
{
	bool downLeftOpen = false;
	if (!InvalidCoords(x - 1, y + 1)) downLeftOpen = GetPixelData(x - 1, y + 1)->density < data.density;
	bool downRightOpen = false;
	if (!InvalidCoords(x + 1, y + 1)) downRightOpen = GetPixelData(x + 1, y + 1)->density < data.density;

	if (downLeftOpen && downRightOpen)
	{
		uint pixelIdx = y * w + x;
		uint PCG = PCG32_hash(PCG32_hash(pixelIdx) + dIters);
		if (toFloat01(PCG) < 0.5f) downLeftOpen = false;
		else downRightOpen = false;
	}

	if (downLeftOpen)
	{
		SwapPixels(x, y, x - 1, y + 1);
		return true;
	}

	if (downRightOpen)
	{
		SwapPixels(x, y, x + 1, y + 1);
		return true;
	}
	return false;
}

__device__ bool TrySwapDownVel(int x, int y, int velocity, PixelData data)
{
	int downVel = 0;
	bool downOpen;
	do
	{
		downVel++;
		downOpen = false;
		if (!InvalidCoords(x, y + downVel)) downOpen = GetPixelData(x, y + downVel)->density < data.density;
	} while (downOpen && downVel <= velocity);

	downVel--;
	downOpen = false;
	if (!InvalidCoords(x, y + downVel)) downOpen = GetPixelData(x, y + downVel)->density < data.density;

	if (downOpen)
	{
		SwapPixels(x, y, x, y + downVel);
		return true;
	}

	return false;
}

__device__ bool TrySwapHorizontal(int x, int y, int velocity, PixelData data)
{
	int leftVel = 0;
	bool leftOpen;
	do
	{
		leftVel++;
		leftOpen = false;
		if (!InvalidCoords(x - leftVel, y)) leftOpen = GetPixelData(x - leftVel, y)->density < data.density;
	} while (leftOpen && leftVel <= velocity);

	leftVel--;
	leftOpen = false;
	if (!InvalidCoords(x - leftVel, y)) leftOpen = GetPixelData(x - leftVel, y)->density < data.density;


	int rightVel = 0;
	bool rightOpen;
	do
	{
		rightVel++;
		rightOpen = false;
		if (!InvalidCoords(x + rightVel, y)) rightOpen = GetPixelData(x + rightVel, y)->density < data.density;
	} while (rightOpen && rightVel <= velocity);

	rightVel--;
	rightOpen = false;
	if (!InvalidCoords(x + rightVel, y)) rightOpen = GetPixelData(x + rightVel, y)->density < data.density;


	if (leftOpen && rightOpen)
	{
		uint pixelIdx = y * w + x;
		uint PCG = PCG32_hash(PCG32_hash(pixelIdx) + dIters);
		if (toFloat01(PCG) < 0.5f) leftOpen = false;
		else rightOpen = false;
	}

	if (leftOpen)
	{
		SwapPixels(x, y, x - leftVel, y);
		return true;
	}

	if (rightOpen)
	{
		SwapPixels(x, y, x + rightVel, y);
		return true;
	}
	return false;
}

__device__ bool TrySwapUpDiag(int x, int y, PixelData data)
{
	bool upLeftOpen = false;
	if (!InvalidCoords(x - 1, y - 1)) upLeftOpen = GetPixelData(x - 1, y - 1)->density < data.density;
	bool upRightOpen = false;
	if (!InvalidCoords(x + 1, y - 1)) upRightOpen = GetPixelData(x + 1, y - 1)->density < data.density;

	if (upLeftOpen && upRightOpen)
	{
		uint pixelIdx = y * w + x;
		uint PCG = PCG32_hash(PCG32_hash(pixelIdx) + dIters);
		if (toFloat01(PCG) < 0.5f) upLeftOpen = false;
		else upRightOpen = false;
	}

	if (upLeftOpen)
	{
		SwapPixels(x, y, x - 1, y - 1);
		return true;
	}

	if (upRightOpen)
	{
		SwapPixels(x, y, x + 1, y - 1);
		return true;
	}
	return false;
}

__device__ bool TrySwapUp(int x, int y, PixelData data)
{
	bool topOpen = false;
	if (!InvalidCoords(x, y - 1)) topOpen = GetPixelData(x, y - 1)->density < data.density;

	if (topOpen)
	{
		SwapPixels(x, y, x, y - 1);
		return true;
	}
	return false;
}
