#pragma once

#include "PixelHelperFuncs.h"


__device__ void FallingSandUpdate(int x, int y, int t)
{
	if (InvalidCoords(x, y)) return;
	if (PixelUpdated(x, y)) return;

	uint8_t id = GetPixelData(x, y)->id;
	if (id == 0) return;

	RGBA currentPix = GetPixel(x, y);
	GetPixelData(x, y)->updated = true;

	PixelData data = *GetPixelData(x, y);
	uint pixelIdx = y * w + x;
	uint PCG_state = PCG32_hash(PCG32_hash(pixelIdx) + PCG32_hash(dIters));

	switch (data.id)
	{
	case 1:
		if (TrySwapDownVel(x, y, 3, data)) return;
		if (TrySwapDownDiag(x, y, data)) return;
		return;
	case 2:
		if (TrySwapDownVel(x, y, 3, data)) return;
		if (TrySwapDownDiag(x, y, data)) return;
		if (TrySwapHorizontal(x, y, 3, data)) return;
		return;
	case 3:
		if (TrySwapDownDiag(x, y, data)) return;
		if (TrySwapDown(x, y, data)) return;
		return;
	case 4:
		PCG_state = PCG32_hash(PCG_state);
		if (toFloat01(PCG_state) < 0.1f)
			if (TrySwapDown(x, y, data)) return;
		if (toFloat01(PCG_state) < 0.3f)
			if (TrySwapDownDiag(x, y, data)) return;
		if (toFloat01(PCG_state) < 0.8f)
			if (TrySwapUpDiag(x, y, data)) return;
		if (TrySwapUp(x, y, data)) return;
		return;
	case 5:
		if (TrySwapDownVel(x, y, 3, data)) return;
		if (TrySwapDownDiag(x, y, data)) return;
		if (TrySwapHorizontal(x, y, 3, data)) return;
		return;
	case 6:
		PCG_state = PCG32_hash(PCG_state);
		if (toFloat01(PCG_state) < 0.1f)
			if (TrySwapDown(x, y, data)) return;
		if (toFloat01(PCG_state) < 0.3f)
			if (TrySwapDownDiag(x, y, data)) return;
		if (toFloat01(PCG_state) < 0.8f)
			if (TrySwapUpDiag(x, y, data)) return;
		if (TrySwapUp(x, y, data)) return;
		return;
	default:
		return;
	}
}

__device__ void PixelReactionUpdate(int x, int y, int t)
{
	if (InvalidCoords(x, y)) return;

	uint8_t id = GetPixelData(x, y)->id;
	if (id == 0) return;

	RGBA currentPix = GetPixel(x, y);

	PixelData data = *GetPixelData(x, y);
	uint pixelIdx = y * w + x;
	uint PCG_state = PCG32_hash(PCG32_hash(pixelIdx) * PCG32_hash(dIters) + PCG32_hash(t));
	switch (data.id)
	{
	case 2:
		TouchReactionChunk(x, y);
		if (toFloat01(PCG_state) < 0.0004f)
		{
			SetPixelFromPrefab(SMOKE, x, y);
			return;
		}
		return;
	case 3:
		TouchReactionChunk(x, y);
		if (toFloat01(PCG_state) < 0.0002f)
		{
			SetPixelFromPrefab(WATER, x, y);
			return;
		}
		return;
	case 4:
		TouchReactionChunk(x, y);
		if (toFloat01(PCG_state) < 0.8f)
			GetPixelData(x, y)->lifetime--;
		if (GetPixelData(x, y)->lifetime <= 0)
			SetPixelFromPrefab(AIR, x, y);
		return;
	case 6:
		TouchReactionChunk(x, y);
		for (int xOffset = -2; xOffset <= 2; xOffset++)
		{
			for (int yOffset = -2; yOffset <= 2; yOffset++)
			{
				if (xOffset == 0 && yOffset == 0) continue;
				if (InvalidCoords(x + xOffset, y + yOffset)) continue;
				PixelData adjData = *GetPixelData(x + xOffset, y + yOffset);
				if (!adjData.flammable) continue;
				PCG_state = PCG32_hash(PCG_state);
				if (toFloat01(PCG_state) < 0.05f)
				{
					SetPixelFromPrefab(FIRE, x + xOffset, y + yOffset);
				}
			}
		}
		PCG_state = PCG32_hash(PCG_state);
		if (toFloat01(PCG_state) < 0.2f)
		{
			GetPixelData(x, y)->lifetime--;
			int r = FIRE.defaultColor.r - -2 * (10 - data.lifetime);
			int g = FIRE.defaultColor.g - 12 * (10 - data.lifetime);
			int b = FIRE.defaultColor.b - 7 * (10 - data.lifetime);
			RGBA randomized = randomizeColor(RGBA(r, g, b), 8, PCG_state);
			SetPixel(randomized, x, y);
		}
		if (GetPixelData(x, y)->lifetime <= 0)
			SetPixelFromPrefab(SMOKE, x, y);
		return;
	default:
		return;
	}
}
