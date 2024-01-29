#pragma once

#include "pixelDatatypes.h"
#include "pixelDataImpl.h"
#include "xoshiro.h"

#include <cmath>

__device__ constexpr int midx(int x, int y)
{
	return ((y + h) % h) * w + ((x + w) % w);
}
__device__ constexpr int idx(int x, int y)
{
	return y * w + x;
}

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

__device__ RGBA randomizeColor(const RGBA base, const int variation, const uint seed)
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

__device__ void SetPixel_Mat(Material mat, int x, int y)
{
	mats[idx(x, y)] = mat;
	uint PCG_state = PCG32_hash(PCG32_hash(idx(x, y)) * PCG32_hash(dIters) + 37);
	RGBA c = RGBA();
	if(mat != AIR) c = randomizeColor(materials[mat].meanColor, 16, PCG_state);
	pixelStates[idx(x, y)] = PixelState(c);
	pixelStates[idx(x, y)].exactPos = Vec2f(x, y);
	TouchChunk(x, y);
}

__device__ void MoveTo(int x1, int y1, int x2, int y2)
{
	mats[midx(x2, y2)] = mats[idx(x1, y1)];
	pixelStates[midx(x2, y2)] = pixelStates[idx(x1, y1)];
	SetPixel_Mat(AIR, x1, y1);
	TouchChunk(x2, y2);
}

__device__ Vec2f ApplyForces(int x, int y, float dt)
{
	Vec2f v = pixelStates[idx(x, y)].velocity;

	const Vec2f gravity = { 0, 0.3f };
	v = v + dt * gravity * (1 + (0.1f * toFloat01(PCG32_hash(PCG32_hash(x * 1253 + y % 2314 + dIters * 23891)))));

	float m = v.magnitude();
	if (m > ((CHUNKDIM >> 1) / dt)) v = v * ((CHUNKDIM >> 1) / (dt * m));

	return v;
}

__device__ Vec2f ProjectVelocity(int x, int y, Vec2f velocity)
{
	bool dOpen = mats[midx(x, y + 1)] == AIR;
	if (dOpen) return velocity;

	bool dLOpen = mats[midx(x - 1, y + 1)] == AIR;
	bool dROpen = mats[midx(x + 1, y + 1)] == AIR;
	if (dLOpen && dROpen)
	{
		float a = velocity.angle();
		if (a > 90 && a < 270 - 10) dROpen = false;
		else if (a > 270 + 10) dLOpen = false;
		else
		{
			uint n = PCG32_hash(x * dIters + y * 187 + *(int*)&velocity.y);
			bool b = (PCG32_hash(n) & (1 << (dIters % 30))) > 0;
			if (b) dLOpen = false;
			else dROpen = false;
		}
	}
	if (dLOpen || dROpen)
	{
		pixelStates[idx(x, y)].exactPos.y = fminf(pixelStates[idx(x, y)].exactPos.y, ceilf(pixelStates[idx(x, y)].exactPos.y) - 0.2f);

		Vec2f belowVel = pixelStates[midx(x, y + 1)].velocity;
		Vec2f velDiff = velocity - belowVel;

		float magnitude = velDiff.magnitude();
		float adjustedMagnitude = sqrtf(magnitude * 3 + 0.01f) + 0.5f;
		Vec2f v = Vec2f(0.707f, 0.707f) * adjustedMagnitude;
		if (dLOpen) v.x *= -1;
		return v + (belowVel * 1.0f);
	}
	return Vec2f(0, 0);
}

__device__ void MoveWithVelocity(int x, int y, Vec2f velocity, float dt)
{
	if (velocity == Vec2f(0, 0)) return;

	constexpr int velLoops = 8;

	Vec2f origPos = pixelStates[idx(x, y)].exactPos;
	Vec2i lastPos = Vec2i(origPos);
	Vec2i newPos;
	Vec2f preciseNewPos = origPos;

	for(int i = 1; i <= velLoops; i++)
	{
		preciseNewPos += velocity * (dt / velLoops);
		newPos = Vec2i(preciseNewPos);
		
		if (newPos != lastPos)
		{
			if (InvalidCoords(newPos.x, newPos.y)) {
				Vec2f actualVel = velocity * ((float)i / velLoops);
				pixelStates[idx(x, y)].velocity = actualVel;
				preciseNewPos -= velocity * (dt / velLoops);
				break;
			}
			if (mats[midx(newPos.x, newPos.y)] > AIR) {
				bool moved = true;
				if (fabsf(velocity.x) > fabsf(velocity.y)) {
					int delta = (velocity.x > 0) * 2 - 1;
					if (mats[midx(newPos.x + delta, newPos.y)] == AIR) preciseNewPos.x = (int)preciseNewPos.x + 1.5f * delta;
					else if (mats[midx(newPos.x - delta, newPos.y)] == AIR) preciseNewPos.x = (int)preciseNewPos.x - 0.5f * delta;
					else moved = false;
				}
				else {
					int delta = (velocity.y > 0) * 2 - 1;
					if (mats[midx(newPos.x, newPos.y + delta)] == AIR) preciseNewPos.y = (int)preciseNewPos.y + 1.5f * delta;
					else if (mats[midx(newPos.x, newPos.y - delta)] == AIR) preciseNewPos.y = (int)preciseNewPos.y - 0.5f * delta;
					else moved = false;
				}

				if (!moved) {
					Vec2f actualVel = velocity * ((float)i / velLoops);
					pixelStates[idx(x, y)].velocity = actualVel;
					preciseNewPos -= velocity * (dt / velLoops);
					break;
				}
				i += 2;
			}
			lastPos = newPos;
		}
	}

	//if (i < velLoops) //collision
	//{
	//	Vec2f actualVel = velocity * ((float)i / velLoops);
	//	pixelStates[idx(x, y)].velocity = actualVel;
	//}

	preciseNewPos.x = fmodf(preciseNewPos.x, w);
	preciseNewPos.y = fmodf(preciseNewPos.y, h);

	pixelStates[idx(x, y)].exactPos = preciseNewPos;

	TouchChunk(x, y);

	//if((preciseLastPos - origPos).magnitude() > 0.01f && PCG32_hash(PCG32_hash(x * 8135 + y % 87)) < 100000)
	//	printf("this delta: %f %f\n", preciseLastPos.x - origPos.x, preciseLastPos.y - origPos.y);

	if(Vec2i(x, y) != Vec2i(preciseNewPos))
		MoveTo(x, y, (int)preciseNewPos.x, (int)preciseNewPos.y);
}