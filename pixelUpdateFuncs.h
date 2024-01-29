#pragma once

#include "PixelHelperFuncs.h"

//Update logic - Verlet-style delta-pos velocity updating
//Update velocity from forces
//Move with velocity
//Falling sand step (no velocity needed)
//Set new velocity to newPos - oldPos


__device__ void FallingSandUpdate(int x, int y, float dt)
{
	if (InvalidCoords(x, y)) return;
	if (pixelsUpdated[idx(x, y)]) return;

	Material mat = mats[idx(x, y)];
	if (mat == AIR) return;
	PixelState& state = pixelStates[idx(x, y)];
	pixelsUpdated[idx(x, y)] = true;

	if (mat == SAND)
	{
		Vec2f oldPos = state.exactPos;
		Vec2f velocity = ApplyForces(x, y, dt);
		velocity = ProjectVelocity(x, y, velocity);
		pixelStates[idx(x, y)].velocity = velocity;
		MoveWithVelocity(x, y, velocity, dt);
	}
	else if (mat == WALL)
		pixelStates[idx(x, y)].velocity = Vec2f(0, 0);
}