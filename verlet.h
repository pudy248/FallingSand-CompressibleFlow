#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "error.h"
#include "xoshiro.h"

#include <chrono>
#include <SFML/Graphics.hpp>

using namespace sf;
using namespace std;

struct vec2 {
	float x;
	float y;

	__host__ __device__ vec2() {
		x = 0;
		y = 0;
	}
	__host__ __device__ vec2(float _x, float _y) {
		x = _x;
		y = _y;
	}


	__host__ __device__ float magnitude() {
		return sqrtf(x * x + y * y);
	}
	__host__ __device__ vec2 normalized() {
		return vec2(x / this->magnitude(), y / this->magnitude());
	}

	__host__ __device__ vec2 operator+(vec2 other) {
		return vec2(x + other.x, y + other.y);
	}
	__host__ __device__ vec2 operator-(vec2 other) {
		return vec2(x - other.x, y - other.y);
	}
	
	__host__ __device__ vec2 operator*(float scalar) {
		return vec2(x * scalar, y * scalar);
	}
	__host__ __device__ vec2 operator/(float scalar) {
		return vec2(x / scalar, y / scalar);
	}
};
__host__ __device__ float dot(vec2 a, vec2 b) {
	return a.x * b.x + a.y * b.y;
}

__host__ __device__ vec2 proj(vec2 vec, vec2 axis) {
	float k = dot(vec, axis) / dot(axis, axis);
	return axis * k;
}

__host__ __device__ vec2 reflectDampen(vec2 vel, vec2 axis, float bounciness) {
	vec2 perpendicular = proj(vel, axis);
	vec2 parallel = vel - perpendicular;

	return parallel - (perpendicular * bounciness);
}



struct Ball {
	vec2 lastPosition;
	vec2 position;
	vec2 _lastPosition;
	vec2 _position;
	vec2 nextPosition;
	vec2 acceleration;
	vec2 verletMovement;
	float radius;
	float mass;
};

__device__ __constant__ Ball* balls;
__device__ __constant__ int ballCount;
Ball* dBalls;
Ball* hBalls;


/*__device__ void CollideStatic(Ball& ball, vec2 collisionOverlap) {
	ball.position = ball.position - collisionOverlap;
	ball.velocity = reflectDampen(ball.velocity, collisionOverlap, 0.8f);
}

__device__ void CollideDynamic(Ball& ball, Ball& ball2, vec2 collisionOverlap) {
	ball.position = ball.position + (collisionOverlap / 2);
	ball2.position = ball2.position - (collisionOverlap / 2);

	float elasticity = 0.8f;
	vec2 vel1 = ((ball2.velocity - ball.velocity) * elasticity * ball2.mass) + (ball.velocity * ball.mass) + (ball2.velocity * ball2.mass);
	vec2 vel2 = ((ball.velocity - ball2.velocity) * elasticity * ball.mass) + (ball.velocity * ball.mass) + (ball2.velocity * ball2.mass);
	ball.velocity = vel1 / (ball.mass + ball2.mass);
	ball2.velocity = vel2 / (ball.mass + ball2.mass);
}*/


__global__ void UpdateForces(float rate) {
	uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
	uint32_t stride = blockDim.x * gridDim.x;

	for (int ballIdx = index; ballIdx < ballCount; ballIdx += stride) {
		Ball& ball = balls[ballIdx];
		vec2 dr = ball.position - ball.lastPosition;
		vec2 vel = dr * rate;
		float speed = vel.magnitude();

		constexpr auto airResistanceFactor = 0.000001f;
		float airResistance = -1 * airResistanceFactor * speed * speed;
		vec2 airResistanceAccel = vel * airResistance;
		vec2 gravity = vec2(0, 200);

		balls[ballIdx].acceleration = airResistanceAccel + gravity;
	}
}

__global__ void UpdateBalls(float originX, float originY, float maxRadius) {
	uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
	uint32_t stride = blockDim.x * gridDim.x;

	for (int ballIdx = index; ballIdx < ballCount; ballIdx += stride) {
		Ball& ball = balls[ballIdx];

		vec2 origin(originX, originY);
		vec2 newRel = ball.position - origin; //origin -> ball
		float relativeMagnitude = newRel.magnitude();
		if (relativeMagnitude > maxRadius - ball.radius) {
			vec2 oldRel = ball.lastPosition - origin;

			vec2 newEdge = newRel.normalized() * (maxRadius - ball.radius);
			vec2 oldEdge = oldRel.normalized() * (maxRadius - ball.radius);

			vec2 newToEdge = newEdge - newRel;
			vec2 oldToEdge = oldEdge - oldRel;

			vec2 newMirrored = newEdge + newToEdge;
			vec2 oldMirrored = oldEdge + oldToEdge;

			float scaleFactor;
			if ((newToEdge + oldToEdge).magnitude() > 0.000001f)
				scaleFactor = (oldToEdge.magnitude() / (newToEdge + oldToEdge).magnitude());
			else
				scaleFactor = 0.5f;

			vec2 newVel = newMirrored - oldMirrored;
			vec2 velIntercept = newVel * scaleFactor;
			vec2 posIntercept = oldMirrored + velIntercept;

			vec2 intersectToNewEdge = newEdge - posIntercept;
			vec2 oldEdgeToIntersect = posIntercept - oldEdge;

			constexpr auto normalDampingCoeff = 0.8f;

			vec2 dampedNewVel = intersectToNewEdge + (newToEdge * normalDampingCoeff);
			vec2 dampedOldVel = oldEdgeToIntersect - (oldToEdge * normalDampingCoeff);
			vec2 finalNewPos = posIntercept + dampedNewVel;
			vec2 finalOldPos = posIntercept - dampedOldVel;

			ball.position = origin + finalNewPos;
			ball.lastPosition = origin + finalOldPos;
		}

		ball._position = ball.position;
		ball._lastPosition = ball.lastPosition;
	}
}

__global__ void CollideBalls() {
	uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
	uint32_t stride = blockDim.x * gridDim.x;

	for (int ballIdx = index; ballIdx < ballCount; ballIdx += stride) {
		Ball& ball = balls[ballIdx];
		for (int i = 0; i < ballCount; i++) {
			if (i == ballIdx) continue;
			Ball& ball2 = balls[i];

			constexpr auto dampingCoeff = 0.4f;

			
			vec2 relativePosition = ball2.position - ball.position;
			float distance = relativePosition.magnitude();
			if (distance < ball.radius + ball2.radius) {
				vec2 vel = ball.position - ball.lastPosition;
				vec2 vel2 = ball2.position - ball2.lastPosition;
				vec2 velRel = vel - vel2;

				vec2 velRelNormal = proj(velRel, relativePosition);
				vec2 overlap = relativePosition.normalized() * (ball.radius + ball2.radius - distance);
				
				float scaleFactor;
				if (velRelNormal.magnitude() > 0.000001f)
					scaleFactor = 1 - (overlap.magnitude() / velRelNormal.magnitude());
				else scaleFactor = 0.5f;

				vec2 velRelIntercept = velRel * scaleFactor;
				vec2 velRelTangent = velRel - velRelNormal;
				vec2 velRelTanScaled = velRelTangent * (1 - scaleFactor);
				vec2 velRelNormScaled = velRelNormal * (1 - scaleFactor);

				vec2 velRelNormDampened = velRelNormScaled * -dampingCoeff;
				vec2 velRelFinal = velRelNormDampened + velRelTanScaled;

				vec2 velRelOrig2nd = velRel - velRelIntercept;

				vec2 velRelFinalDelta = velRelFinal = velRelOrig2nd;


				vec2 velScaled = vel * scaleFactor;
				vec2 posIntercept = ball.lastPosition + velScaled;

				vec2 velBaseScaled = vel2 * (1 - scaleFactor);

				vec2 newVelScaled = velBaseScaled + velRelFinal;
				vec2 newVelScaledPt1 = newVelScaled * (scaleFactor / (1 - scaleFactor));

				vec2 finalOldPos = posIntercept - newVelScaledPt1;
				vec2 finalNewPos = posIntercept + newVelScaled;


				ball._position = finalNewPos;
				ball._lastPosition = finalOldPos;
			}
		}
	}
}

__global__ void MoveBalls(float dt) {
	uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
	uint32_t stride = blockDim.x * gridDim.x;

	for (int ballIdx = index; ballIdx < ballCount; ballIdx += stride) {
		Ball& ball = balls[ballIdx];

		ball.nextPosition = (ball._position * 2) - ball._lastPosition + (ball.acceleration * dt * dt);

		ball.lastPosition = ball._position;
		ball.position = ball.nextPosition;
	}
}



constexpr auto _ballCount = 30;
constexpr auto NUMBLOCKS = 256;
constexpr auto BLOCKSIZE = 128;
constexpr auto limitRadius = 350;

void verletInit(const int width, const int height) {
	hBalls = (Ball*)malloc(_ballCount * sizeof(Ball));

	Xoshiro256ssc rnd((uint64_t)(chrono::system_clock::now().time_since_epoch().count()));
	vec2 origin(width / 2.0f, height / 2.0f);

	for (int i = 0; i < _ballCount; i++) {
		do {
			hBalls[i].position = vec2(rnd.Randomf(width / 2.0f - limitRadius, width / 2.0f + limitRadius), rnd.Randomf(height / 2.0f - limitRadius, height / 2.0f + limitRadius));
		} while ((hBalls[i].position - origin).magnitude() > limitRadius);
		hBalls[i].lastPosition = hBalls[i].position;
		hBalls[i].radius = 10;
		hBalls[i].mass = 10;
	}

	checkCudaErrors(cudaMalloc(&dBalls, _ballCount * sizeof(Ball)));
	checkCudaErrors(cudaMemcpyToSymbol(balls, &dBalls, sizeof(Ball*)));
	checkCudaErrors(cudaMemcpyToSymbol(ballCount, &_ballCount, sizeof(int)));
	checkCudaErrors(cudaMemcpy(dBalls, hBalls, _ballCount * sizeof(Ball), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaDeviceSynchronize());
}

void verletRender(RenderWindow& window, const int width, const int height, const float dt) {

	UpdateForces<<<NUMBLOCKS, BLOCKSIZE>>>(1 / dt);
	checkCudaErrors(cudaDeviceSynchronize());
	UpdateBalls<<<NUMBLOCKS, BLOCKSIZE>>>(width / 2.0, height / 2.0, limitRadius);
	checkCudaErrors(cudaDeviceSynchronize());
	CollideBalls<<<NUMBLOCKS, BLOCKSIZE>>>();
	checkCudaErrors(cudaDeviceSynchronize());
	MoveBalls<<<NUMBLOCKS, BLOCKSIZE>>>(dt);
	checkCudaErrors(cudaDeviceSynchronize());

	checkCudaErrors(cudaMemcpy(hBalls, dBalls, _ballCount * sizeof(Ball), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaDeviceSynchronize());

	//Drawing
	window.clear(Color(64, 64, 64));

	sf::CircleShape bounds(limitRadius, 120);
	bounds.setFillColor(Color::Black);
	bounds.setPosition(width / 2 - limitRadius, height / 2 - limitRadius);
	window.draw(bounds);

	for (int i = 0; i < _ballCount; i++) {
		sf::CircleShape shape(hBalls[i].radius);
		shape.setFillColor(Color::White);
		shape.setPosition(hBalls[i].position.x - hBalls[i].radius, hBalls[i].position.y - hBalls[i].radius);

		window.draw(shape);
	}
}

void verletCleanup() {
	cudaFree(dBalls);
	free(hBalls);
}