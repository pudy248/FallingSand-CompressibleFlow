#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

typedef unsigned char uint8_t;
typedef unsigned int uint;
typedef unsigned long long uint64_t;

__host__ __device__ uint64_t splitMix64(uint64_t state)
{
	uint64_t result = state + 0x9E3779B97f4A7C15;
	result = (result ^ (result >> 30)) * 0xBF58476D1CE4E5B9;
	result = (result ^ (result >> 27)) * 0x94D049BB133111EB;
	return result ^ (result >> 31);
}
__host__ __device__ uint64_t rol64(uint64_t x, int k)
{
	return (x << k) | (x >> (64 - k));
}

__host__ __device__ static uint PCG32_hash(uint input)
{
	uint state = input * 747796405u + 2891336453u;
	uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
	return (word >> 22u) ^ word;
}

__host__ __device__ float toFloat01(uint bits)
{
	uint mask1 = 0x3F800000U;
	uint mask2 = 0x3FFFFFFFU;
	uint to_12 = (bits | mask1) & mask2;
	float d = *(float*)&to_12;
	return d - 1;
}

struct Xoshiro256ssc
{
	uint64_t world_seed;
	uint64_t state0;
	uint64_t state1;
	uint64_t state2;
	uint64_t state3;

	__host__ __device__ Xoshiro256ssc(uint64_t seed = 1)
	{
		world_seed = seed;
		uint64_t sm = splitMix64(seed);
		state0 = sm;
		sm = splitMix64(sm);
		state1 = sm;
		sm = splitMix64(sm);
		state2 = sm;
		sm = splitMix64(sm);
		state3 = sm;
	}

	__host__ __device__ void SetRandomSeed(double x, double y)
	{
		uint64_t sm1 = splitMix64(world_seed);
		uint64_t sm2 = splitMix64(*(uint64_t*)&x + 0xAF0B4194EACB8ECFU);
		uint64_t sm3 = splitMix64(*(uint64_t*)&y + 0x131E6CD14935C660U);
		uint64_t smComp = sm1 ^ sm2 ^ sm3; //I made this up but it might be random enough to be safe
		state0 = smComp;
		smComp = splitMix64(smComp);
		state1 = smComp;
		smComp = splitMix64(smComp);
		state2 = smComp;
		smComp = splitMix64(smComp);
		state3 = smComp;
	}

	__host__ __device__ uint64_t NextU()
	{
		uint64_t result = rol64(state1 * 5, 7) * 9;
		uint64_t t = state1 << 17;
		state2 ^= state0;
		state3 ^= state1;
		state1 ^= state2;
		state0 ^= state3;

		state2 ^= t;
		state3 = rol64(state3, 45);

		return result;
	}
	__host__ __device__ double Next()
	{
		uint64_t bits = NextU();
		uint64_t mask1 = 0x3FF0000000000000UL;
		uint64_t mask2 = 0x3FFFFFFFFFFFFFFFUL;
		uint64_t to_12 = (bits | mask1) & mask2;
		double d = *(double*)&to_12;
		return d - 1;
	}

	__host__ __device__ int Randomi(int a, int b)
	{
		uint64_t bits = NextU();
		return ((int)bits % (b - a)) + a;
	}
	__host__ __device__ float Randomf(float a, float b)
	{
		return (float)(a + ((b - a) * Next()));
	}

	__host__ __device__ void jump() {
		static const uint64_t JUMP[] = { 0x180ec6d33cfd0aba, 0xd5a61266f0c9392c, 0xa9582618e03fc9aa, 0x39abdc4529b1661c };

		uint64_t s0 = 0;
		uint64_t s1 = 0;
		uint64_t s2 = 0;
		uint64_t s3 = 0;
		for (int i = 0; i < sizeof JUMP / sizeof * JUMP; i++)
			for (int b = 0; b < 64; b++) {
				if (JUMP[i] & uint64_t(1) << b) {
					s0 ^= state0;
					s1 ^= state1;
					s2 ^= state2;
					s3 ^= state3;
				}
				NextU();
			}

		state0 = s0;
		state1 = s1;
		state2 = s2;
		state3 = s3;
	}

	__host__ __device__ void long_jump() {
		static const uint64_t LONG_JUMP[] = { 0x76e15d3efefdcbbf, 0xc5004e441c522fb3, 0x77710069854ee241, 0x39109bb02acbe635 };

		uint64_t s0 = 0;
		uint64_t s1 = 0;
		uint64_t s2 = 0;
		uint64_t s3 = 0;
		for (int i = 0; i < sizeof LONG_JUMP / sizeof * LONG_JUMP; i++)
			for (int b = 0; b < 64; b++) {
				if (LONG_JUMP[i] & uint64_t(1) << b) {
					s0 ^= state0;
					s1 ^= state1;
					s2 ^= state2;
					s3 ^= state3;
				}
				NextU();
			}

		state0 = s0;
		state1= s1;
		state2 = s2;
		state3 = s3;
	}
};