#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "error.h"
#include "xoshiro.h"
#include "pixelUpdateFuncs.h"

#include <chrono>
#include <SFML/Graphics.hpp>

using namespace sf;
using namespace std;

Texture tex;

struct MouseInput
{
	bool lmbDown;
	bool rmbDown;
	int x;
	int y;
};

__global__ void HandleInput(MouseInput input)
{
	if (input.lmbDown)
		for (int xOffset = -INPUT_RADIUS; xOffset <= INPUT_RADIUS; xOffset++)
		{
			for (int yOffset = -INPUT_RADIUS; yOffset <= INPUT_RADIUS; yOffset++)
			{
				if (InvalidCoords(input.x + xOffset, input.y + yOffset)) continue;
				SetPixelFromPrefab(OIL, input.x + xOffset, input.y + yOffset);
			}
		}
	else if (input.rmbDown)
		for (int xOffset = -INPUT_RADIUS; xOffset <= INPUT_RADIUS; xOffset++)
		{
			for (int yOffset = -INPUT_RADIUS; yOffset <= INPUT_RADIUS; yOffset++)
			{
				if (InvalidCoords(input.x + xOffset, input.y + yOffset)) continue;
				SetPixelFromPrefab(FIRE, input.x + xOffset, input.y + yOffset);
			}
		}
}

__global__ void TextureDiffusePass(double t, int pass)
{
	uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
	uint32_t stride = blockDim.x * gridDim.x;

	if (index == 0 && pass == 0)
	{
		dIters++;

		if (dIters % 3 == 0)
		{
			SetPixelFromPrefab(SAND, 200, 200);
			if (t > 5)
			{
				SetPixelFromPrefab(OIL, 300, 200);
			}
		}
	}

	__syncthreads();

	for (int chunkIdx = index; chunkIdx < (w / CHUNKDIM + 1) * (h / CHUNKDIM + 1); chunkIdx += stride)
	{
		if (!chunkData[chunkIdx].lastUpdated) continue;
		int cx = chunkIdx % (w / CHUNKDIM + 1);
		int cy = chunkIdx / (w / CHUNKDIM + 1);

		int xOffset = (CHUNKDIM / 2) * (pass % 2);
		int yOffset = (CHUNKDIM / 2) * (pass / 2);

		int baseX = cx * CHUNKDIM + xOffset;
		int baseY = cy * CHUNKDIM + yOffset;

		for (int px = baseX; px < baseX + (CHUNKDIM / 2); px++)
		{
			for (int py = baseY; py < baseY + (CHUNKDIM / 2); py++)
			{
				FallingSandUpdate(px, py, t);
			}
		}
	}
}

__global__ void UpdateReactions(double t)
{
	uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
	uint32_t stride = blockDim.x * gridDim.x;

	for (int chunkIdx = index; chunkIdx < (w / CHUNKDIM + 1) * (h / CHUNKDIM + 1); chunkIdx += stride)
	{
		if (!chunkData[chunkIdx].reactionUpdated) continue;
		chunkData[chunkIdx].reactionUpdated = false;
		int cx = chunkIdx % (w / CHUNKDIM + 1);
		int cy = chunkIdx / (w / CHUNKDIM + 1);

		int baseX = cx * CHUNKDIM;
		int baseY = cy * CHUNKDIM;

		for (int px = baseX; px < baseX + CHUNKDIM; px++)
		{
			for (int py = baseY; py < baseY + CHUNKDIM; py++)
			{
				PixelReactionUpdate(px, py, t);
			}
		}
	}
}

__global__ void ResetUpdates()
{
	uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
	uint32_t stride = blockDim.x * gridDim.x;

	for (int pix = index; pix < w * h; pix += stride)
	{
		int x = pix % w;
		int y = pix / w;

		GetPixelData(x, y)->updated = false;
	}

	for (int chunk = index; chunk < (w / CHUNKDIM + 1) * (h / CHUNKDIM + 1); chunk += stride)
	{
		chunkData[chunk].lastUpdated = chunkData[chunk].updated;
		chunkData[chunk].reactionUpdated |= chunkData[chunk].updated;
		chunkData[chunk].updated = false;
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

		uint64_t num[1] = { rnd.NextU() };
		uint8_t* bytes = (uint8_t*)num;
		for (int i = 0; i < 8; i++)
		{
			uint8_t b = bytes[i];
			int x2 = (pix + i) % w;
			int y2 = (pix + i) / w;

			if (b < 220) SetPixelFromPrefab(AIR, x2, y2);
			else if(b < 230) SetPixelFromPrefab(SAND, x2, y2);
			else SetPixelFromPrefab(OIL, x2, y2);
		}
	}

	for (int chunk = index; chunk < (w / CHUNKDIM + 1) * (h / CHUNKDIM + 1); chunk += stride)
	{
		chunkData[chunk].lastUpdated = true;
		chunkData[chunk].updated = true;
		chunkData[chunk].reactionUpdated = true;
	}
}

void texInit(const int _width, const int _height)
{
	int width = _width / DOWNSCALE;
	int height = _height / DOWNSCALE;

	hPixels = (RGBA*)malloc(width * height * 4);
	hChunkData = (ChunkData*)malloc((width / CHUNKDIM + 1) * (height / CHUNKDIM + 1) * sizeof(ChunkData));

	checkCudaErrors(cudaMalloc(&dPixels, width * height * 4));
	checkCudaErrors(cudaMalloc(&dPixelData, width * height * sizeof(PixelData)));
	checkCudaErrors(cudaMalloc(&dChunkData, (width / CHUNKDIM + 1) * (height / CHUNKDIM + 1) * sizeof(ChunkData)));

	checkCudaErrors(cudaMemcpyToSymbol(texture, &dPixels, sizeof(RGBA*)));
	checkCudaErrors(cudaMemcpyToSymbol(pixelData, &dPixelData, sizeof(PixelData*)));
	checkCudaErrors(cudaMemcpyToSymbol(chunkData, &dChunkData, sizeof(ChunkData*)));

	checkCudaErrors(cudaMemcpyToSymbol(w, &width, sizeof(int)));
	checkCudaErrors(cudaMemcpyToSymbol(h, &height, sizeof(int)));

	InitTexture << <NUMBLOCKS, BLOCKSIZE >> > (1234);
	checkCudaErrors(cudaDeviceSynchronize());

	tex = Texture();
	tex.create(width, height);
}

void texRender(RenderWindow& window, const int _width, const int _height, const float totalSeconds)
{
	int width = _width / DOWNSCALE;
	int height = _height / DOWNSCALE;

	MouseInput mouseInput = { Mouse::isButtonPressed(Mouse::Left), Mouse::isButtonPressed(Mouse::Right), Mouse::getPosition().x / DOWNSCALE, Mouse::getPosition().y / DOWNSCALE };
	HandleInput << <1, 1 >> > (mouseInput);

	bool paused = Keyboard::isKeyPressed(Keyboard::Space);
	if (!paused)
	{
		for (int loops = 0; loops < LOOPCOUNT; loops++)
		{
			hIters++;
			ResetUpdates << <NUMBLOCKS, BLOCKSIZE >> > ();
			checkCudaErrors(cudaDeviceSynchronize());

			if (hIters % REACTION_RATE == 0)
			{
				UpdateReactions << <NUMBLOCKS, BLOCKSIZE >> > (totalSeconds);
				checkCudaErrors(cudaDeviceSynchronize());
			}

			for (int pass = 0; pass < 4; pass++)
			{
				TextureDiffusePass << <NUMBLOCKS, BLOCKSIZE >> > (totalSeconds, pass);
				checkCudaErrors(cudaDeviceSynchronize());
			}
		}
	}
	checkCudaErrors(cudaMemcpy(hPixels, dPixels, 4 * width * height, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaDeviceSynchronize());

	//Drawing
	tex.update((uint8_t*)hPixels);
	Sprite sprite = Sprite(tex);
	sprite.setScale(Vector2f(DOWNSCALE, DOWNSCALE));
	window.draw(sprite);

	//Chunk Updater 9000
	if (Keyboard::isKeyPressed(Keyboard::Tab))
	{
		int loadedChunks = 0;
		int reactionChunks = 0;
		checkCudaErrors(cudaMemcpy(hChunkData, dChunkData, (width / CHUNKDIM + 1) * (height / CHUNKDIM + 1) * sizeof(ChunkData), cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaDeviceSynchronize());
		for (int chunkIdx = 0; chunkIdx < (width / CHUNKDIM + 1) * (height / CHUNKDIM + 1); chunkIdx++)
		{
			ChunkData dat = hChunkData[chunkIdx];
			if (!dat.updated && !dat.reactionUpdated) continue;

			if(dat.updated)
				loadedChunks++;
			if (dat.reactionUpdated)
				reactionChunks++;

			int cx = chunkIdx % (width / CHUNKDIM + 1);
			int cy = chunkIdx / (width / CHUNKDIM + 1);

			RectangleShape shape(Vector2f(CHUNKDIM * DOWNSCALE, CHUNKDIM * DOWNSCALE));
			shape.setFillColor(Color(0, 0, 0, 0));
			shape.setOutlineThickness(-1);

			if(dat.updated)
				shape.setOutlineColor(Color(255, 255, 0));
			else
				shape.setOutlineColor(Color(255, 0, 0));

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

		sprintf(buffer, "%i/%i reaction chunks", reactionChunks, (width / CHUNKDIM + 1) * (height / CHUNKDIM + 1));
		chunkCounter.setString(buffer);
		chunkCounter.setPosition(Vector2f(0, 50));
		window.draw(chunkCounter);

	}
}

void texCleanup()
{
	cudaFree(dPixels);
	cudaFree(dPixelData);
	cudaFree(dChunkData);
	free(hPixels);
	free(hChunkData);
}