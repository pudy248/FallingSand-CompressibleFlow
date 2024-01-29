#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "error.h"

#include <stdio.h>
#include <chrono>
#include <SFML/Graphics.hpp>

#include "Fluid.h"
//#include "verlet.h"

using namespace sf;
using namespace std;

int main()
{
	constexpr auto width = 1920;
	constexpr auto height = 1080;

	RenderWindow window(VideoMode(width, height), "SFML Window", Style::Fullscreen, ContextSettings(24U, 8U, 4U, 4U, 0U, sf::ContextSettings::Attribute::Default));
	window.setPosition(Vector2i(0, 0));
	window.setVerticalSyncEnabled(true);
	
	texInit(width, height);
	//verletInit(width, height);

	//FPS init
	constexpr auto fpsAverage = 100;
	int fpsIdx = 0;
	double frameTimes[fpsAverage];
	for (int i = 0; i < fpsAverage; i++) frameTimes[i] = 1.0f / fpsAverage;
	Font arial;
	arial.loadFromFile("arial.ttf");
	Text fpsCounter;
	fpsCounter.setFont(arial);
	fpsCounter.setCharacterSize(24);
	fpsCounter.setFillColor(Color::White);

	bool keyToggle = false;

	chrono::steady_clock::time_point time0 = chrono::steady_clock::now();
	chrono::steady_clock::time_point lastTimepoint = time0;
	while (window.isOpen())//for(int iters2 = 0; iters2 < 100; iters2++)//
	{
		chrono::steady_clock::time_point time1 = chrono::steady_clock::now();
		chrono::nanoseconds sinceStart = time1 - time0;
		chrono::nanoseconds sinceLast = time1 - lastTimepoint;
		lastTimepoint = time1;
		double totalSeconds = sinceStart.count() / 1'000'000'000.0;
		double frameSeconds = sinceLast.count() / 1'000'000'000.0;

		//frameSeconds = fminf(frameSeconds, 1.0 / 60);

		Event event;
		while (window.pollEvent(event))
		{
			if (event.type == Event::Closed)
				window.close();
		}

		window.clear();

		if (Keyboard::isKeyPressed(Keyboard::Z))
		{
			texCleanup();
			texInit(width, height);
		}
		else
		{
			texRender(window, width, height, (float)frameSeconds);
			//verletRender(window, width, height, 1 / 60.0f);
		}
		
		keyToggle = Keyboard::isKeyPressed(Keyboard::Z);

		//FPS counter
		frameTimes[fpsIdx] = frameSeconds;
		fpsIdx = (fpsIdx + 1) % fpsAverage;
		double frameAverage = 0;
		for (int i = 0; i < fpsAverage; i++) frameAverage += frameTimes[i];
		frameAverage /= fpsAverage;
		double frameRate = 1.0f / frameAverage;
		char buffer[20];
		sprintf(buffer, "%.2f", frameRate);
		fpsCounter.setString(buffer);
		window.draw(fpsCounter);


		window.display();
	}

	texCleanup();
	//verletCleanup();

	return 0;
}