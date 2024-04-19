#include <cstdint>
#include <cmath>

#define abs(a) ((a) < 0 ? -(a) : (a))
#define min(a,b) ((a) < (b) ? (a) : (b))
#define max(a,b) ((a) > (b) ? (a) : (b))

struct UniversalColor {
	union {
		float r;
		float h;
	};
	union {
		float g;
		float s;
	};
	union {
		float b;
		float l;
	};
};

__host__ __device__ UniversalColor rgb2hsl(UniversalColor rgb) {
	UniversalColor result;

	rgb.r /= 255;
	rgb.g /= 255;
	rgb.b /= 255;

	float _max = max(max(rgb.r, rgb.g), rgb.b);
	float _min = min(min(rgb.r, rgb.g), rgb.b);

	result.h = result.s = result.l = (_max + _min) / 2;

	if (abs(_max - _min) < 0.0001f) {
		result.h = result.s = 0; // achromatic
	}
	else {
		float d = _max - _min;
		result.s = (result.l > 0.5f) ? d / (2 - _max - _min) : d / (_max + _min);

		if (abs(_max - rgb.r) < 0.0001f) {
			result.h = (rgb.g - rgb.b) / d + (rgb.g < rgb.b ? 6 : 0);
		}
		else if (abs(_max - rgb.g) < 0.0001f) {
			result.h = (rgb.b - rgb.r) / d + 2;
		}
		else if (abs(_max - rgb.b) < 0.0001f) {
			result.h = (rgb.r - rgb.g) / d + 4;
		}

		result.h /= 6;
	}

	return result;
}

__host__ __device__ static float hue2rgb(float p, float q, float t) {
	if (t < 0)
		t += 1;
	if (t > 1)
		t -= 1;
	if (t < 1.f / 6)
		return p + (q - p) * 6 * t;
	if (t < 1.f / 2)
		return q;
	if (t < 2.f / 3)
		return p + (q - p) * (2.f / 3 - t) * 6;

	return p;
}

__host__ __device__ UniversalColor hsl2rgb(UniversalColor hsl) {
	UniversalColor result;

	if (abs(hsl.s) < 0.0001f) {
		result.r = result.g = result.b = hsl.l * 255; // achromatic
	}
	else {
		float q = hsl.s < 0.5f ? hsl.l * (1 + hsl.s) : hsl.l + hsl.s - hsl.l * hsl.s;
		float p = 2 * hsl.l - q;
		result.r = hue2rgb(p, q, hsl.h + 1.f / 3) * 255 * hsl.l;
		result.g = hue2rgb(p, q, hsl.h) * 255 * hsl.l;
		result.b = hue2rgb(p, q, hsl.h - 1.f / 3) * 255 * hsl.l;
	}

	return result;
}
