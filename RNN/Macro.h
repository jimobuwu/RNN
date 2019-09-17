#pragma once

#ifdef DOUBLE_PRECISION
typedef double FLOAT;
#else
typedef float FLOAT;
#endif 

#define BINARY_NUM 8
#define randVal(x) ((FLOAT)rand() / RAND_MAX * x)
