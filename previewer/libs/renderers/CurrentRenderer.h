#if defined RENDER_FF_VBO
	#include "FF_VBO.h"
#elif defined RENDER_PP_GEOM
	#include "PP_GEOM.h"
#elif defined RENDER_PP_FBO
	#include "PP_FBO.h"
#elif defined RENDER_PP_FBOF
	#include "PP_FBOF.h"
#endif