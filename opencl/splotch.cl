#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable

typedef struct {
	float r, g, b;
} cu_color;

typedef struct {
	float x, y, r, I;
	int type;
	unsigned short maxx, minx;
	cu_color e;
	unsigned long posInFragBuf;
} cu_particle_splotch;

typedef struct {
	float p[12];
	bool projection;
	int xres, yres;
	float fovfct, dist, xfac;
	float minrad_pix;
	int ptypes;
	float zmaxval, zminval;
	bool col_vector[8];
	float brightness[8];
	float rfac;
} cu_param;

typedef struct {
	float val;
	cu_color color;
} cu_color_map_entry;

typedef struct {
	cu_color_map_entry *map;
	int mapSize;
	int *ptype_points;
	int ptypes;
} cu_colormap_info;

typedef struct {
	float aR, aG, aB;
} cu_fragment_AeqE;

typedef struct {
	float aR, aG, aB;
	float qR, qG, qB;
} cu_fragment_AneqE;

cu_color get_color(int ptype, float val, int mapSize, int ptypes,__global cu_color_map_entry *dmap_in,__global int *ptype_points_in)
{
	int map_size;
	int map_ptypes;

	map_size = mapSize;
	map_ptypes = ptypes;

	int start, end;
	start =ptype_points_in[ptype];
	if ( ptype == map_ptypes-1) end =map_size-1;
	else	end =ptype_points_in[ptype+1]-1;

	int i=start;
	while ((val>dmap_in[i+1].val) && (i<end)) ++i;

	const float fract = (val-dmap_in[i].val)/(dmap_in[i+1].val-dmap_in[i].val);
	cu_color clr1=dmap_in[i].color, clr2=dmap_in[i+1].color;
	cu_color clr;
	clr.r =clr1.r + fract*(clr2.r-clr1.r);
	clr.g =clr1.g + fract*(clr2.g-clr1.g);
	clr.b =clr1.b + fract*(clr2.b-clr1.b);

	return clr;
}

__kernel void k_render1
(__global cu_particle_splotch *p, int nP, __global
		cu_fragment_AeqE *buf, float grayabsorb, int mapSize, int types,__global cu_param *dparams1,
		__global cu_color_map_entry *dmap_in,__global int *ptype_points_in)
{

	int m;
	m = get_global_id(0);
	if (m<nP) 
        {
		int ptype = p[m].type;
		float col1=p[m].e.r,col2=p[m].e.g,col3=p[m].e.b;
		clamp (0.0000001,0.9999999,col1);
		if (dparams1->col_vector[ptype])
		{
			col2 = clamp (0.0000001,0.9999999,col2);
			col3 = clamp (0.0000001,0.9999999,col3);
		}
		float intensity=p[m].I;
		intensity = clamp (0.0000001,0.9999999,intensity);
		intensity *= dparams1->brightness[ptype];

		cu_color e;
		if (dparams1->col_vector[ptype])
		{
			e.r=col1*intensity;
			e.g=col2*intensity;
			e.b=col3*intensity;
		}
		else
		{
			if (ptype<types)
			{
				e = get_color(ptype, col1, mapSize, types,dmap_in,ptype_points_in);
				e.r *= intensity;
				e.g *= intensity;
				e.b *= intensity;
			}
			else
			{	e.r =e.g =e.b =0.0;}
		}

		const float sigma0 =0.584287628;
		float r = p[m].r;
		const float radsq = 2.25*r*r;
		const float stp = -0.5/(r*r*sigma0*sigma0);
		const float powtmp = pow(3.14159265358979323846264338327950288,1./3.);
		const float intens = -0.5/(2*sqrt(3.14159265358979323846264338327950288)*powtmp);
		e.r*=intens; e.g*=intens; e.b*=intens;

		const float posx=p[m].x, posy=p[m].y;
		unsigned int fpos =p[m].posInFragBuf;
		
		int minx=p[m].minx;
		int maxx=p[m].maxx;

		const float rfacr=dparams1->rfac*r;		
		int miny=(posy-rfacr+1);
		miny=max(miny,0);
		int maxy=(posy+rfacr+1);
		maxy = min(dparams1->yres, maxy);

		for (int x=minx; x<maxx; ++x)
		{
			float dxsq=(x-posx)*(x-posx);
			for (int y=miny; y<maxy; ++y)
			{
				float dsq = (y-posy)*(y-posy) + dxsq;
				if (dsq<radsq)
				{
					float att = pow((float)2.71828,(stp*dsq));
					buf[fpos].aR = att*e.r;
					buf[fpos].aG = att*e.g;
					buf[fpos].aB = att*e.b;
				}
				else
				{
					buf[fpos].aR =0.0;
					buf[fpos].aG =0.0;
					buf[fpos].aB =0.0;
				}

				fpos++;
			}
		}
	}
}

 
__kernel void k_render2
(__global cu_particle_splotch *p, int nP,__global
cu_fragment_AneqE *buf, float grayabsorb, int mapSize, int types,__global cu_param *dparams1,__global cu_color_map_entry *dmap_in,__global int *ptype_points_in)
{

	int m;
	m = get_global_id(0);
	if (m<nP) 
	{
		int ptype = p[m].type;
		float col1=p[m].e.r,col2=p[m].e.g,col3=p[m].e.b;
		clamp (0.0000001,0.9999999,col1);
		if (dparams1->col_vector[ptype])
		{
			col2 = clamp (0.0000001,0.9999999,col2);
			col3 = clamp (0.0000001,0.9999999,col3);
		}
		float intensity=p[m].I;
		intensity = clamp (0.0000001,0.9999999,intensity);
		intensity *= dparams1->brightness[ptype];

		cu_color e;
		if (dparams1->col_vector[ptype])
		{
			e.r=col1*intensity;
			e.g=col2*intensity;
			e.b=col3*intensity;
		}
		else
		{
			if (ptype<types)
			{
				e = get_color(ptype, col1, mapSize, types,dmap_in,ptype_points_in);
				e.r *= intensity;
				e.g *= intensity;
				e.b *= intensity;
			}
			else
			{	e.r =e.g =e.b =0.0f;}
		}

		const float powtmp = pow(3.14159265358979323846264338327950288,1./3.);
		const float sigma0 = powtmp/sqrt(2*3.14159265358979323846264338327950288);

		 float r = p[m].r;
		const float radsq = 2.25f*r*r;
		const float stp = -0.5/(r*r*sigma0*sigma0);

		cu_color q;

		q.r = e.r/(e.r+grayabsorb);
		q.g = e.g/(e.g+grayabsorb);
		q.b = e.b/(e.b+grayabsorb);

		const float intens = -0.5/(2*sqrt(3.14159265358979323846264338327950288)*powtmp);
		e.r*=intens; e.g*=intens; e.b*=intens;

		const float posx=p[m].x, posy=p[m].y;
		unsigned int fpos =p[m].posInFragBuf;

		int minx=p[m].minx;
		int maxx=p[m].maxx;

		const float rfacr=dparams1->rfac*r;		
		int miny=(int)(posy-rfacr+1);
		miny=max(miny,0);
		int maxy=(int)(posy+rfacr+1);
                maxy = min(dparams1->yres, maxy);

		for (int x=minx; x<maxx; ++x)
		{
			float dxsq=(x-posx)*(x-posx);
			for (int y=miny; y<maxy; ++y)
			{
				float dsq = (y-posy)*(y-posy) + dxsq;
				if (dsq<radsq)
				{
					float att =pow((float)2.71828,(stp*dsq));
					float expm1 = pow((float)2.71828,(att*e.r))-1.0;

					buf[fpos].aR = expm1;
					buf[fpos].qR = q.r;
					expm1= pow((float)2.71828,(att*e.g))-1.0;
					buf[fpos].aG = expm1;
					buf[fpos].qG = q.g;
					expm1= pow((float)2.71828,(att*e.b))-1.0;
					buf[fpos].aB = expm1;
					buf[fpos].qB = q.b;
				}
				else
				{
					buf[fpos].aR =0.0;
					buf[fpos].aG =0.0;
					buf[fpos].aB =0.0;
					buf[fpos].qR =1.0;
					buf[fpos].qG =1.0;
					buf[fpos].qB =1.0;
				}

				fpos++;
			} //y
		} //x
	}}
	
