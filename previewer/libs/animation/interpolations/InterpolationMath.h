/*
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 *
 * 		Tim Dykes and Ian Cant
 * 		University of Portsmouth
 *
 */

#ifndef SPLOTCH_PREVIEWER_LIBS_ANIMATION_INTERPOLATIONS_INTERPOLATION_MATH
#define SPLOTCH_PREVIEWER_LIBS_ANIMATION_INTERPOLATIONS_INTERPOLATION_MATH

namespace previewer
{
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//												Interpolation method templates												    //
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////



	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Linear interpolation
	// Takes 2 control points and interval parameter mu
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	//templates for input data type and interval type
	template <typename dataType, typename paramType>

	dataType LinearInterpolate(paramType mu, dataType d1, dataType d2)
	{
		return( (d1 * (1-mu)) + (d2 * mu) );
	}

	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Cubic polynomial interpolation
	// Takes 4 control points and interval parameter mu
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	template <typename dataType, typename paramType>

	dataType CubicInterpolate(paramType mu, dataType d0, dataType d1, dataType d2, dataType d3)
	{
		dataType coef0, coef1, coef2, coef3;

		paramType mu2 = mu * mu;
		paramType mu3 = mu2 * mu;

		coef0 = d3 - d2 - d0 + d1;
		coef1 = d0 - d1 - coef0;
		coef2 = d2 - d0;
		coef3 = d1;

		return( (coef0 * mu3) + (coef1 * mu2) + (coef2 * mu) + coef3 );
	}


	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Hermite-form Cubic interpolation with Kochanek-Bartels parameters tension, bias, continuity
	// This version will only work for interpolation with constant intervals
	// parameters data0, data1, data2, data3, mu, tension, bias, continuity
	// interpolation occurs between data1 and data2
	// Catmull-Rom curve: t = b = c = 0
	// Cardinal curver: b = c = 0, t = 1
	// d data
	// ct control tangent
	// mu = interval (constant intervals)
	//
	// d(mu) = (2*mu3 - 3*mu2 + 1)d1 + (mu3 - 2*mu2 + mu)ct1 + (-2*mu3 + 3*mu2)d2 + (mu3 - mu2)ct2
	//
	// ct1 = ( ((1-t)*(1+b)*(1+c)) / 2*(mu_ - mu ) * (d1 - d0) + ( ((1-t)*(1-b)*(1-c)) / 2*(mu - _mu) ) * (d2 - d1)
	// ct2 = ( ((1-t)*(1+b)*(1-c)) / 2*(mu__ - mu_) ) * (d2 - d1) + ( ((1-t)*(1-b)*(1-c)) / 2*(mu_ - mu) ) * (d3 - d2)
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	template <typename dataType, typename paramType>

	dataType CHKBInterpolate(paramType mu, dataType d0, dataType d1, dataType d2, dataType d3, paramType t, paramType b, paramType c)
	{

		//coefficients, control tangents for data1 and data2
		paramType coef0, coef1, coef2, coef3;
		dataType ct1, ct2;
		paramType mu2 = mu * mu;
		paramType mu3 = mu * mu2;

		coef0 = (2*mu3) - (3*mu2) + 1;
		coef1 = mu3 - (2*mu2) + mu;
		coef2 = (-2*mu3) + (3*mu2);
		coef3 = mu3 - mu2;

		//note: templated data comes first in multiplications to avoid errors with custom datatypes using lhs-only operators
		ct1  = (d1 - d0) * ( ((1-t)*(1+b)*(1+c)) /2 );
		ct1 += (d2 - d1) * ( ((1-t)*(1-b)*(1-c)) /2 );

		ct2  = (d2 - d1) * ( ((1-t)*(1+b)*(1-c)) /2 );
		ct2 += (d3 - d2) * ( ((1-t)*(1-b)*(1+c)) /2 );

		return( (d1*coef0) + (ct1*coef1) + (d2*coef2) + (ct2*coef3) );

	}

}

#endif