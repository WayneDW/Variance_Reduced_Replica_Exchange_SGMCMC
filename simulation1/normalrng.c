/*
 * Copyrigtht 2014 Georgios Karagiannis
 *
 * This file is part of PISAA_Rastrigin.
 *
 * PISAA_Rastrigin is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation version 2 of the License.
 *
 * PISAA_Rastrigin is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PISAA_Rastrigin.  If not, see <http://www.gnu.org/licenses/>.
*/

/*
 * Georgios Karagiannis 
 * Postdoctoral research associate
 * Department of Mathematics, Purdue University
 * 150 N. University Street
 * West Lafayette, IN 47907-2067, USA
 *
 * Telephone: +1 (765) 496-1007
 *
 * Email: gkaragia@purdue.edu
 *
 * Contact email: georgios.stats@gmail.com
*/


#include <math.h>

#include "RNG.h"

/* 
NORMAL RANDOM VARIABLE 
BY POLAR COORDINATES 
*/

double normalrng(void)
{
	static int normalrng_IR = 0 ;
	static double normalrng_AN ;

	double c ;
	double w ;
	double v1 ;
	double v2 ;
 
    double rnd ;

	if  (normalrng_IR == 0)
    {
		do
		{
			v1 = 2.0*uniformrng() - 1.0 ;
			v2 = 2.0*uniformrng() - 1.0 ;
			w = v1*v1 + v2*v2 ;
		} while (w >= 1.0) ;
		c = sqrt( -2.0*log(w)/w ) ;
		normalrng_AN = v1*c ;
		normalrng_IR = 1 ;
		rnd = v2*c ;
	}
	else
	{
		normalrng_IR = 0 ;
		rnd = normalrng_AN ;
	}

    return rnd ;

}

/* 
NORMAL RANDOM VARIABLE 
BY POLAR COORDINATES
*/

double normalrng_polar(void)
{
	static int normalrng_IR = 0 ;
	static double normalrng_AN ;

	double c ;
	double w ;
	double v1 ;
	double v2 ;

    double rnd ;

	if  (normalrng_IR == 0)
    {
		do
		{
			v1 = 2.0*uniformrng() - 1.0 ;
			v2 = 2.0*uniformrng() - 1.0 ;
			w = v1*v1 + v2*v2 ;
		} while (w >= 1.0) ;
		c = sqrt(-2.0*log(w)/w) ;
		normalrng_AN = v1*c ;
		normalrng_IR = 1 ;
		rnd = v2*c ;
	}
	else
	{
		normalrng_IR = 0 ;
		rnd = normalrng_AN ;
	}

    return rnd ;

}

/*
NORMAL RANDOM VARIABLE
BY RATIO OF UNIFORMS 
*/

double normalrng_ratio(void)
{

	double rnd ;
	double u ;
	double v ;
	double z ;
	int Q ;

	do
	{

		u = uniformrng() ;

		v = 0.857763884960707*(2.0*uniformrng()-1.0) ;

		rnd = v / u ;

		z = 0.25*rnd*rnd ;

		if ( z < 1.0-u ) break ;

		Q = ( z > (0.259240260645892/u+0.35) ) || ( z > -log(u) ) ;

	} while ( Q ) ;

	return rnd ;

}


