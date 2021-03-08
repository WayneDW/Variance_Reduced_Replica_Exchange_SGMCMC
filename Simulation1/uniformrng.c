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


/* declare the headers */

#include <stdlib.h>
#include <math.h>

#include "RNG.h"

/* initializes a seed */

void setseedrng(unsigned long s)
{
	srand( s ) ;
}

/* DEFAULT : generates a random number on (0,1)-real-interval */

double uniformrng(void)
{
	double rnd ;

	do {
		rnd = (double) ( rand() / ( RAND_MAX + 1.0 ) ) ;
	} while ( rnd == 0.0 || rnd == 1.0 ) ;

	return rnd ;
}

/* DEFAULT : generates a random number on [a,b]-real-interval */

int integerrng(int a, int b)
{
	return a +floor(uniformrng()*(b-a+1)) ;
}












