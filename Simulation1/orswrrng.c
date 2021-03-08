/*
 * Copyrigtht 2014 Georgios Karagiannis
 *
 * This file is part of RNGC.
 *
 * RNGC is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation version 2 of the License.
 *
 * RNGC is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with RNGC.  If not, see <http://www.gnu.org/licenses/>.
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



/*
 * Ordered Random Selection Without Replacement
 * A. F. Bissell
 * Journal of the Royal Statistical Society. Series C (Applied Statistics),
 * Vol. 35, No. 1 (1986), pp. 73-75
 * Published by: Blackwell Publishing for the Royal Statistical Society
 *
 * DEPENDS ON :
 *              uniformrng.c
 */

#include <math.h>
#include <stdio.h>

#include "RNG.h"

void abort(void) ;

void orswrrng(int *v, int N, int m) {

	/*
	 * Arrays: v are zero-based
	 * */

   int r ;
   int NN, rr ;
   int i ;
   double u, pp ;

   if ( N < m ) { printf("ERROR::rngc::orswrrng.c") ; abort(); }

   r = N-m ;
   NN = N ;
   rr = r ;

   for (i=1; i<=m; i++) {

	   u = uniformrng() ;

	   pp = ((double) rr)/NN ;

	   while (pp > u) {

		   NN = NN-1 ;

		   rr = rr-1 ;

		   pp = pp*rr/((double)NN) ;

	   }

	   v[i-1] = N-NN+1 ;

	   NN = NN-1 ;

   }

}



