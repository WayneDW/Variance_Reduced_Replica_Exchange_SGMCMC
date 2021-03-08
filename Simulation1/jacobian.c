#include <math.h>

#include "nrutil.h"

#define EPS 1.0e-4 //Approximate square root of the machine precision.

void fdjac(double *df, int n, double *x, double (*func)(int, double*) ) {

	// df, x and func are zero based ...

	int j;

	double h;

	double temp;

	double fxh;

	double fx;

	fx = (*func)(n,x) ;

	for (j=0; j<=(n-1); j++) {

		temp = x[j] ;

		h = EPS*fabs(temp) ;

		if (h == 0.0) h = EPS ;

		x[j] = temp + h ; //Trick to reduce finite precision error.

		h = x[j] - temp ;

		fxh = (*func)(n,x) ;

		x[j] = temp ;

		df[j] = ( fxh - fx ) / h ;
	}
}




