/*
 * Georgios Karagiannis (C) 2015
 * Postdoctoral research associate
 * Department of Mathematics, Purdue University
 * 150 N. University Street
 * West Lafayette, IN 47907-2067, USA
 *
 * Telephone: +1 765 496-1007
 *
 * Email: gkaragia@purdue.edu
 *
 * Contact email: georgios.stats@gmail.com
*/


#include <math.h>

double normal_logpdf(double x, double mu, double sigma2) {

	double myPI = 3.141592653589793 ;

	double logpdf ;

	logpdf = -0.5*(log(2.0) +log(myPI) +log(sigma2) +(x-mu)*(x-mu)/sigma2 ) ;

	return logpdf ;
}



