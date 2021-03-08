/*
 * Author: (C) 2020, Georgios Karagiannis
 * Assistant Professor in Statistics
 * Department of Mathematical Sciences, University of Durham
 * Stockton Road, Durham DH1 3LE, UK
 *
 * Telephone: +44 (0) 1913342718
 *
 * Email: georgios.karagiannis@durham.ac.uk
 *
 * Contact email: georgios.stats@gmail.com
 *
 * URL: http://www.maths.dur.ac.uk/~mffk55
 *
 * URL: https://github.com/georgios-stats
*/

/*
 * Deng, W., Feng, Q., Karagiannis, G., Lin, G., & Liang, F. (2021). 
 * Accelerating Convergence of Replica Exchange Stochastic Gradient MCMC via 
 * Variance Reduction. International Conference on Learning Representations 
 * (ICLR'21)
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "nrutil.h"
#include "logPDF.h"
#include "RNG.h"
#include "Bayesian_model.h"

#define MAX(x,y) (((x)>(y))?(x):(y))
#define MIN(x,y) (((x)<(y))?(x):(y))

/*
 * DATA
 * */

void alloc_data(struct_data **data, int en_y) {

	int i ;

	*data = (struct_data*)malloc((size_t)sizeof(struct_data)) ;

	(*data)->en_y = en_y ;

	(*data)->y = dvector(1, en_y) ;

	for (i=1; i<=(*data)->en_y; i++) (*data)->y[i] = 0.0 ;
}

void alloc_and_generate_data(struct_data **data, struct_fixed_parameters *fixpar, int en_y) {

	int i ;

	double un ;

	double weight_real = fixpar->weight_real ;

	double gamma_real = fixpar->gamma_real ;

	double mu_real = fixpar->mu_real ;

	double sig2_real = fixpar->sig2_real ;

	*data = (struct_data*)malloc((size_t)sizeof(struct_data)) ;

	(*data)->en_y = en_y ;

	(*data)->y = dvector(1, en_y) ;

	for (i=1; i<=(*data)->en_y; i++) (*data)->y[i] = 0.0 ;

	for (i=1 ; i<=(*data)->en_y ; i++) {

		un = uniformrng() ;

		if ( weight_real > un )
			(*data)->y[i] = mu_real+sqrt(sig2_real)*normalrng() ;
		else
			(*data)->y[i] = (gamma_real-mu_real)+sqrt(sig2_real)*normalrng() ;

	}
}

void destroy_data(struct_data *data) {

	free_dvector(data->y, 1, data->en_y) ;

	free((char*) data) ;
}

/*
 * RANDOM PARAMETERS
 * */

void alloc_random_parameters(struct_random_parameters **randpar) {

	*randpar = (struct_random_parameters*)malloc((size_t)sizeof(struct_random_parameters)) ;
}

void seed_random_parameters(struct_random_parameters *randpar) {

	randpar->dim_theta = 1 ;

	randpar->theta = 50.0*uniformrng() ;
}

void set_external_random_parameters(struct_random_parameters *randpar, int argc, char *argv[]) {
	int i ;
	for (i = 1; i < argc; i++)
		if (strcmp("-randpar->dim_theta", argv[i]) == 0)
			randpar->dim_theta = atoi(argv[++i]) ;
		else if (strcmp("-randpar->theta", argv[i]) == 0)
			randpar->theta = atof(argv[++i]) ;
}

void print_random_parameters(struct_random_parameters *randpar) {

	printf("randpar->dim_theta:  \t %d \n", randpar->dim_theta) ;

	printf("randpar->theta:  \t %f \n", randpar->theta) ;
}

void destroy_random_parameters(struct_random_parameters *randpar) {
	free((char*) randpar) ;
}

void copy_random_parameters(struct_random_parameters *randpar_from,
								struct_random_parameters *randpar_to) {
	randpar_to->dim_theta = randpar_from->dim_theta ;
	randpar_to->theta = randpar_from->theta ;
}

void swap_random_parameters(struct_random_parameters *randpar_1,
                            struct_random_parameters *randpar_2,
                            struct_random_parameters *randpar_aux) {

    copy_random_parameters(randpar_1, randpar_aux) ;

    copy_random_parameters(randpar_2, randpar_1) ;

    copy_random_parameters(randpar_aux, randpar_2) ;
}

/*
 * FIXED PARAMETERS
 * */

void alloc_fixed_parameters(struct_fixed_parameters **fixpar) {

	*fixpar = (struct_fixed_parameters*)malloc((size_t)sizeof(struct_fixed_parameters)) ;
}

void initialise_fixed_parameters(struct_fixed_parameters *fixpar) {

	fixpar->mu_0 = 0.0 ;

	fixpar->sig2_0 = 100.0;

	fixpar->weight_real = 0.5 ;

	fixpar->mu_real = -5.0 ;

	fixpar->sig2_real = 25.0 ;

	fixpar->gamma_real = 20.0;
}

void set_external_fixed_parameters(struct_fixed_parameters *fixpar, int argc, char *argv[]) {
	int i ;
	for (i = 1; i < argc; i++)
		if (strcmp("-fixpar->mu_0", argv[i]) == 0)
			fixpar->mu_0 = atof(argv[++i]) ;
		else if (strcmp("-fixpar->sig2_0", argv[i]) == 0)
			fixpar->sig2_0 = atof(argv[++i]) ;
		else if (strcmp("-fixpar->weight_real", argv[i]) == 0)
			fixpar->weight_real = atof(argv[++i]) ;
		else if (strcmp("-fixpar->mu_real", argv[i]) == 0)
			fixpar->mu_real = atof(argv[++i]) ;
		else if (strcmp("-fixpar->sig2_real", argv[i]) == 0)
			fixpar->sig2_real = atof(argv[++i]) ;
		else if (strcmp("-fixpar->gamma_real", argv[i]) == 0)
			fixpar->gamma_real = atof(argv[++i]) ;
}

void print_fixed_parameters(struct_fixed_parameters *fixpar) {

	printf("fixpar->mu_0:  \t %f \n", fixpar->mu_0) ;

	printf("fixpar->sig2_0:  \t %f  \n", fixpar->sig2_0) ;

	printf("fixpar->weight_real:  \t %f  \n", fixpar->weight_real) ;

	printf("fixpar->mu_real:  \t %f  \n", fixpar->mu_real) ;

	printf("fixpar->sig2_real:  \t %f  \n", fixpar->sig2_real) ;

	printf("fixpar->gamma_real:  \t %f  \n", fixpar->gamma_real) ;
}

void destroy_fixed_parameters(struct_fixed_parameters *fixpar) {

	free((char*) fixpar) ;
}


/*
 * PRIORS
 * */

double comp_log_prior(struct_random_parameters* randpar,
						struct_fixed_parameters* fixpar) {

	double mu_0 = fixpar->mu_0 ;

	double sig2_0 = fixpar->sig2_0 ;

	double theta = randpar->theta ;

	double fval ;

	fval = normal_logpdf(theta, mu_0, sig2_0) ;

	return (fval) ;
}

double comp_prior(struct_random_parameters* randpar,
					struct_fixed_parameters* fixpar) {

	double fval = exp( comp_log_prior(randpar, fixpar) )  ;

	return (fval) ;
}

void comp_gradient_log_prior(double* gradient_log_prior,
                            struct_random_parameters* randpar,
                            struct_fixed_parameters* fixpar) {

	double mu_0 = fixpar->mu_0 ;

	double sig2_0 = fixpar->sig2_0 ;

	double theta = randpar->theta ;

	(*gradient_log_prior) = -(theta-mu_0)/sig2_0 ;
}

/*
 * LIKELIHOOD
 * */

double comp_unit_log_lik(double yi, struct_random_parameters* randpar,
							struct_fixed_parameters* fixpar) {

	double theta = randpar->theta ;

	double weight_real = fixpar->weight_real ;

	double gamma_real = fixpar->gamma_real ;

	double sig2_real = fixpar->sig2_real ;

	double logf1 ;

	double logf2 ;

	double max_logf1_logf2 ;

	double fval ;

	logf1 = normal_logpdf(yi, theta, sig2_real) + log(weight_real) ;

	logf2 = normal_logpdf(yi, gamma_real-theta, sig2_real) +  log(1.0-weight_real) ;

	max_logf1_logf2 = MAX( logf1 , logf2 ) ;

	fval = max_logf1_logf2 +
			log(
                            exp( logf1 - max_logf1_logf2 )
                            + exp( logf2 - max_logf1_logf2 )
                            ) ;

	return (fval) ;
}

double comp_unit_lik(double yi, struct_random_parameters* randpar,
		struct_fixed_parameters* fixpar) {

	double fval ;

	fval = exp( comp_unit_log_lik( yi, randpar, fixpar) ) ;

	return (fval) ;
}


double comp_gradient_unit_log_lik(double yi, struct_random_parameters* randpar,
                                    struct_fixed_parameters* fixpar) {

	double EPS ;
	double theta ;
	double temp ;
	double theta_pl ;
	double h ;
	double fxh ;
	double fx ;
	double fval;
	struct_random_parameters* randpar_new ;

	EPS = 1.0e-4 ;

	theta = randpar->theta ;

	temp = theta ;

	h = EPS*fabs(temp) ;

	if (h == 0.0) h = EPS ;

	theta_pl = temp + h ;

	alloc_random_parameters(&randpar_new)  ;

	seed_random_parameters(randpar_new) ;

	copy_random_parameters(randpar, randpar_new) ;

	randpar_new->theta = theta_pl ;

	h = theta_pl - temp ;

	fx = comp_unit_log_lik( yi,  randpar,  fixpar) ;

	fxh = comp_unit_log_lik( yi,  randpar_new,  fixpar) ;

	fval = ( fxh - fx ) / h ;

	destroy_random_parameters(randpar_new) ;

	return (fval) ;
}


/*
 * POSTERIOR
 * */

double comp_log_posterior(struct_data* data, struct_random_parameters* randpar,
							struct_fixed_parameters* fixpar) {

	int i ;

	double yi ;

	int en_y = data->en_y ;

	double fval ;

	fval = comp_log_prior( randpar,  fixpar) ;

	for ( i=1 ; i<=en_y ; i++) {

		yi = data->y[i] ;

		fval +=  comp_unit_log_lik( yi, randpar, fixpar) ;
	}

	return (fval) ;
}



