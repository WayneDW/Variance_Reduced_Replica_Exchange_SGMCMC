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
#include "sgld.h"

#define MAX(x,y) (((x)>(y))?(x):(y))
#define MIN(x,y) (((x)<(y))?(x):(y))

/*
 * SGLD PARAMETERS
 */


void alloc_sgld_parameters(struct_sgld_parameters **sgld_parameters) {
	*sgld_parameters = (struct_sgld_parameters*)malloc((size_t)sizeof(struct_sgld_parameters)) ;
}

void set_sgld_parameters(struct_sgld_parameters *sgld_parameters) {
	sgld_parameters->eta = 0.00001 ;
	sgld_parameters->tau = 1.0 ;
	/*1:without replacement; other: with replacement*/
	sgld_parameters->subsample_type = 0 ;
}

void set_external_sgld_parameters(struct_sgld_parameters *sgld_parameters, int argc, char *argv[]) {
	int i ;
	for (i = 1; i < argc; i++)
		if (strcmp("-sgld_parameters->eta", argv[i]) == 0)
			sgld_parameters->eta = atof(argv[++i]) ;
		else if (strcmp("-sgld_parameters->tau", argv[i]) == 0)
			sgld_parameters->tau = atof(argv[++i]) ;
		else if (strcmp("-sgld_parameters->subsample_type", argv[i]) == 0)
			sgld_parameters->subsample_type = atoi(argv[++i]) ;
}

void print_sgld_parameters(struct_sgld_parameters *sgld_parameters) {
	printf("sgld_parameters->eta:  \t %f \n", sgld_parameters->eta) ;
	printf("sgld_parameters->tau:  \t %f \n", sgld_parameters->tau) ;
	printf("sgld_parameters->subsample_type:  \t %d \n", sgld_parameters->subsample_type) ;
}

void destroy_sgld_parameters(struct_sgld_parameters *sgld_parameters) {
	free((char*) sgld_parameters) ;
}

/*
 * SUB-DATA
 */

void alloc_sgld_subdata(struct_sgld_subdata **sgld_subdata, struct_data *data, int en_ysub) {

	int i ;

	*sgld_subdata = (struct_sgld_subdata*)malloc((size_t)sizeof(struct_sgld_subdata)) ;

	(*sgld_subdata)->en_ysub = en_ysub ;

	(*sgld_subdata)->I_ysub = ivector(1, en_ysub) ;

	for (i=1; i<=(*sgld_subdata)->en_ysub; i++)
		(*sgld_subdata)->I_ysub[i] = i ;

	(*sgld_subdata)->data = data ;
}

void sample_sgld_subdata(struct_sgld_subdata *sgld_subdata,
					struct_data *data,
					struct_sgld_parameters *sgld_parameters) {

	int i ;

	int en_ysub = sgld_subdata->en_ysub ;

	int en_y = data->en_y ;

	int *I_ysub = sgld_subdata->I_ysub ;

	int subsample_type = sgld_parameters->subsample_type ;

	/*subsample*/
	if ( subsample_type == 1) {
		/*sample with replacement*/
		for (i=1; i<=en_ysub; i++) I_ysub[i] = i ;
		orswrrng( &I_ysub[1], en_y, en_ysub) ;
	} else {
		/*sample without replacement*/
		for (i=1; i<=en_ysub; i++)   I_ysub[i] = integerrng( 1 , en_y) ;
	}

	/*not needed*/
	sgld_subdata->data = data ;
}

void print_sgld_subdata(struct_sgld_subdata *sgld_subdata) {

	int i ;

	int en_y = sgld_subdata->data->en_y ;

	int en_ysub = sgld_subdata->en_ysub ;

	int *I_ysub = sgld_subdata->I_ysub ;

	printf("sgld_subdata->data->en_y  \t %d , \n", en_y) ;

	printf("sgld_subdata->en_ysub  \t %d , \n", en_ysub) ;

	printf("sgld_subdata->I_ysub[%i : %i] \n", 1,MIN(en_ysub,50)) ;
	for (i=1; i<=MIN(en_ysub,50); i++) printf("%i, " , I_ysub[i]) ;
	printf(" \n") ;
}

void destroy_sgld_subdata(struct_sgld_subdata *sgld_subdata) {

	free_ivector((sgld_subdata->I_ysub), 1, sgld_subdata->en_ysub) ;

	free((char*) sgld_subdata) ;
}


/*
 * GRADIENT ESTIMATOR
 */

void comp_sgld_grad_log_lik_estimate(double * sgld_grad_log_lik_est,
                                            struct_sgld_subdata *sgld_subdata,
                                            struct_random_parameters* randpar,
                                            struct_fixed_parameters* fixpar) {

	int i ;

	int en_yall = sgld_subdata->data->en_y ;

	int en_ysub = sgld_subdata->en_ysub ;

	int *I_ysub = sgld_subdata->I_ysub ;

	double *yall = sgld_subdata->data->y ;

	double ysub_i ;

	(*sgld_grad_log_lik_est) = 0.0 ;

	for (i=1 ; i<=en_ysub ; i++) {

		ysub_i = yall[I_ysub[i]] ;

		(*sgld_grad_log_lik_est) = (*sgld_grad_log_lik_est)
                                    +comp_gradient_unit_log_lik( ysub_i,
                                                                randpar,
                                                                fixpar) ;
	}

	(*sgld_grad_log_lik_est) *= ((double)en_yall)/((double)en_ysub) ;
}

void comp_sgld_udpate_random_parameters(struct_sgld_subdata *sgld_subdata,
                                        struct_random_parameters* randpar,
                                        struct_fixed_parameters* fixpar,
                                        struct_sgld_parameters *sgld_parameters,
                                        double * sgld_grad_log_lik_est,
                                        double *sgld_grad_log_prior) {

	double eta = sgld_parameters->eta ;

	double tau = sgld_parameters->tau ;

	double xi ;

	comp_gradient_log_prior(sgld_grad_log_prior, randpar, fixpar) ;

	comp_sgld_grad_log_lik_estimate( sgld_grad_log_lik_est,
                                        sgld_subdata,
                                        randpar,
                                        fixpar) ;

	xi = normalrng() ;

	randpar->theta = randpar->theta
                            + eta * (*sgld_grad_log_lik_est)
                            + eta * (*sgld_grad_log_prior)
                            + sqrt( 2.0 * eta * tau ) * xi ;
}







