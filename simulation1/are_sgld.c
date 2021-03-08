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
#include "are_sgld.h"

#define MAX(x,y) (((x)>(y))?(x):(y))
#define MIN(x,y) (((x)<(y))?(x):(y))

/*
 * SGLD PARAMETERS
 */

void alloc_are_sgld_parameters(struct_are_sgld_parameters **are_sgld_parameters) {
	*are_sgld_parameters = (struct_are_sgld_parameters*)malloc((size_t)sizeof(struct_are_sgld_parameters)) ;
}

void set_are_sgld_parameters(struct_are_sgld_parameters *are_sgld_parameters) {
	are_sgld_parameters->eta_1 = 0.00001 ;
	are_sgld_parameters->tau_1 = 1.0 ;
	are_sgld_parameters->eta_2 = 0.00001 ;
	are_sgld_parameters->tau_2 = 100.0 ;
	/*1:without replacement; other: with replacement*/
	are_sgld_parameters->subsample_type = 0 ;
	are_sgld_parameters->sig2hat_rep = 100 ; //
	are_sgld_parameters->gain_sig2hat_t0 = 1000 ; //
	are_sgld_parameters->gain_sig2hat_c0 = 0.2 ; //
	are_sgld_parameters->Fscl = 100.0 ; //
	are_sgld_parameters->sig2hat_update_rate = 200 ; //
}

void set_external_are_sgld_parameters(struct_are_sgld_parameters *are_sgld_parameters, int argc, char *argv[]) {
	int i ;
	for (i = 1; i < argc; i++)
            if (strcmp("-are_sgld_parameters->eta_1", argv[i]) == 0)
                    are_sgld_parameters->eta_1 = atof(argv[++i]) ;
            else if (strcmp("-are_sgld_parameters->eta_2", argv[i]) == 0)
                            are_sgld_parameters->eta_2 = atof(argv[++i]) ;
            else if (strcmp("-are_sgld_parameters->tau_1", argv[i]) == 0)
                    are_sgld_parameters->tau_1 = atof(argv[++i]) ;
            else if (strcmp("-are_sgld_parameters->tau_2", argv[i]) == 0)
                    are_sgld_parameters->tau_2 = atof(argv[++i]) ;
            else if (strcmp("-are_sgld_parameters->subsample_type", argv[i]) == 0)
                    are_sgld_parameters->subsample_type = atoi(argv[++i]) ;
            else if (strcmp("-are_sgld_parameters->sig2hat_rep", argv[i]) == 0)
                    are_sgld_parameters->sig2hat_rep = atoi(argv[++i]) ;
            else if (strcmp("-are_sgld_parameters->gain_sig2hat_t0", argv[i]) == 0)
                    are_sgld_parameters->gain_sig2hat_t0 = atoi(argv[++i]) ;
            else if (strcmp("-are_sgld_parameters->gain_sig2hat_c0", argv[i]) == 0)
                    are_sgld_parameters->gain_sig2hat_c0 = atof(argv[++i]) ;
            else if (strcmp("-are_sgld_parameters->Fscl", argv[i]) == 0)
                    are_sgld_parameters->Fscl = atof(argv[++i]) ;
            else if (strcmp("-are_sgld_parameters->sig2hat_update_rate", argv[i]) == 0)
                    are_sgld_parameters->sig2hat_update_rate = atoi(argv[++i]) ;
}

void print_are_sgld_parameters(struct_are_sgld_parameters *are_sgld_parameters) {
	printf("are_sgld_parameters->eta_1:  \t %f \n", are_sgld_parameters->eta_1) ;
	printf("are_sgld_parameters->eta_2:  \t %f \n", are_sgld_parameters->eta_2) ;
	printf("are_sgld_parameters->tau_1:  \t %f \n", are_sgld_parameters->tau_1) ;
	printf("are_sgld_parameters->tau_2:  \t %f \n", are_sgld_parameters->tau_2) ;
	printf("are_sgld_parameters->subsample_type:  \t %d \n", are_sgld_parameters->subsample_type) ;
	printf("are_sgld_parameters->sig2hat_rep:  \t %d \n", are_sgld_parameters->sig2hat_rep) ;
	printf("are_sgld_parameters->gain_sig2hat_t0:  \t %d \n", are_sgld_parameters->gain_sig2hat_t0) ;
	printf("are_sgld_parameters->gain_sig2hat_c0:  \t %f \n", are_sgld_parameters->gain_sig2hat_c0) ;
	printf("are_sgld_parameters->Fscl:  \t %f \n", are_sgld_parameters->Fscl) ;
	printf("are_sgld_parameters->sig2hat_update_rate:  \t %d \n", are_sgld_parameters->sig2hat_update_rate) ;
}

void destroy_are_sgld_parameters(struct_are_sgld_parameters *are_sgld_parameters) {
	free((char*) are_sgld_parameters) ;
}

/*
 * SUB-DATA
 */

void alloc_are_sgld_subdata(struct_are_sgld_subdata **are_sgld_subdata, struct_data *data, int en_ysub) {

	int i ;

	*are_sgld_subdata = (struct_are_sgld_subdata*)malloc((size_t)sizeof(struct_are_sgld_subdata)) ;

	(*are_sgld_subdata)->en_ysub = en_ysub ;

	(*are_sgld_subdata)->I_ysub = ivector(1, en_ysub) ;

	for (i=1; i<=(*are_sgld_subdata)->en_ysub; i++)
		(*are_sgld_subdata)->I_ysub[i] = i ;

	(*are_sgld_subdata)->data = data ;
}

void sample_are_sgld_subdata(struct_are_sgld_subdata *are_sgld_subdata,
					struct_data *data,
					struct_are_sgld_parameters *are_sgld_parameters) {

	int i ;

	int en_ysub = are_sgld_subdata->en_ysub ;

	int en_y = data->en_y ;

	int *I_ysub = are_sgld_subdata->I_ysub ;

	int subsample_type = are_sgld_parameters->subsample_type ;


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
	are_sgld_subdata->data = data ;
}

void print_are_sgld_subdata(struct_are_sgld_subdata *are_sgld_subdata) {

	int i ;

	int en_y = are_sgld_subdata->data->en_y ;

	int en_ysub = are_sgld_subdata->en_ysub ;

	int *I_ysub = are_sgld_subdata->I_ysub ;

	printf("are_sgld_subdata->data->en_y  \t %d , \n", en_y) ;

	printf("are_sgld_subdata->en_ysub  \t %d , \n", en_ysub) ;

	printf("are_sgld_subdata->I_ysub[%i : %i] \n", 1,MIN(en_ysub,50)) ;
	for (i=1; i<=MIN(en_ysub,50); i++) printf("%i, " , I_ysub[i]) ;
	printf(" \n") ;
}

void destroy_are_sgld_subdata(struct_are_sgld_subdata *are_sgld_subdata) {

	free_ivector((are_sgld_subdata->I_ysub), 1, are_sgld_subdata->en_ysub) ;

	free((char*) are_sgld_subdata) ;
}


/*
 * GRADIENT ESTIMATOR
 */

void comp_are_sgld_grad_log_lik_estimate(double * are_sgld_grad_log_lik_est,
                                        struct_are_sgld_subdata *are_sgld_subdata,
                                        struct_random_parameters* randpar,
                                        struct_fixed_parameters* fixpar) {

	int i ;

	int en_yall = are_sgld_subdata->data->en_y ;

	int en_ysub = are_sgld_subdata->en_ysub ;

	int *I_ysub = are_sgld_subdata->I_ysub ;

	double *yall = are_sgld_subdata->data->y ;

	double ysub_i ;

	(*are_sgld_grad_log_lik_est) = 0.0 ;

	for (i=1 ; i<=en_ysub ; i++) {

		ysub_i = yall[I_ysub[i]] ;

		(*are_sgld_grad_log_lik_est) = (*are_sgld_grad_log_lik_est)
                                                +comp_gradient_unit_log_lik( ysub_i,
                                                                            randpar,
                                                                            fixpar) ;
	}

	(*are_sgld_grad_log_lik_est) *= ((double)en_yall)/((double)en_ysub) ;
}

void comp_are_sgld_udpate_random_parameters(struct_are_sgld_subdata *are_sgld_subdata,
                                            struct_random_parameters* randpar_1,
                                            struct_random_parameters* randpar_2,
                                            struct_fixed_parameters* fixpar,
                                            struct_are_sgld_parameters *are_sgld_parameters,
                                            double * are_sgld_grad_log_lik_est_1,
                                            double * are_sgld_grad_log_lik_est_2,
                                            double * are_sgld_grad_log_prior_1,
                                            double * are_sgld_grad_log_prior_2) {

	double eta_1 = are_sgld_parameters->eta_1 ;

	double eta_2 = are_sgld_parameters->eta_2 ;

	double tau_1 = are_sgld_parameters->tau_1 ;

	double tau_2 = are_sgld_parameters->tau_2 ;

	double xi ;

	/*Component 1*/

	comp_gradient_log_prior(are_sgld_grad_log_prior_1, randpar_1, fixpar) ;

	comp_are_sgld_grad_log_lik_estimate( are_sgld_grad_log_lik_est_1,
                                                are_sgld_subdata,
                                                randpar_1,
                                                fixpar) ;

	xi = normalrng() ;

	randpar_1->theta = randpar_1->theta
                                    + eta_1 * (*are_sgld_grad_log_lik_est_1)
                                    + eta_1 * (*are_sgld_grad_log_prior_1)
                                    + sqrt( 2.0 * eta_1 * tau_1 ) * xi ;

	/*Component 2*/

	comp_gradient_log_prior(are_sgld_grad_log_prior_2, randpar_2, fixpar) ;

	comp_are_sgld_grad_log_lik_estimate( are_sgld_grad_log_lik_est_2,
                                                are_sgld_subdata,
                                                randpar_2,
                                                fixpar) ;

	xi = normalrng() ;

	randpar_2->theta = randpar_2->theta
                            + eta_2 * (*are_sgld_grad_log_lik_est_2)
                            + eta_2 * (*are_sgld_grad_log_prior_2)
                            + sqrt( 2.0 * eta_2 * tau_2 ) * xi ;
}



/*
 * Compute log likelihood estimate
 */



double  comp_are_sgld_log_lik_estimate(struct_are_sgld_subdata *are_sgld_subdata,
                                        struct_random_parameters* randpar,
                                        struct_fixed_parameters* fixpar) {

	int i ;

	int en_yall = are_sgld_subdata->data->en_y ;

	int en_ysub = are_sgld_subdata->en_ysub ;

	int *I_ysub = are_sgld_subdata->I_ysub ;

	double *yall = are_sgld_subdata->data->y ;

	double ysub_i ;

	double loglik_est ;

	loglik_est = 0.0 ;

	for (i=1 ; i<=en_ysub ; i++) {

		ysub_i = yall[I_ysub[i]] ;

		loglik_est = loglik_est + comp_unit_log_lik(ysub_i, randpar, fixpar) ;
	}

	loglik_est = ((double)en_yall)/((double)en_ysub) * loglik_est ;

	return (loglik_est) ;
}



/*
 * Compute the stochastisity correction term
 */

double comp_are_sgld_gain_sig2hat(int iter,
				struct_are_sgld_parameters *are_sgld_parameters) {

	double gt ;

	double t_0 = are_sgld_parameters->gain_sig2hat_t0 ;

	double c_0 = are_sgld_parameters->gain_sig2hat_c0 ;
#if 0
	gt = (double)t_0 / fmax ( (double)t_0, (double)iter) ;

	gt = fmax( gt, (double) c_0  ) ;
#else
	gt = are_sgld_parameters->gain_sig2hat_c0 ;
#endif
	return ( gt ) ;
}

double comp_are_sgld_sig2tilde(struct_are_sgld_subdata *are_sgld_subdata,
                                struct_random_parameters* randpar_1,
                                struct_random_parameters* randpar_2,
                                struct_fixed_parameters* fixpar,
                                struct_are_sgld_parameters *are_sgld_parameters,
                                struct_are_sgld_subdata *are_sgld_subdata_aux) {

	int i ;

	double nrep = are_sgld_parameters->sig2hat_rep ;

	double loglik_est_1 ;
	double loglik_est_2 ;

	double sxx ;
	double sx ;

	double sig2tilde ;

	sxx = 0.0 ;
	sx = 0.0 ;

	are_sgld_subdata_aux->data = are_sgld_subdata->data ;

	are_sgld_subdata_aux->en_ysub = are_sgld_subdata->en_ysub ;

	for ( i=1; i<=nrep; i++) {

            sample_are_sgld_subdata( are_sgld_subdata_aux,
                                    are_sgld_subdata->data,
                                    are_sgld_parameters) ;

            loglik_est_1 = comp_are_sgld_log_lik_estimate(are_sgld_subdata_aux,
                                                        randpar_1,
                                                        fixpar) ;

            loglik_est_2 = comp_are_sgld_log_lik_estimate(are_sgld_subdata_aux,
                                                        randpar_2,
                                                        fixpar) ;

            sx += (loglik_est_1-loglik_est_2) ;
            sxx += (loglik_est_1-loglik_est_2)*(loglik_est_1-loglik_est_2) ;
	}

	sig2tilde = sxx/nrep - (sx/nrep)*(sx/nrep) ;
	
	sig2tilde = fabs(sig2tilde) ;

	return	(sig2tilde) ;
}

void comp_are_sgld_sig2hat(double *sig2hat, double sig2tilde, double gt) {

	(*sig2hat) = gt*sig2tilde + (1.0-gt)* (*sig2hat) ;
}


/*
 * Compute the Metropolis Hastings
 */

void comp_are_sgld_acceptance_ratio(double *logMHAccRat, double *logMHAccRatCorr,
                                    struct_are_sgld_subdata *are_sgld_subdata,
                                    struct_random_parameters* randpar_1,
                                    struct_random_parameters* randpar_2,
                                    struct_fixed_parameters* fixpar,
                                    struct_are_sgld_parameters *are_sgld_parameters,
                                    double sig2hat) {

	double tau_1 = are_sgld_parameters->tau_1 ;

	double tau_2 = are_sgld_parameters->tau_2 ;

	double Fscl = are_sgld_parameters->Fscl ;

	double Energy_1 ;

	double Energy_2 ;

	int i ;

	int en_ysub = are_sgld_subdata->en_ysub ;

	int *I_ysub = are_sgld_subdata->I_ysub ;

	double *yall = are_sgld_subdata->data->y ;

	double ysub_i ;

	Energy_1 = - comp_log_prior(randpar_1, fixpar) ;

	Energy_2 = - comp_log_prior(randpar_2, fixpar) ;

	for (i=1 ; i<=en_ysub ; i++ ) {

		ysub_i = yall[I_ysub[i]] ;

		Energy_1 += comp_unit_log_lik(ysub_i, randpar_1, fixpar) ;

		Energy_2 += comp_unit_log_lik(ysub_i, randpar_2, fixpar) ;
	}

	(*logMHAccRat) = (1/tau_2-1/tau_1)*(Energy_2-Energy_1) ;

	(*logMHAccRatCorr) = (*logMHAccRat)
                            -(1/tau_2-1/tau_1)*(1/tau_2-1/tau_1)*sig2hat/Fscl ;
}




















