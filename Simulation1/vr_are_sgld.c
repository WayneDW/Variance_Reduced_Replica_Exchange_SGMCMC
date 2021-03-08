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
#include"Bayesian_model.h"
#include "svrg_are_sgld.h"

#define MAX(x,y) (((x)>(y))?(x):(y))
#define MIN(x,y) (((x)<(y))?(x):(y))

/*
 * SVRG ARE SGLD PARAMETERS
 */

void alloc_svrg_are_sgld_parameters(struct_svrg_are_sgld_parameters **svrg_are_sgld_parameters) {
	*svrg_are_sgld_parameters = (struct_svrg_are_sgld_parameters*)malloc((size_t)sizeof(struct_svrg_are_sgld_parameters)) ;
}

void set_svrg_are_sgld_parameters(struct_svrg_are_sgld_parameters *svrg_are_sgld_parameters) {
	svrg_are_sgld_parameters->eta_1 = 0.00001 ;
	svrg_are_sgld_parameters->tau_1 = 1.0 ;
	svrg_are_sgld_parameters->eta_2 = 0.00001 ;
	svrg_are_sgld_parameters->tau_2 = 1000.0 ;
	/*1:without replacement; other: with replacement*/
	svrg_are_sgld_parameters->subsample_type = 0 ;
	svrg_are_sgld_parameters->sig2hat_rep = 200 ; //
	svrg_are_sgld_parameters->gain_sig2hat_t0 = 1000 ; //
	svrg_are_sgld_parameters->gain_sig2hat_c0 = 0.2 ; //
	svrg_are_sgld_parameters->Fscl = 1.0 ; //
	svrg_are_sgld_parameters->CV_update_rate = 50 ; //
	svrg_are_sgld_parameters->sig2hat_update_rate = 100 ; //
}

void set_external_svrg_are_sgld_parameters(struct_svrg_are_sgld_parameters *svrg_are_sgld_parameters, int argc, char *argv[]) {
	int i ;
	for (i = 1; i < argc; i++)
		if (strcmp("-svrg_are_sgld_parameters->eta_1", argv[i]) == 0)
			svrg_are_sgld_parameters->eta_1 = atof(argv[++i]) ;
		else if (strcmp("-svrg_are_sgld_parameters->eta_2", argv[i]) == 0)
				svrg_are_sgld_parameters->eta_2 = atof(argv[++i]) ;
		else if (strcmp("-svrg_are_sgld_parameters->tau_1", argv[i]) == 0)
			svrg_are_sgld_parameters->tau_1 = atof(argv[++i]) ;
		else if (strcmp("-svrg_are_sgld_parameters->tau_2", argv[i]) == 0)
			svrg_are_sgld_parameters->tau_2 = atof(argv[++i]) ;
		else if (strcmp("-svrg_are_sgld_parameters->subsample_type", argv[i]) == 0)
			svrg_are_sgld_parameters->subsample_type = atoi(argv[++i]) ;
		else if (strcmp("-svrg_are_sgld_parameters->sig2hat_rep", argv[i]) == 0)
			svrg_are_sgld_parameters->sig2hat_rep = atoi(argv[++i]) ;
		else if (strcmp("-svrg_are_sgld_parameters->gain_sig2hat_t0", argv[i]) == 0)
			svrg_are_sgld_parameters->gain_sig2hat_t0 = atoi(argv[++i]) ;
		else if (strcmp("-svrg_are_sgld_parameters->gain_sig2hat_c0", argv[i]) == 0)
			svrg_are_sgld_parameters->gain_sig2hat_c0 = atof(argv[++i]) ;
		else if (strcmp("-svrg_are_sgld_parameters->Fscl", argv[i]) == 0)
			svrg_are_sgld_parameters->Fscl = atof(argv[++i]) ;
		else if (strcmp("-svrg_are_sgld_parameters->CV_update_rate", argv[i]) == 0)
			svrg_are_sgld_parameters->CV_update_rate = atoi(argv[++i]) ;
		else if (strcmp("-svrg_are_sgld_parameters->sig2hat_update_rate", argv[i]) == 0)
			svrg_are_sgld_parameters->sig2hat_update_rate = atoi(argv[++i]) ;
}

void print_svrg_are_sgld_parameters(struct_svrg_are_sgld_parameters *svrg_are_sgld_parameters) {
	printf("svrg_are_sgld_parameters->eta_1:  \t %f \n", svrg_are_sgld_parameters->eta_1) ;
	printf("svrg_are_sgld_parameters->eta_2:  \t %f \n", svrg_are_sgld_parameters->eta_2) ;
	printf("svrg_are_sgld_parameters->tau_1:  \t %f \n", svrg_are_sgld_parameters->tau_1) ;
	printf("svrg_are_sgld_parameters->tau_2:  \t %f \n", svrg_are_sgld_parameters->tau_2) ;
	printf("svrg_are_sgld_parameters->subsample_type:  \t %d \n", svrg_are_sgld_parameters->subsample_type) ;
	printf("svrg_are_sgld_parameters->sig2hat_rep:  \t %d \n", svrg_are_sgld_parameters->sig2hat_rep) ;
	printf("svrg_are_sgld_parameters->gain_sig2hat_t0:  \t %d \n", svrg_are_sgld_parameters->gain_sig2hat_t0) ;
	printf("svrg_are_sgld_parameters->gain_sig2hat_c0:  \t %f \n", svrg_are_sgld_parameters->gain_sig2hat_c0) ;
	printf("svrg_are_sgld_parameters->Fscl:  \t %f \n", svrg_are_sgld_parameters->Fscl) ;
	printf("svrg_are_sgld_parameters->CV_update_rate:  \t %d \n", svrg_are_sgld_parameters->CV_update_rate) ;
	printf("svrg_are_sgld_parameters->sig2hat_update_rate:  \t %d \n", svrg_are_sgld_parameters->sig2hat_update_rate) ;
}

void destroy_svrg_are_sgld_parameters(struct_svrg_are_sgld_parameters *svrg_are_sgld_parameters) {
	free((char*) svrg_are_sgld_parameters) ;
}

/*
 * SUB-DATA
 */

void alloc_svrg_are_sgld_subdata(struct_svrg_are_sgld_subdata **svrg_are_sgld_subdata, struct_data *data, int en_ysub) {

	int i ;

	*svrg_are_sgld_subdata = (struct_svrg_are_sgld_subdata*)malloc((size_t)sizeof(struct_svrg_are_sgld_subdata)) ;

	(*svrg_are_sgld_subdata)->en_ysub = en_ysub ;

	(*svrg_are_sgld_subdata)->I_ysub = ivector(1, en_ysub) ;

	for (i=1; i<=(*svrg_are_sgld_subdata)->en_ysub; i++)
		(*svrg_are_sgld_subdata)->I_ysub[i] = i ;

	(*svrg_are_sgld_subdata)->data = data ;
}

void sample_svrg_are_sgld_subdata(struct_svrg_are_sgld_subdata *svrg_are_sgld_subdata,
					struct_data *data,
					struct_svrg_are_sgld_parameters *svrg_are_sgld_parameters) {

	int i ;

	int en_ysub = svrg_are_sgld_subdata->en_ysub ;

	int en_y = data->en_y ;

	int *I_ysub = svrg_are_sgld_subdata->I_ysub ;

	int subsample_type = svrg_are_sgld_parameters->subsample_type ;

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
	svrg_are_sgld_subdata->data = data ;
}

void print_svrg_are_sgld_subdata(struct_svrg_are_sgld_subdata *svrg_are_sgld_subdata) {

	int i ;

	int en_y = svrg_are_sgld_subdata->data->en_y ;

	int en_ysub = svrg_are_sgld_subdata->en_ysub ;

	int *I_ysub = svrg_are_sgld_subdata->I_ysub ;

	printf("svrg_are_sgld_subdata->data->en_y  \t %d , \n", en_y) ;

	printf("svrg_are_sgld_subdata->en_ysub  \t %d , \n", en_ysub) ;

	printf("svrg_are_sgld_subdata->I_ysub[%i : %i] \n", 1,MIN(en_ysub,50)) ;
	for (i=1; i<=MIN(en_ysub,50); i++) printf("%i, " , I_ysub[i]) ;
	printf(" \n") ;
}

void destroy_svrg_are_sgld_subdata(struct_svrg_are_sgld_subdata *svrg_are_sgld_subdata) {

	free_ivector((svrg_are_sgld_subdata->I_ysub), 1, svrg_are_sgld_subdata->en_ysub) ;

	free((char*) svrg_are_sgld_subdata) ;
}


/*
 * GRADIENT ESTIMATOR
 */

void comp_svrg_are_sgld_grad_log_lik_estimate(double * svrg_are_sgld_grad_log_lik_est,
                                            struct_svrg_are_sgld_subdata *svrg_are_sgld_subdata,
                                            struct_random_parameters* randpar,
                                            struct_fixed_parameters* fixpar) {

	int i ;

	int en_yall = svrg_are_sgld_subdata->data->en_y ;

	int en_ysub = svrg_are_sgld_subdata->en_ysub ;

	int *I_ysub = svrg_are_sgld_subdata->I_ysub ;

	double *yall = svrg_are_sgld_subdata->data->y ;

	double ysub_i ;

	(*svrg_are_sgld_grad_log_lik_est) = 0.0 ;

	for (i=1 ; i<=en_ysub ; i++) {

		ysub_i = yall[I_ysub[i]] ;

		(*svrg_are_sgld_grad_log_lik_est) = (*svrg_are_sgld_grad_log_lik_est)
                                                            +comp_gradient_unit_log_lik( ysub_i,
                                                                                        randpar,
                                                                                        fixpar) ;
	}

	(*svrg_are_sgld_grad_log_lik_est) *= ((double)en_yall)/((double)en_ysub) ;
}

void comp_svrg_are_sgld_udpate_random_parameters(struct_svrg_are_sgld_subdata *svrg_are_sgld_subdata,
                                                struct_random_parameters* randpar_1,
                                                struct_random_parameters* randpar_2,
                                                struct_fixed_parameters* fixpar,
                                                struct_svrg_are_sgld_parameters *svrg_are_sgld_parameters,
                                                double * svrg_are_sgld_grad_log_lik_est_1,
                                                double * svrg_are_sgld_grad_log_lik_est_2,
                                                double * svrg_are_sgld_grad_log_prior_1,
                                                double * svrg_are_sgld_grad_log_prior_2) {

	double eta_1 = svrg_are_sgld_parameters->eta_1 ;

	double eta_2 = svrg_are_sgld_parameters->eta_2 ;

	double tau_1 = svrg_are_sgld_parameters->tau_1 ;

	double tau_2 = svrg_are_sgld_parameters->tau_2 ;

	double xi ;
	
	double val ;
	
	/*Component 1*/
	
	comp_gradient_log_prior(svrg_are_sgld_grad_log_prior_1, randpar_1, fixpar) ;
	
	randpar_1->theta = randpar_1->theta 
                                    + eta_1 * (*svrg_are_sgld_grad_log_prior_1)  ;

	comp_svrg_are_sgld_grad_log_lik_estimate( svrg_are_sgld_grad_log_lik_est_1,
                                                    svrg_are_sgld_subdata,
                                                    randpar_1,
                                                    fixpar) ;
	
	randpar_1->theta = randpar_1->theta 
                                            + eta_1 * (*svrg_are_sgld_grad_log_lik_est_1)  ;

	xi = normalrng() ;
			
	randpar_1->theta = randpar_1->theta
						+ sqrt( 2.0 * eta_1 * tau_1 ) * xi ;

	/*Component 2*/
	
	comp_gradient_log_prior(svrg_are_sgld_grad_log_prior_2, randpar_2, fixpar) ;

	randpar_2->theta = randpar_2->theta 
						+ eta_2 * (*svrg_are_sgld_grad_log_prior_2) ;

	comp_svrg_are_sgld_grad_log_lik_estimate( svrg_are_sgld_grad_log_lik_est_2,
                                                    svrg_are_sgld_subdata,
                                                    randpar_2,
                                                    fixpar) ;

	randpar_2->theta = randpar_2->theta
                                    + eta_2 * (*svrg_are_sgld_grad_log_lik_est_2) ;
	
	randpar_2->theta = randpar_2->theta 
                                    + sqrt( 2.0 * eta_2 * tau_2 ) * xi ;
}



/*
 * Compute log likelihood estimate
 */

void alloc_svrg_are_sgld_controle_variate(struct_svrg_are_sgld_controle_variate **svrg_are_sgld_controle_variate) {

	*svrg_are_sgld_controle_variate = (struct_svrg_are_sgld_controle_variate*)malloc((size_t)sizeof(struct_svrg_are_sgld_controle_variate)) ;

	alloc_random_parameters( &( (*svrg_are_sgld_controle_variate)->randpar )  ) ;

	(*svrg_are_sgld_controle_variate)->log_lik = 0.0 ;
}

void destroy_svrg_are_sgld_controle_variate(struct_svrg_are_sgld_controle_variate *svrg_are_sgld_controle_variate) {

	destroy_random_parameters( svrg_are_sgld_controle_variate->randpar ) ;

	free((char*) svrg_are_sgld_controle_variate) ;
}

void  swap_svrg_are_sgld_control_variate(struct_svrg_are_sgld_controle_variate *svrg_are_sgld_controle_variate_1,
                                        struct_svrg_are_sgld_controle_variate *svrg_are_sgld_controle_variate_2,
                                        struct_random_parameters *randpar_aux) {

    int i ;
    
    double log_lik_aux ;

    log_lik_aux = svrg_are_sgld_controle_variate_2->log_lik ;
    svrg_are_sgld_controle_variate_2->log_lik = svrg_are_sgld_controle_variate_1->log_lik ;
    svrg_are_sgld_controle_variate_1->log_lik = log_lik_aux ;
    
    swap_random_parameters(svrg_are_sgld_controle_variate_1->randpar,
                           svrg_are_sgld_controle_variate_2->randpar,
                            randpar_aux) ;
}

void  comp_svrg_are_sgld_controle_variate(struct_svrg_are_sgld_subdata *svrg_are_sgld_subdata,
                                            struct_random_parameters* randpar_1,
                                            struct_random_parameters* randpar_2,
                                            struct_fixed_parameters* fixpar,
                                            struct_svrg_are_sgld_controle_variate *svrg_are_sgld_controle_variate_1,
                                            struct_svrg_are_sgld_controle_variate *svrg_are_sgld_controle_variate_2) {

	double *log_lik_1 = &(svrg_are_sgld_controle_variate_1->log_lik) ;

	double *log_lik_2 = &(svrg_are_sgld_controle_variate_2->log_lik) ;

	double *yall = svrg_are_sgld_subdata->data->y ;

	int en_yall = svrg_are_sgld_subdata->data->en_y ;

	int i ;

	double yall_i ;

	copy_random_parameters(randpar_1, svrg_are_sgld_controle_variate_1->randpar) ;

	copy_random_parameters(randpar_2, svrg_are_sgld_controle_variate_2->randpar) ;

	(*log_lik_1) = 0.0 ;

	(*log_lik_2) = 0.0 ;

	for ( i=1 ; i<=en_yall ; i++ ) {

		yall_i = yall[i] ;

		(*log_lik_1) += comp_unit_log_lik( yall_i , randpar_1 , fixpar ) ;

		(*log_lik_2) += comp_unit_log_lik( yall_i , randpar_2 , fixpar ) ;
	}
}


double  comp_svrg_are_sgld_log_lik_estimate(struct_svrg_are_sgld_subdata *svrg_are_sgld_subdata,
                                            struct_random_parameters* randpar,
                                            struct_fixed_parameters* fixpar,
                                            struct_svrg_are_sgld_controle_variate *svrg_are_sgld_controle_variate) {

	int i ;

	int en_yall = svrg_are_sgld_subdata->data->en_y ;

	int en_ysub = svrg_are_sgld_subdata->en_ysub ;

	int *I_ysub = svrg_are_sgld_subdata->I_ysub ;

	double *yall = svrg_are_sgld_subdata->data->y ;

	struct_random_parameters *svrg_randpar = svrg_are_sgld_controle_variate->randpar ;

	double svrg_log_lik = svrg_are_sgld_controle_variate->log_lik ;

	double ysub_i ;

	double loglik_est ;

	double en_rate ;

	en_rate = ((double)en_yall)/((double)en_ysub) ;

	loglik_est = 0.0 ;

	for (i=1 ; i<=en_ysub ; i++) {

            ysub_i = yall[I_ysub[i]] ;

            loglik_est += en_rate * comp_unit_log_lik(ysub_i, randpar, fixpar) ;

            loglik_est -= en_rate * comp_unit_log_lik(ysub_i, svrg_randpar, fixpar) ;
	}

	loglik_est += svrg_log_lik ;

	return (loglik_est) ;
}



/*
 * Compute the stochastisity correction term
 */

double comp_svrg_are_sgld_gain_sig2hat(int iter,
					struct_svrg_are_sgld_parameters *svrg_are_sgld_parameters) {

	double gt ;

	double t_0 = svrg_are_sgld_parameters->gain_sig2hat_t0 ;

	double c_0 = svrg_are_sgld_parameters->gain_sig2hat_c0 ;
#if 0
	gt = (double)t_0 / fmax ( (double)t_0, (double)iter) ;

	gt = fmax( gt, (double) c_0  ) ;
#else	
	gt = svrg_are_sgld_parameters->gain_sig2hat_c0 ;
#endif	
	return ( gt ) ;
}

double comp_svrg_are_sgld_sig2tilde(struct_svrg_are_sgld_subdata *svrg_are_sgld_subdata,
                                    struct_random_parameters* randpar_1,
                                    struct_random_parameters* randpar_2,
                                    struct_fixed_parameters* fixpar,
                                    struct_svrg_are_sgld_controle_variate *svrg_are_sgld_controle_variate_1,
                                    struct_svrg_are_sgld_controle_variate *svrg_are_sgld_controle_variate_2,
                                    struct_svrg_are_sgld_parameters *svrg_are_sgld_parameters,
                                    struct_svrg_are_sgld_subdata *svrg_are_sgld_subdata_aux) {

	int i ;

	double nrep = svrg_are_sgld_parameters->sig2hat_rep ;

	double loglik_est_1 ;
	double loglik_est_2 ;

	double sxx ;
	double sx ;

	double sig2tilde ;

	sxx = 0.0 ;
        
	sx = 0.0 ;

	svrg_are_sgld_subdata_aux->data = svrg_are_sgld_subdata->data ;

	svrg_are_sgld_subdata_aux->en_ysub = svrg_are_sgld_subdata->en_ysub ;

	for ( i=1; i<=nrep; i++) {

		sample_svrg_are_sgld_subdata( svrg_are_sgld_subdata_aux,
                                                svrg_are_sgld_subdata->data,
                                                svrg_are_sgld_parameters) ;

		loglik_est_1 = comp_svrg_are_sgld_log_lik_estimate(svrg_are_sgld_subdata_aux,
                                                                    randpar_1,
                                                                    fixpar,
                                                                    svrg_are_sgld_controle_variate_1) ;

		loglik_est_2 = comp_svrg_are_sgld_log_lik_estimate(svrg_are_sgld_subdata_aux,
                                                                    randpar_2,
                                                                    fixpar,
                                                                    svrg_are_sgld_controle_variate_2) ;

		sx += (loglik_est_1-loglik_est_2) ;
                
		sxx += (loglik_est_1-loglik_est_2)*(loglik_est_1-loglik_est_2) ;
	}

	sig2tilde = sxx/nrep - (sx/nrep)*(sx/nrep) ;
	
        /*Round-off error*/
	sig2tilde = MAX(sig2tilde,0.0) ;

	return	(sig2tilde) ;
}

void comp_svrg_are_sgld_sig2hat(double *sig2hat,
				double sig2tilde,
				double gt) {

	(*sig2hat) = gt*sig2tilde + (1.0-gt)* (*sig2hat) ;
}


/*
 * Compute the Metropolis Hastings
 */

void comp_svrg_are_sgld_acceptance_ratio(double *logMHAccRat, double *logMHAccRatCorr,
                                        struct_svrg_are_sgld_subdata *svrg_are_sgld_subdata,
                                        struct_random_parameters* randpar_1,
                                        struct_random_parameters* randpar_2,
                                        struct_fixed_parameters* fixpar,
                                        struct_svrg_are_sgld_controle_variate *svrg_are_sgld_controle_variate_1,
                                        struct_svrg_are_sgld_controle_variate *svrg_are_sgld_controle_variate_2,
                                        struct_svrg_are_sgld_parameters *svrg_are_sgld_parameters,
                                        double sig2hat) {

	double tau_1 = svrg_are_sgld_parameters->tau_1 ;

	double tau_2 = svrg_are_sgld_parameters->tau_2 ;

	double Fscl = svrg_are_sgld_parameters->Fscl ;

	double Energy_1 ;

	double Energy_2 ;

	int i ;

	int en_ysub = svrg_are_sgld_subdata->en_ysub ;

	int *I_ysub = svrg_are_sgld_subdata->I_ysub ;

	double *yall = svrg_are_sgld_subdata->data->y ;

	int en_yall = svrg_are_sgld_subdata->data->en_y ;

	double ysub_i ;

	double en_rate ;

	struct_random_parameters *svrg_randpar_1 = svrg_are_sgld_controle_variate_1->randpar ;

	double svrg_log_lik_1 = svrg_are_sgld_controle_variate_1->log_lik ;

	struct_random_parameters *svrg_randpar_2 = svrg_are_sgld_controle_variate_2->randpar ;

	double svrg_log_lik_2 = svrg_are_sgld_controle_variate_2->log_lik ;

	en_rate = ((double)en_yall)/((double)en_ysub) ;

	Energy_1 = - comp_log_prior(randpar_1, fixpar) ;

	Energy_2 = - comp_log_prior(randpar_2, fixpar) ;

	for (i=1 ; i<=en_ysub ; i++ ) {

            ysub_i = yall[I_ysub[i]] ;

            Energy_1 += en_rate * comp_unit_log_lik(ysub_i, randpar_1, fixpar) ;
            Energy_1 -= en_rate * comp_unit_log_lik(ysub_i, svrg_randpar_1, fixpar) ;

            Energy_2 += en_rate * comp_unit_log_lik(ysub_i, randpar_2, fixpar) ;
            Energy_2 -= en_rate * comp_unit_log_lik(ysub_i, svrg_randpar_2, fixpar) ;
	}

	Energy_1 += svrg_log_lik_1 ;

	Energy_2 += svrg_log_lik_2 ;

	(*logMHAccRat) = (1/tau_2-1/tau_1)*(Energy_2-Energy_1) ;

	(*logMHAccRatCorr) = (*logMHAccRat)
				-(1/tau_2-1/tau_1)*(1/tau_2-1/tau_1)*sig2hat/Fscl ;
}




















