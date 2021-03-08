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


#ifndef __file_svrg_are_sgld_data__
	#define __file_svrg_are_sgld_data__ 1
#endif
#ifndef __file_svrg_are_sgld_theta_1__
	#define __file_svrg_are_sgld_theta_1__ 1
#endif
#ifndef __file_svrg_are_sgld_theta_2__
	#define __file_svrg_are_sgld_theta_2__ 1
#endif
#ifndef __file_svrg_are_sgld_grad_log_lik_est_1__
	#define __file_svrg_are_sgld_grad_log_lik_est_1__ 0
#endif
#ifndef __file_svrg_are_sgld_grad_log_lik_est_2__
	#define __file_svrg_are_sgld_grad_log_lik_est_2__ 0
#endif
#ifndef __file_svrg_are_sgld_logMHAccRatCorr__
	#define __file_svrg_are_sgld_logMHAccRatCorr__ 1
#endif
#ifndef __file_svrg_are_sgld_sig2tilde__
	#define __file_svrg_are_sgld_sig2tilde__ 1
#endif
#ifndef __file_svrg_are_sgld_sig2hat__
	#define __file_svrg_are_sgld_sig2hat__ 1
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>

#include "nrutil.h"
#include "logPDF.h"
#include "RNG.h"
#include"Bayesian_model.h"
#include "svrg_are_sgld.h"

#define MAX(x,y) (((x)>(y))?(x):(y))
#define MIN(x,y) (((x)<(y))?(x):(y))

int comp_epoch(int *epoch, int iter, int n_sub, int n_all) {

	int Q ;
	
	(*epoch) = (int) ( (iter * n_sub) / ((double) n_all) ) ;

	Q = ( (iter * n_sub) % n_all ==0 ) ;
	
	return( Q ) ;
}


int main(int argc, char *argv[]){

	int rng_seed ;
	int i ;
	double un ;
	struct timeval t1;
	
	int Qepoch ;
	int epoch ;

	struct_fixed_parameters *fixpar ;

    struct_random_parameters *randpar_1 ;
	struct_random_parameters *randpar_2 ;
	struct_random_parameters *randpar_aux ;

	struct_svrg_are_sgld_parameters *svrg_are_sgld_parameters ;

	int en_y ;
	struct_data *data ;

	int en_ysub ;
	struct_svrg_are_sgld_subdata *svrg_are_sgld_subdata ;

	struct_svrg_are_sgld_controle_variate *svrg_are_sgld_controle_variate_1 ;
	struct_svrg_are_sgld_controle_variate *svrg_are_sgld_controle_variate_2 ;

	double svrg_are_sgld_grad_log_lik_est_1 ;
	double svrg_are_sgld_grad_log_lik_est_2 ;
	double svrg_are_sgld_grad_log_prior_1 ;
	double svrg_are_sgld_grad_log_prior_2 ;

	int iter_mcmc ;
	int N_mcmc ;

	double gt_sig2hat ;
	double sig2tilde ;
	double sig2hat ;

	double logMHAccRat ;
	double logMHAccRatCorr ;
	double MHAccProb ;

	char output_dir[200] ;
	char file_name[200] ;
#if __file_svrg_are_sgld_data__
	FILE *ins_svrg_are_sgld_data = NULL ;
#endif
#if __file_svrg_are_sgld_theta_1__
	FILE *ins_svrg_are_sgld_theta_1 = NULL ;
#endif
#if __file_svrg_are_sgld_theta_2__
	FILE *ins_svrg_are_sgld_theta_2 = NULL ;
#endif
#if __file_svrg_are_sgld_grad_log_lik_est_1__
	FILE *ins_svrg_are_sgld_grad_log_lik_est_1 = NULL ;
#endif
#if __file_svrg_are_sgld_grad_log_lik_est_2__
	FILE *ins_svrg_are_sgld_grad_log_lik_est_2 = NULL ;
#endif
#if __file_svrg_are_sgld_logMHAccRatCorr__
	FILE *ins_svrg_are_sgld_logMHAccRatCorr = NULL ;
#endif
#if __file_svrg_are_sgld_sig2tilde__
	FILE *ins_svrg_are_sgld_sig2tilde = NULL ;
#endif
#if __file_svrg_are_sgld_sig2hat__
	FILE *ins_svrg_are_sgld_sig2hat = NULL ;
#endif

	// auxiliary

	struct_svrg_are_sgld_subdata *svrg_are_sgld_subdata_aux ;

	/*
	 * SET DEFAULT EXAMPLE SETTINGS ---------------------------------------
	 * */
	printf("\n\n  ***** SET ALGORITHMIC SETTINGS  ***** \n\n") ;

	/*
	 * INITIALIZE THE RNG -----------------------------------------------------
	 * */
	printf("\n\n  ***** INITIALIZE THE RNG  ***** \n\n") ;

	/* .. default */
	//rng_seed =  time(NULL) ;
	gettimeofday(&t1, NULL);
	rng_seed = abs((t1.tv_sec * 1000) + (t1.tv_usec / 1000)) ;

	/* .. external */
	for (i = 1; i < argc; i++)
		if (strcmp("-rng_seed", argv[i]) == 0)
			rng_seed = atoi(argv[++i]) ;

	/* .. print */
	printf("rng_seed: \t %d \n", rng_seed);

	setseedrng( (unsigned long) rng_seed ) ;
	for ( i=1 ; i<=10 ; i++ ) un = uniformrng() ;

	/*
	 * OPEN FILES -------------------------------------------------------------
	 * */
	printf("\n\n ***** OPEN FILES ***** \n\n") ;

	snprintf(output_dir, sizeof output_dir, "%s", "./output_files_vr_are_sgld");
	for (i = 1; i < argc; i++)
		if (strcmp("-output_path", argv[i]) == 0)
			snprintf(output_dir, sizeof output_dir, "%s", argv[++i]);

#if __file_svrg_are_sgld_data__
	snprintf(file_name, sizeof file_name, "%s/svrg_are_sgld_data.dat",
			output_dir);
	ins_svrg_are_sgld_data = fopen( file_name , "w" ) ;
	printf("==> %s \n", file_name) ;
#endif
#if __file_svrg_are_sgld_theta_1__
	snprintf(file_name, sizeof file_name, "%s/svrg_are_sgld_theta_1.out",
			output_dir);
	ins_svrg_are_sgld_theta_1 = fopen( file_name , "w" ) ;
	printf("==> %s \n", file_name) ;
#endif
#if __file_svrg_are_sgld_theta_2__
	snprintf(file_name, sizeof file_name, "%s/svrg_are_sgld_theta_2.out",
			output_dir);
	ins_svrg_are_sgld_theta_2 = fopen( file_name , "w" ) ;
	printf("==> %s \n", file_name) ;
#endif
#if __file_svrg_are_sgld_grad_log_lik_est_1__
	snprintf(file_name, sizeof file_name, "%s/svrg_are_sgld_grad_log_lik_est_1.out",
			output_dir);
	ins_svrg_are_sgld_grad_log_lik_est_1 = fopen( file_name , "w" ) ;
	printf("==> %s \n", file_name) ;
#endif
#if __file_svrg_are_sgld_grad_log_lik_est_2__
	snprintf(file_name, sizeof file_name, "%s/svrg_are_sgld_grad_log_lik_est_2.out",
			output_dir);
	ins_svrg_are_sgld_grad_log_lik_est_2 = fopen( file_name , "w" ) ;
	printf("==> %s \n", file_name) ;
#endif
#if __file_svrg_are_sgld_logMHAccRatCorr__
	snprintf(file_name, sizeof file_name, "%s/svrg_are_sgld_logMHAccRatCorr.out",
			output_dir);
	ins_svrg_are_sgld_logMHAccRatCorr = fopen( file_name , "w" ) ;
	printf("==> %s \n", file_name) ;
#endif
#if __file_svrg_are_sgld_sig2tilde__
	snprintf(file_name, sizeof file_name, "%s/svrg_are_sgld_sig2tilde.out",
			output_dir);
	ins_svrg_are_sgld_sig2tilde = fopen( file_name , "w" ) ;
	printf("==> %s \n", file_name) ;
#endif
#if __file_svrg_are_sgld_sig2hat__
	snprintf(file_name, sizeof file_name, "%s/svrg_are_sgld_sig2hat.out",
			output_dir);
	ins_svrg_are_sgld_sig2hat = fopen( file_name , "w" ) ;
	printf("==> %s \n", file_name) ;
#endif
	/*
	 * SET FIXED BAYESIAN MODEL PARAMETERS --------------------------------
	 * */
	printf("\n\n ***** SET FIXED BAYESIAN MODEL PARAMETERS ***** \n\n") ;

	/* .. allocate memory */
	alloc_fixed_parameters(&fixpar) ;

	/* .. default */
	initialise_fixed_parameters(fixpar) ;

	set_external_fixed_parameters(fixpar, argc, argv) ;

	print_fixed_parameters( fixpar ) ;

	/*
	 * GENERATE THE WHOLE DATA SET --------------------------------
	 * */
	printf("\n\n ***** GENERATE THE WHOLE DATA SET ***** \n\n") ;

	/*fix RNG*/
	rng_seed = 1983000 ;
	setseedrng( (unsigned long) rng_seed ) ;
	for ( i=1 ; i<=10 ; i++ ) un = uniformrng() ;
	
	/* .. default */
	en_y = 1000000 ;

	/* .. external */
	for (i = 1; i < argc; i++)
		if (strcmp("-data->en_y", argv[i]) == 0)
			en_y = atoi(argv[++i]) ;

	/* .. allocate */
	/*alloc_data(data, en_y) ;*/

	/* .. generate */
	alloc_and_generate_data(&data, fixpar, en_y) ;
	
	/*resume RNG*/
	gettimeofday(&t1, NULL);
	rng_seed = abs((t1.tv_sec * 1000) + (t1.tv_usec / 1000)) ;
	setseedrng( (unsigned long) rng_seed ) ;
	for ( i=1 ; i<=10 ; i++ ) un = uniformrng() ;

	printf("-data->en_y: \t %d \n", en_y) ;
#if __file_svrg_are_sgld_data__
	if (ins_svrg_are_sgld_data != NULL)
		for (i=1; i<=data->en_y; i++)
			fprintf(ins_svrg_are_sgld_data,"%f \n", data->y[i]) ;
#endif
	/*
	 * SET RANDOM BAYESIAN MODEL PARAMETERS --------------------------------
	 * */
	printf("\n\n ***** SET RANDOM BAYESIAN MODEL PARAMETERS ***** \n\n") ;

	/* .. allocate */
	alloc_random_parameters( &randpar_1) ;
	alloc_random_parameters( &randpar_2) ;

	/* .. initialise */
	seed_random_parameters(randpar_1) ;
	seed_random_parameters(randpar_2) ;

	for (i = 1; i < argc; i++)
		if (strcmp("-randpar_1->dim_theta", argv[i]) == 0)
			randpar_1->dim_theta = atoi(argv[++i]) ;
		else if (strcmp("-randpar_1->theta", argv[i]) == 0)
			randpar_1->theta = atof(argv[++i]) ;
		else if (strcmp("-randpar_2->dim_theta", argv[i]) == 0)
			randpar_2->dim_theta = atoi(argv[++i]) ;
		else if (strcmp("-randpar_2->theta", argv[i]) == 0)
			randpar_2->theta = atof(argv[++i]) ;

	/* .. print */
	printf("low temperature \n") ;
	print_random_parameters(randpar_1) ;
	printf("high temperature \n") ;
	print_random_parameters(randpar_2) ;

	/*
	 * SET SGLD MCMC PARAMETERS --------------------------------
	 * */
	printf("\n\n ***** SET SGLD MCMC PARAMETERS ***** \n\n") ;

	/* .. allocate */
	alloc_svrg_are_sgld_parameters( &svrg_are_sgld_parameters ) ;

	/* .. initialise */
	set_svrg_are_sgld_parameters( svrg_are_sgld_parameters ) ;

	N_mcmc = 1000000 ;

	for (i = 1; i < argc; i++)
		if (strcmp("-N_mcmc", argv[i]) == 0)
			N_mcmc = atoi(argv[++i]) ;

	set_external_svrg_are_sgld_parameters( svrg_are_sgld_parameters, argc, argv) ;

	/* .. print */
	printf("N_mcmc:  \t %d \n", N_mcmc) ;
	print_svrg_are_sgld_parameters( svrg_are_sgld_parameters ) ;

	/*
	 * SET SGLD SUBDATA PARAMETERS --------------------------------
	 */
	printf("\n\n ***** SET SGLD SUBDATA PARAMETER ***** \n\n") ;

	en_ysub = 1000 ;

	for (i = 1; i < argc; i++)
		if (strcmp("-svrg_are_sgld_subdata->en_ysub", argv[i]) == 0)
			en_ysub = atoi(argv[++i]) ;

	/* .. alloc */
	/* .. it points to the data !!!!	 */
	alloc_svrg_are_sgld_subdata( &svrg_are_sgld_subdata, data, en_ysub ) ;

	/* .. sample */
	sample_svrg_are_sgld_subdata( svrg_are_sgld_subdata, data, svrg_are_sgld_parameters) ;

	/* .. print */
	print_svrg_are_sgld_subdata(svrg_are_sgld_subdata) ;

	/*
	 * SET SVRG PARAMETERS --------------------------------
	 * */
	printf("\n\n ***** SET SVRG PARAMETERS ***** \n\n") ;

	alloc_svrg_are_sgld_controle_variate( &svrg_are_sgld_controle_variate_1 ) ;
	alloc_svrg_are_sgld_controle_variate( &svrg_are_sgld_controle_variate_2 ) ;

	/*
	 * AUXILIARY AND WORKING ALLOCATIONS --------------------------------
	 */
	printf("\n\n ***** SET SGLD SUBDATA PARAMETER ***** \n\n") ;

	/* .. alloc */
	/* .. it points to the data !!!!	 */
	alloc_svrg_are_sgld_subdata( &svrg_are_sgld_subdata_aux, data, en_ysub ) ;
	/* .. sample */
	sample_svrg_are_sgld_subdata( svrg_are_sgld_subdata_aux, data, svrg_are_sgld_parameters) ;

	/* .. allocate */
	alloc_random_parameters( &randpar_aux ) ;
	seed_random_parameters( randpar_aux ) ;

	/*
	 * PERFORM THE SGLD ITERATIONS --------------------------------
	 * */
	printf("\n\n ***** SET SGLD MCMC PARAMETERS ***** \n\n") ;

	/*
	* Initialise / seed
	* */
	
	for (iter_mcmc = 0 ; iter_mcmc <= 0 ; iter_mcmc++) {

		/*
		* Initialise sub sample
		* */

		sample_svrg_are_sgld_subdata( svrg_are_sgld_subdata, data, svrg_are_sgld_parameters) ;

		/*
		 * Initialise Randpar's
		 * */

		//copy_random_parameters( randpar_1, randpar_1 ) ; 

		//copy_random_parameters( randpar_2, randpar_2 ) ;

		/*
		 * Seed SVRG control variates
		 * */

		comp_svrg_are_sgld_controle_variate( svrg_are_sgld_subdata,
                                                        randpar_1,
                                                        randpar_2,
                                                        fixpar,
                                                        svrg_are_sgld_controle_variate_1,
                                                        svrg_are_sgld_controle_variate_2) ;

		/*
		 * Seed SVRG sig2hat
		 * */

		sig2hat = comp_svrg_are_sgld_sig2tilde(svrg_are_sgld_subdata,
                                                        randpar_1,
                                                        randpar_2,
                                                        fixpar,
                                                        svrg_are_sgld_controle_variate_1,
                                                        svrg_are_sgld_controle_variate_2,
                                                        svrg_are_sgld_parameters,
                                                        svrg_are_sgld_subdata_aux) ;
	}

	
	
	/*
	* Iterate
	* */       
	for (iter_mcmc = 1 ; iter_mcmc <= N_mcmc ; iter_mcmc++) {

            /*
             * COUNTER
             * */

            if ( (iter_mcmc % (N_mcmc/100)) == 0 ) {
                    printf("%d%%, ", (N_mcmc-iter_mcmc)/(N_mcmc/100)) ;
                    fflush(stdout) ;
            }

            /*
             * Sub sample
             * */

            sample_svrg_are_sgld_subdata( svrg_are_sgld_subdata, data, svrg_are_sgld_parameters) ;

            /*
             * Update randpar's
             * */

            comp_svrg_are_sgld_udpate_random_parameters(svrg_are_sgld_subdata,
                                                        randpar_1,
                                                        randpar_2,
                                                        fixpar,
                                                        svrg_are_sgld_parameters,
                                                        &svrg_are_sgld_grad_log_lik_est_1,
                                                        &svrg_are_sgld_grad_log_lik_est_2,
                                                        &svrg_are_sgld_grad_log_prior_1,
                                                        &svrg_are_sgld_grad_log_prior_2) ;

            /*
             * Update SVRG control variates
             * */

            if ( (iter_mcmc % svrg_are_sgld_parameters->CV_update_rate)==0 ) {
                    comp_svrg_are_sgld_controle_variate( svrg_are_sgld_subdata,
                                                        randpar_1,
                                                        randpar_2,
                                                        fixpar,
                                                        svrg_are_sgld_controle_variate_1,
                                                        svrg_are_sgld_controle_variate_2) ;
            }

            /*
             * Update sig2hat
             * */

            if ( (iter_mcmc % svrg_are_sgld_parameters->sig2hat_update_rate)==0 ) {

                    gt_sig2hat = comp_svrg_are_sgld_gain_sig2hat(iter_mcmc,  svrg_are_sgld_parameters) ;
                    //gt_sig2hat = 0.2 ;

                    if (iter_mcmc == 1 ) {
                            sig2hat = 0.0 ;
                            gt_sig2hat = 1.0 ;
                    }

                    sig2tilde = comp_svrg_are_sgld_sig2tilde(svrg_are_sgld_subdata,
                                                            randpar_1,
                                                            randpar_2,
                                                            fixpar,
                                                            svrg_are_sgld_controle_variate_1,
                                                            svrg_are_sgld_controle_variate_2,
                                                            svrg_are_sgld_parameters,
                                                            svrg_are_sgld_subdata_aux) ;

                    sig2hat = gt_sig2hat * sig2tilde + (1.0 - gt_sig2hat) * sig2hat ;
            }

            /*
             * Swapping step
             * */

            comp_svrg_are_sgld_acceptance_ratio( &logMHAccRat,
                                                &logMHAccRatCorr,
                                                svrg_are_sgld_subdata,
                                                randpar_1,
                                                randpar_2,
                                                fixpar,
                                                svrg_are_sgld_controle_variate_1,
                                                svrg_are_sgld_controle_variate_2,
                                                svrg_are_sgld_parameters,
                                                sig2hat) ;

            MHAccProb = exp( fmin( logMHAccRatCorr , 0.0 ) ) ;

            un = uniformrng() ;

            if ( MHAccProb > un ) {
                    swap_random_parameters( randpar_1, randpar_2, randpar_aux) ;
                    swap_svrg_are_sgld_control_variate( svrg_are_sgld_controle_variate_1,
                                                        svrg_are_sgld_controle_variate_2,
                                                        randpar_aux) ;			
            }

            /*
             * Record
             * */


#if __file_svrg_are_sgld_theta_1__
            if (ins_svrg_are_sgld_theta_1 != NULL)
                    fprintf(ins_svrg_are_sgld_theta_1,"%f \n", randpar_1->theta) ;
#endif
#if __file_svrg_are_sgld_theta_2__
            if (ins_svrg_are_sgld_theta_2 != NULL)
                    fprintf(ins_svrg_are_sgld_theta_2,"%f \n", randpar_2->theta) ;
#endif

            Qepoch = comp_epoch(&epoch, iter_mcmc, 
                            svrg_are_sgld_subdata->en_ysub, 
                            data->en_y) ;

printf("%f %f \n", randpar_1->theta, randpar_2->theta) ;

            if ( Qepoch ) {

#if __file_svrg_are_sgld_grad_log_lik_est_1__
            if (ins_svrg_are_sgld_grad_log_lik_est_1 != NULL)
                    fprintf(ins_svrg_are_sgld_grad_log_lik_est_1,"%f \n", svrg_are_sgld_grad_log_lik_est_1) ;
#endif
#if __file_svrg_are_sgld_grad_log_lik_est_2__
            if (ins_svrg_are_sgld_grad_log_lik_est_2 != NULL)
                    fprintf(ins_svrg_are_sgld_grad_log_lik_est_2,"%f \n", svrg_are_sgld_grad_log_lik_est_2) ;
#endif
#if __file_svrg_are_sgld_logMHAccRatCorr__
            if (ins_svrg_are_sgld_logMHAccRatCorr != NULL)
                    fprintf(ins_svrg_are_sgld_logMHAccRatCorr,"%f \n", logMHAccRatCorr) ;
#endif
#if __file_svrg_are_sgld_sig2hat__
            if (ins_svrg_are_sgld_sig2hat != NULL)
                    fprintf(ins_svrg_are_sgld_sig2hat,"%f \n", sig2hat) ;
#endif
#if __file_svrg_are_sgld_sig2tilde__
            if (ins_svrg_are_sgld_sig2tilde != NULL)
                    fprintf(ins_svrg_are_sgld_sig2tilde,"%f \n", sig2tilde) ;
#endif

            }
		
	}

	/*
	 * FREE MEMORY ------------------------------------------------------------
	 * */
	printf("\n\n ***** FREE MEMORY ***** \n\n") ;

	destroy_random_parameters(randpar_aux) ;

	destroy_svrg_are_sgld_controle_variate( svrg_are_sgld_controle_variate_1 ) ;

	destroy_svrg_are_sgld_controle_variate( svrg_are_sgld_controle_variate_2 ) ;

	destroy_svrg_are_sgld_subdata( svrg_are_sgld_subdata_aux ) ;

	destroy_svrg_are_sgld_subdata( svrg_are_sgld_subdata ) ;

	destroy_svrg_are_sgld_parameters(svrg_are_sgld_parameters) ;
        
        destroy_random_parameters(randpar_1) ;

	destroy_random_parameters(randpar_2) ;

	destroy_data( data ) ;

	destroy_fixed_parameters(fixpar) ;

	/*
	 * CLOSE FILES ------------------------------------------------------------
	 * */
	printf("\n\n ***** CLOSE FILES ***** \n\n") ;

#if __file_svrg_are_sgld_sig2tilde__
	if (ins_svrg_are_sgld_sig2tilde != NULL) fprintf(ins_svrg_are_sgld_sig2tilde, "\n") ;
	if (ins_svrg_are_sgld_sig2tilde != NULL) fclose(ins_svrg_are_sgld_sig2tilde);
#endif

#if __file_svrg_are_sgld_sig2hat__
	if (ins_svrg_are_sgld_sig2hat != NULL) fprintf(ins_svrg_are_sgld_sig2hat, "\n") ;
	if (ins_svrg_are_sgld_sig2hat != NULL) fclose(ins_svrg_are_sgld_sig2hat);
#endif

#if __file_svrg_are_sgld_grad_log_lik_est_1__
	if (ins_svrg_are_sgld_grad_log_lik_est_1 != NULL) fprintf(ins_svrg_are_sgld_grad_log_lik_est_1, "\n") ;
	if (ins_svrg_are_sgld_grad_log_lik_est_1 != NULL) fclose(ins_svrg_are_sgld_grad_log_lik_est_1);
#endif

#if __file_svrg_are_sgld_grad_log_lik_est_2__
	if (ins_svrg_are_sgld_grad_log_lik_est_2 != NULL) fprintf(ins_svrg_are_sgld_grad_log_lik_est_2, "\n") ;
	if (ins_svrg_are_sgld_grad_log_lik_est_2 != NULL) fclose(ins_svrg_are_sgld_grad_log_lik_est_2);
#endif

#if __file_svrg_are_sgld_data__
	if (ins_svrg_are_sgld_data != NULL) fprintf(ins_svrg_are_sgld_data, "\n") ;
	if (ins_svrg_are_sgld_data != NULL) fclose(ins_svrg_are_sgld_data);
#endif

#if __file_svrg_are_sgld_theta_1__
	if (ins_svrg_are_sgld_theta_1 != NULL) fprintf(ins_svrg_are_sgld_theta_1, "\n") ;
	if (ins_svrg_are_sgld_theta_1 != NULL) fclose(ins_svrg_are_sgld_theta_1);
#endif

#if __file_svrg_are_sgld_theta_2__
	if (ins_svrg_are_sgld_theta_2 != NULL) fprintf(ins_svrg_are_sgld_theta_2, "\n") ;
	if (ins_svrg_are_sgld_theta_2 != NULL) fclose(ins_svrg_are_sgld_theta_2);
#endif

#if __file_svrg_are_sgld_logMHAccRatCorr__
	if (ins_svrg_are_sgld_logMHAccRatCorr != NULL) fprintf(ins_svrg_are_sgld_logMHAccRatCorr, "\n") ;
	if (ins_svrg_are_sgld_logMHAccRatCorr != NULL) fclose(ins_svrg_are_sgld_logMHAccRatCorr);
#endif

	/*
	 * DONE!!! -----------------------------------
	 * */
	printf("\n\n ***** DONE!!! ***** \n\n") ;
}












