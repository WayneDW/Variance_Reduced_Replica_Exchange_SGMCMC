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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>

#include "nrutil.h"
#include "logPDF.h"
#include "RNG.h"
#include "Bayesian_model.h"
#include "sgld.h"

#ifndef __file_sgld_data__
	#define __file_sgld_data__ 1
#endif
#ifndef __file_sgld_theta__
	#define __file_sgld_theta__ 1
#endif
#ifndef __file_sgld_grad_log_lik_est__
	#define __file_sgld_grad_log_lik_est__ 0
#endif

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
	
	int Qepoch ;
	int epoch ;

	struct_fixed_parameters *fixpar ;

	struct_random_parameters *randpar ;

	struct_sgld_parameters *sgld_parameters ;

	int en_y ;
	struct_data *data ;

	int en_ysub ;
	struct_sgld_subdata *sgld_subdata ;

	double sgld_grad_log_lik_est ;
        double sgld_grad_log_prior ;

	int iter_mcmc ;
	int N_mcmc ;

	char output_dir[100] ;
	char file_name[100] ;
#if __file_sgld_data__
	FILE *ins_data = NULL ;
#endif
#if __file_sgld_theta__
	FILE *ins_chain_theta = NULL ;
#endif
#if __file_sgld_grad_log_lik_est__
	FILE *ins_chain_sgld_grad_log_lik_est = NULL ;
#endif

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
	struct timeval t1;
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
	
	snprintf(output_dir, sizeof output_dir, "%s", "./output_files_sgld");
	for (i = 1; i < argc; i++)
		if (strcmp("-output_path", argv[i]) == 0)
			snprintf(output_dir, sizeof output_dir, "%s", argv[++i]);

#if __file_sgld_data__
	snprintf(file_name, sizeof file_name, "%s/sgld_data.dat",
			output_dir);
	ins_data = fopen( file_name , "w" ) ;
	printf("==> %s \n", file_name) ;
#endif

#if __file_sgld_theta__
	snprintf(file_name, sizeof file_name, "%s/sgld_theta.out",
			output_dir);
	ins_chain_theta = fopen( file_name , "w" ) ;
	printf("==> %s \n", file_name) ;
#endif

#if __file_sgld_grad_log_lik_est__
	snprintf(file_name, sizeof file_name, "%s/sgld_grad_log_lik_est.out",
			output_dir);
	ins_chain_sgld_grad_log_lik_est = fopen( file_name , "w" ) ;
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
	
	/* .. default */
	en_y = 1000000 ;

	/* .. external */
	for (i = 1; i < argc; i++)
		if (strcmp("-data->en_y", argv[i]) == 0)
			en_y = atoi(argv[++i]) ;

	/*fix RNG*/
	rng_seed = 1983000 ;
	setseedrng( (unsigned long) rng_seed ) ;
	for ( i=1 ; i<=10 ; i++ ) un = uniformrng() ;

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
#if __file_sgld_data__
	if (ins_data != NULL)
		for (i=1; i<=data->en_y; i++)
			fprintf(ins_data,"%f \n", data->y[i]) ;
#endif 
	/*
	 * SET RANDOM BAYESIAN MODEL PARAMETERS --------------------------------
	 * */
	printf("\n\n ***** SET RANDOM BAYESIAN MODEL PARAMETERS ***** \n\n") ;

	/* .. allocate */
	alloc_random_parameters( &randpar) ;

	/* .. initialise */
	seed_random_parameters(randpar) ;

	set_external_random_parameters( randpar, argc, argv ) ; 

	/* .. print */
	print_random_parameters(randpar) ;

	/*
	 * SET SGLD MCMC PARAMETERS --------------------------------
	 * */
	printf("\n\n ***** SET SGLD MCMC PARAMETERS ***** \n\n") ;

	/* .. allocate */
	alloc_sgld_parameters( &sgld_parameters ) ;

	/* .. initialise */
	set_sgld_parameters( sgld_parameters ) ;

	N_mcmc = 1000000 ;
	for (i = 1; i < argc; i++)
		if (strcmp("-N_mcmc", argv[i]) == 0) N_mcmc = atoi(argv[++i]) ;

	set_external_sgld_parameters(sgld_parameters, argc, argv ) ;

	/* .. print */
	printf("N_mcmc:  \t %d \n", N_mcmc) ;
	print_sgld_parameters( sgld_parameters ) ;

	/*
	 * SET SGLD SUBDATA PARAMETERS --------------------------------
	 */
	printf("\n\n ***** SET SGLD SUBDATA PARAMETER ***** \n\n") ;

	en_ysub = 1000 ;

	for (i = 1; i < argc; i++)
		if (strcmp("-sgld_subdata->en_ysub", argv[i]) == 0)
			en_ysub = atoi(argv[++i]) ;

	/* .. alloc */
	/* .. it points to the data !!!!	 */
	alloc_sgld_subdata( &sgld_subdata, data, en_ysub ) ;

	/* .. sample */
	sample_sgld_subdata( sgld_subdata, data, sgld_parameters) ;

	/* .. print */
	print_sgld_subdata(sgld_subdata) ;

	/*
	 * PERFORM THE SGLD ITERATIONS --------------------------------
	 * */
	printf("\n\n ***** SET SGLD MCMC PARAMETERS ***** \n\n") ;

	//if ( 0 )
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

		sample_sgld_subdata( sgld_subdata, data, sgld_parameters) ;

		/*
		 * Update randpar
		 * */

		comp_sgld_udpate_random_parameters(sgld_subdata,
                                                    randpar,
                                                    fixpar,
                                                    sgld_parameters,
                                                    &sgld_grad_log_lik_est, 
                                                    &sgld_grad_log_prior) ;

		/*
		 * Record
		 * */
		
		#if __file_sgld_theta__
		if (ins_chain_theta != NULL)
			fprintf(ins_chain_theta,"%f \n", randpar->theta) ;
		#endif

		Qepoch = comp_epoch(&epoch, iter_mcmc, 
				sgld_subdata->en_ysub, 
				data->en_y) ;
		
		if (Qepoch) {

printf("%f \n",randpar->theta) ;
                    
#if __file_sgld_grad_log_lik_est__
		if (ins_chain_sgld_grad_log_lik_est != NULL)
			fprintf(ins_chain_sgld_grad_log_lik_est,"%f \n", sgld_grad_log_lik_est) ;
#endif
		}
		
	}

	/*
	 * FREE MEMORY ------------------------------------------------------------
	 * */
	printf("\n\n ***** FREE MEMORY ***** \n\n") ;

	destroy_sgld_subdata( sgld_subdata ) ;

	destroy_sgld_parameters(sgld_parameters) ;

	destroy_random_parameters(randpar) ;

	destroy_data( data ) ;

	destroy_fixed_parameters(fixpar) ;

	/*
	 * CLOSE FILES ------------------------------------------------------------
	 * */
	printf("\n\n ***** CLOSE FILES ***** \n\n") ;

#if __file_sgld_data__
	if (ins_data != NULL) fprintf(ins_data, "\n") ;
	if (ins_data != NULL) fclose(ins_data);
#endif

#if __file_sgld_theta__
	if (ins_chain_theta != NULL) fprintf(ins_chain_theta, "\n") ;
	if (ins_chain_theta != NULL) fclose(ins_chain_theta);
#endif

#if __file_sgld_grad_log_lik_est__
	if (ins_chain_sgld_grad_log_lik_est != NULL) fprintf(ins_chain_sgld_grad_log_lik_est, "\n") ;
	if (ins_chain_sgld_grad_log_lik_est != NULL) fclose(ins_chain_sgld_grad_log_lik_est);
#endif
	
	/*
	 * DONE!!! -----------------------------------
	 * */
	printf("\n\n ***** DONE!!! ***** \n\n") ;
}



















