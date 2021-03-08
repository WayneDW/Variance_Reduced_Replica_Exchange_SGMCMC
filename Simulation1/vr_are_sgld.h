/*
 * Copyrigtht 2020 Georgios Karagiannis
 *
 * This file is part of 'Variance Reduced Replica Exchange Stochastic Gradient
 * Langevin Dynamics' (VRRESGLD).
 *
 * VRRESGLD is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation version 3 of the License.
 *
 * VRRESGLD is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with VRRESGLD.  If not, see <http://www.gnu.org/licenses/>.
*/

/*
 * Georgios Karagiannis
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
 * SVRG ARE SGLD PARAMETERS
 */

typedef struct {
	double eta_1 ;
	double tau_1 ;
	double eta_2 ;
	double tau_2 ;
	int sig2hat_rep ;
	int gain_sig2hat_t0 ;
	double gain_sig2hat_c0 ;
	int subsample_type ;
	double Fscl ;
	int sig2hat_update_rate ;
	int CV_update_rate ;
} struct_svrg_are_sgld_parameters ;

void alloc_svrg_are_sgld_parameters(struct_svrg_are_sgld_parameters **) ;

void set_svrg_are_sgld_parameters(struct_svrg_are_sgld_parameters *) ;

void set_external_svrg_are_sgld_parameters(struct_svrg_are_sgld_parameters *, int , char *[]) ;

void print_svrg_are_sgld_parameters(struct_svrg_are_sgld_parameters *) ;

void destroy_svrg_are_sgld_parameters(struct_svrg_are_sgld_parameters *) ;

/*
 * SUB-DATA
 */

typedef struct {
	struct_data *data ;
	int en_ysub ;
	int *I_ysub ;
} struct_svrg_are_sgld_subdata ;

void alloc_svrg_are_sgld_subdata(struct_svrg_are_sgld_subdata **, struct_data *, int) ;

void sample_svrg_are_sgld_subdata(struct_svrg_are_sgld_subdata *,
							struct_data *data,
							struct_svrg_are_sgld_parameters *) ;

void print_svrg_are_sgld_subdata(struct_svrg_are_sgld_subdata *) ;

void destroy_svrg_are_sgld_subdata(struct_svrg_are_sgld_subdata *) ;

/*
 * GRADIENT ESTIMATOR
 */

void comp_svrg_are_sgld_grad_log_lik_estimate(double *  ,
										struct_svrg_are_sgld_subdata * ,
										struct_random_parameters*  ,
										struct_fixed_parameters*  ) ;

void comp_svrg_are_sgld_udpate_random_parameters(struct_svrg_are_sgld_subdata * ,
                                                struct_random_parameters*  ,
                                                struct_random_parameters* ,
                                                struct_fixed_parameters*  ,
                                                struct_svrg_are_sgld_parameters * ,
                                                double *,double *,double *,double *) ;

/*
 * Compute log likelihood estimate
 */

typedef struct {
	struct_random_parameters *randpar ;
	double log_lik ;
} struct_svrg_are_sgld_controle_variate ;

void alloc_svrg_are_sgld_controle_variate(struct_svrg_are_sgld_controle_variate **) ;

void destroy_svrg_are_sgld_controle_variate(struct_svrg_are_sgld_controle_variate *) ;

void  swap_svrg_are_sgld_control_variate(struct_svrg_are_sgld_controle_variate *,
                                        struct_svrg_are_sgld_controle_variate *,
                                        struct_random_parameters *) ;

void  comp_svrg_are_sgld_controle_variate(struct_svrg_are_sgld_subdata *,
									struct_random_parameters* ,
									struct_random_parameters* ,
									struct_fixed_parameters* ,
									struct_svrg_are_sgld_controle_variate *,
									struct_svrg_are_sgld_controle_variate *) ;

double  comp_svrg_are_sgld_log_lik_estimate(struct_svrg_are_sgld_subdata *,
									struct_random_parameters* ,
									struct_fixed_parameters*,
									struct_svrg_are_sgld_controle_variate *) ;

/*
 * Compute the stochastisity correction term
 */

double comp_svrg_are_sgld_gain_sig2hat(int , struct_svrg_are_sgld_parameters *) ;

double comp_svrg_are_sgld_sig2tilde(struct_svrg_are_sgld_subdata *,
								struct_random_parameters* ,
								struct_random_parameters* ,
								struct_fixed_parameters* ,
								struct_svrg_are_sgld_controle_variate *,
								struct_svrg_are_sgld_controle_variate *,
								struct_svrg_are_sgld_parameters *,
								struct_svrg_are_sgld_subdata *)  ;

void comp_svrg_are_sgld_sig2hat(double *,
                                double ,
                                double ) ;


/*
 * Compute the Metropolis Hastings
 */

void comp_svrg_are_sgld_acceptance_ratio(double *, double *,
                                        struct_svrg_are_sgld_subdata *,
                                        struct_random_parameters* ,
                                        struct_random_parameters* ,
                                        struct_fixed_parameters* ,
                                        struct_svrg_are_sgld_controle_variate *,
                                        struct_svrg_are_sgld_controle_variate *,
                                        struct_svrg_are_sgld_parameters *,
                                        double ) ;



