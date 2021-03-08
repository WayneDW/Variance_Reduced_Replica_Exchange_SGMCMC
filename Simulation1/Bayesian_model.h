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

typedef struct {
	int en_y ;
	double *y ;
} struct_data ;

typedef struct {
	int dim_theta ;
	double theta ;
} struct_random_parameters ;

typedef struct {
	double mu_0 ;
	double sig2_0 ;
	double weight_real ;
	double mu_real ;
	double sig2_real ;
	double gamma_real ;
} struct_fixed_parameters ;

/*
 * DATA
 * */

void alloc_data(struct_data **, int ) ;

void alloc_and_generate_data(struct_data **, struct_fixed_parameters *, int ) ;

void destroy_data(struct_data *) ;

/*
 * RANDOM PARAMETERS
 * */

void alloc_random_parameters(struct_random_parameters **) ;

void seed_random_parameters(struct_random_parameters *) ;

void set_external_random_parameters(struct_random_parameters *, int, char *[]) ;

void print_random_parameters(struct_random_parameters *) ;

void destroy_random_parameters(struct_random_parameters *) ;

void copy_random_parameters(struct_random_parameters *,
		struct_random_parameters *) ;

void swap_random_parameters(struct_random_parameters *,
							struct_random_parameters *,
							struct_random_parameters *) ;

/*
* FIXED PARAMETERS
* */

void alloc_fixed_parameters(struct_fixed_parameters **) ;

void initialise_fixed_parameters(struct_fixed_parameters *) ;

void set_external_fixed_parameters(struct_fixed_parameters *, int , char *[]) ;

void print_fixed_parameters(struct_fixed_parameters *) ;

void destroy_fixed_parameters(struct_fixed_parameters *) ;

/*
 * PRIORS
 * */

double comp_log_prior(struct_random_parameters* ,
						struct_fixed_parameters* ) ;

double comp_prior(struct_random_parameters* ,
					struct_fixed_parameters* ) ;

void comp_gradient_log_prior(double *,
								struct_random_parameters* ,
								struct_fixed_parameters* ) ;

/*
 * LIKELIHOOD
 * */

double comp_unit_log_lik(double , struct_random_parameters* ,
							struct_fixed_parameters* ) ;

double comp_unit_lik(double , struct_random_parameters* ,
		struct_fixed_parameters* ) ;

double comp_gradient_unit_log_lik(double , struct_random_parameters* ,
							struct_fixed_parameters* ) ;

/*
 * POSTERIOR
 * */

double comp_log_posterior(struct_data* , struct_random_parameters* ,
							struct_fixed_parameters* ) ;








