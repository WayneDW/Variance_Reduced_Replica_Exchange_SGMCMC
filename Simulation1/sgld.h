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
	double eta ;
	double tau ;
	int subsample_type ;
} struct_sgld_parameters ;

typedef struct {
	struct_data *data ;
	int en_ysub ;
	int *I_ysub ;
} struct_sgld_subdata ;

/*
 * SGLD PARAMETERS
 */

void alloc_sgld_parameters(struct_sgld_parameters **) ;

void set_sgld_parameters(struct_sgld_parameters *) ;

void set_external_sgld_parameters(struct_sgld_parameters *, int, char *[]) ;

void print_sgld_parameters(struct_sgld_parameters *) ;

void destroy_sgld_parameters(struct_sgld_parameters *) ;

/*
 * SUB-DATA
 */

void alloc_sgld_subdata(struct_sgld_subdata **, struct_data *, int) ;

void sample_sgld_subdata(struct_sgld_subdata *,
							struct_data *data,
							struct_sgld_parameters *) ;

void print_sgld_subdata(struct_sgld_subdata *) ;

void destroy_sgld_subdata(struct_sgld_subdata *) ;

/*
 * GRADIENT ESTIMATOR
 */

void comp_sgld_grad_log_lik_estimate(double *  ,
										struct_sgld_subdata * ,
										struct_random_parameters*  ,
										struct_fixed_parameters*  ) ;

void comp_sgld_udpate_random_parameters(struct_sgld_subdata * ,
                                        struct_random_parameters*  ,
                                        struct_fixed_parameters*  ,
                                        struct_sgld_parameters *  ,
                                        double *, double *) ;


