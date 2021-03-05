# This is a similar code for writing the simulation part of this paper.

set.seed(66)

# Gaussian mixture simulation following Dubey: Variance reduced SGLD
n = 1000
true_mu = -5
gamma = 20
sd = 5
prob_mixture = rbinom(n, 1, 0.5)
data = prob_mixture * rnorm(n, true_mu, sd) + (1 - prob_mixture) * rnorm(n, (-true_mu+gamma), sd)

# build likelihood function

neg_log_likelihood = function(par) sum(-log(0.5 * dnorm(data, par, sd) + 0.5 * dnorm(data, (-par+gamma), sd)))

########################### Variance reduced reSGLD (VR-reSGLD) ############################
# Run SGLD
VR_type = -1

# Run reSGLD
VR_type = 0

# Run VR-reSGLD
VR_type = 1      
################################################################################
batch_size = 100
mu_high = 0
mu_low = 0
mu_list = c()
T_high = 1000
T_low = 25
# estimate the initial variance
init_losses = c()
for (idx_est_var in 1:20) {
	batch_data = data[sample(n, size=batch_size, replace=FALSE)]
	noisy_f = function(par) n / batch_size * sum(-log(0.5 * dnorm(batch_data, par, sd) + 0.5 * dnorm(batch_data, (-par+gamma), sd)))
    init_losses = c(init_losses, noisy_f(mu_low))
}
hat_var = var(init_losses)
smooth_factor = 0.2
swaps = 0
F = 1
if (VR_type == 1) {
	vr_mu_low = mu_low
	vr_mu_high = mu_high
	exact_f_low = sum(-log(0.5 * dnorm(data, vr_mu_low, sd) + 0.5 * dnorm(data, (-vr_mu_low+gamma), sd)))
	exact_f_high = sum(-log(0.5 * dnorm(data, vr_mu_high, sd) + 0.5 * dnorm(data, (-vr_mu_high+gamma), sd)))
}

avg_var = c()
for (iter in 1: 1e5) {
	lr = 0.002
	if (iter %% 200 == 0) {
		# conduct VR-reSGLD
		if (VR_type == 1) {
	        losses_low = c()
	        losses_high = c()
	        for (idx_est_var in 1:20) {
	        	batch_data = data[sample(n, size=batch_size, replace=FALSE)]
	        	noisy_f = function(par) n / batch_size * sum(-log(0.5 * dnorm(batch_data, par, sd) + 0.5 * dnorm(batch_data, (-par+gamma), sd)))
      			losses_low = c(losses_low, (noisy_f(mu_low) - noisy_f(vr_mu_low) + exact_f_low))
      			losses_high = c(losses_high, (noisy_f(mu_high) - noisy_f(vr_mu_high) + exact_f_high))
	        }
	        var_low = var(losses_low)
	        var_high = var(losses_high)
	        hat_var = (1 - smooth_factor) * hat_var + smooth_factor * (var_high + var_low) / 2
			vr_mu_low = mu_low
			vr_mu_high = mu_high
			exact_f_low = sum(-log(0.5 * dnorm(data, vr_mu_low, sd) + 0.5 * dnorm(data, (-vr_mu_low+gamma), sd)))
			exact_f_high = sum(-log(0.5 * dnorm(data, vr_mu_high, sd) + 0.5 * dnorm(data, (-vr_mu_high+gamma), sd)))
		}
		if (VR_type <= 0) {
			# adaptive variance estimate
	        losses_low = c()
	        losses_high = c()
	        for (idx_est_var in 1:20) {
	        	batch_data = data[sample(n, size=batch_size, replace=FALSE)]
	        	noisy_f = function(par) n / batch_size * sum(-log(0.5 * dnorm(batch_data, par, sd) + 0.5 * dnorm(batch_data, (-par+gamma), sd)))
        		losses_low = c(losses_low, noisy_f(mu_low))
            	losses_high = c(losses_high, noisy_f(mu_high))
	        }
	        var_low = var(losses_low)
	        var_high = var(losses_high)
	        hat_var = (1 - smooth_factor) * hat_var + smooth_factor * (var_high + var_low) / 2
		}
		avg_var = c(avg_var, ((var_high + var_low) / 2))
    }
    idxs = sample(n, size=batch_size, replace=FALSE)
    batch_data = data[idxs]
	noisy_f = function(par) n / batch_size * sum(-log(0.5 * dnorm(batch_data, par, sd) + 0.5 * dnorm(batch_data, (-par+gamma), sd)))
	stochastic_gradient_low = numDeriv::grad(noisy_f, mu_low)
	stochastic_gradient_high = numDeriv::grad(noisy_f, mu_high)
	mu_high = mu_high - lr * stochastic_gradient_high + sqrt(2 * lr * T_high)  * rnorm(1, 0, 1)
	mu_low  = mu_low  - lr * stochastic_gradient_low + sqrt(2 * lr * T_low)  * rnorm(1, 0, 1)

	# conduct VR-reSGLD
	if (VR_type == 1) {
		vr_f_low = noisy_f(mu_low) - noisy_f(vr_mu_low) + exact_f_low
		vr_f_high = noisy_f(mu_high) - noisy_f(vr_mu_high) + exact_f_high
		accept_prob = min(1, exp((1 / T_high - 1 / T_low) * (vr_f_high - vr_f_low - (1 / T_high - 1 / T_low) * hat_var / F)))
	} 
	if (VR_type == 0) accept_prob = min(1, exp((1 / T_high - 1 / T_low) * (noisy_f(mu_high) - noisy_f(mu_low) - (1 / T_high - 1 / T_low) * hat_var / F)))
	if (VR_type == -1) accept_prob = -1

	if (runif(1) < accept_prob) {
        tmp = mu_low
        mu_low = mu_high
        mu_high = tmp
        swaps = swaps + 1
    }
    if (VR_type == 2) {
	    g_alpha_low = g_alpha_low + (noisy_f(mu_low) - noisy_f(alpha_low[idxs])) * batch_size / n
	    g_alpha_high = g_alpha_high + (noisy_f(mu_high) - noisy_f(alpha_high[idxs])) * batch_size / n
	    alpha_low[idxs] = mu_low
	    alpha_high[idxs] = mu_high
    }

    if (iter %% 1e3 == 1) mu_temp = c()
    mu_temp = c(mu_temp, mu_low)
	if (iter %% 1e3 == 0) {
		mu_list = c(mu_list, mu_temp)
		options(digits=3)
		hist(mu_list, 100, main=paste0("Epochs ", iter, " Swaps ", swaps, " Hat Var ", as.integer(hat_var)))
	}
}
quantile(avg_var, probs = seq(0, 1, 0.1))
