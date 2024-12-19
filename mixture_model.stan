// Stan model
data {
int N; // sample size
int D; // dimension of observed vars
int K; // number of latent groups
vector[D] y[N]; // data
}
parameters {
  ordered[K] mu;          // Locations of hidden states
  vector<lower=0>[K] sigma; // Variances of hidden states
  simplex[K] theta[D];    // Mixture proportions
}
model {
  vector[K] obs[D];

  // Priors
  for (k in 1:K) {
    mu[k] ~ normal(0, 10);
    sigma[k] ~ inv_gamma(1, 1);
  }
  for (d in 1:D) {
    theta[d] ~ dirichlet(rep_vector(2.0, K));
  }

  // Likelihood
  for (d in 1:D) {
    for (n in 1:N) {
      for (k in 1:K) {
        obs[d][k] = log(theta[d][k]) + normal_lpdf(y[n][d] | mu[k], sigma[k]);
      }
      target += log_sum_exp(obs[d]);
    }
  }
}
