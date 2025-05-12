import torch
from torch.distributions import Categorical
from tqdm.notebook import tqdm
import numpy as np

class AnnealedSMC:
    def __init__(
        self,
        N,
        x_dim,
        sigma_0,
        sigma_target,
        alpha=0.5,
        mala_step_size=0.1,
        mala_steps=5,
        ess_tol=1e-2,
        ess_min_frac=0.5,
        device='cuda'
    ):
        """
        N             : number of particles
        x_dim         : dimension of x
        sigma_0       : initial (largest) σ
        sigma_target  : final (smallest) σ
        alpha         : resampling threshold fraction (i.e. resample when ESS/N < alpha)
        mala_step_size: MALA step size ε (can be a function of σ)
        mala_steps    : number of MALA iterations per SMC stage
        ess_tol       : tolerance for bisection when adapting σ
        ess_min_frac  : minimal ESS fraction to keep at each stage (e.g. 0.5)
        """
        self.N = N
        self.x_dim = x_dim
        self.sigma_0 = sigma_0
        self.sigma_target = sigma_target
        self.sigma_prev = None # None indicates that we should use the initial logp
        self.alpha = alpha
        self.mala_step_size = mala_step_size
        self.mala_steps = mala_steps
        self.ess_tol = ess_tol
        self.ess_min = ess_min_frac * N
        self.device = device

        # allocate
        self.X = torch.zeros(N, x_dim, device=device, dtype=torch.float32)
        self.logw = torch.zeros(N, device=device, dtype=torch.float32)  # unnormalized log-weights

    def initialize(self, init_sampler, init_logp):
        """
        init_sampler: a function taking (N, x_dim) and returning a tensor of shape (N, x_dim).
        Typically you sample from a very flat p_{σ_0} or from a simple prior.
        """
        self.X = init_sampler(self.N, self.x_dim).to(self.device)
        self.logw.zero_()

    @staticmethod
    def compute_ess(logw):
        w = torch.softmax(logw, 0)
        return 1.0 / torch.sum(w * w)

    def ess_at(self, sigma, log_target, init_logp):
        if self.sigma_prev is not None:
            delta = log_target(self.X, sigma) - log_target(self.X, self.sigma_prev)
            lw = self.logw + delta
        else:
            lw = log_target(self.X, sigma) - init_logp(self.X)
        return self.compute_ess(lw)

    def adapt_sigma(self, log_target, init_logp):
        """
        Find the next σ_t < σ_{t-1} by bisection so that ESS >= ess_min.
        """
        # currently this just gets stuck at the first sigma value

        lo = self.sigma_target
        hi = self.sigma_prev if self.sigma_prev is not None else self.sigma_0

        # if even at target we have enough ESS, we can jump all the way
        if self.ess_at(lo, log_target, init_logp) >= self.ess_min:
            return lo

        # otherwise bisect
        while hi - lo > self.ess_tol:
            mid = 0.5 * (hi + lo)
            if self.ess_at(mid, log_target, init_logp) < self.ess_min:
                # too aggressive → need smaller step (i.e. larger σ)
                lo = mid
            else:
                hi = mid
        return hi

    def resample(self):
        """
        Multinomial resampling (you can swap in systematic, stratified, etc.)
        """
        w = torch.softmax(self.logw, 0)
        idx = Categorical(w).sample((self.N,))
        self.X = self.X[idx]
        self.logw.zero_()

    def mala_kernel(self, X, sigma, log_target, grad_log_target):
        """
        One full MALA sweep on all particles.  Returns updated X.
        """
        if callable(self.mala_step_size):
            eps = self.mala_step_size(sigma)
        else:
            eps = self.mala_step_size
        for _ in range(self.mala_steps):
            with torch.no_grad(): # TODO: figure out which thing requires_grad
                grad = grad_log_target(X, sigma)   # shape (N, x_dim)
                noise = torch.randn_like(X)
                X_prop = X + 0.5 * eps**2 * grad + eps * noise

                # compute Metropolis–Hastings acceptance
                log_u = torch.log(torch.rand(self.N, device=self.device))
                logp_curr = log_target(X, sigma)
                logp_prop = log_target(X_prop, sigma)

                # proposal correction q(x|x') vs q(x'|x)
                # forward: X -> X_prop
                diff_fwd = X_prop - X - 0.5 * eps**2 * grad
                log_q_fwd = -0.5 * (diff_fwd**2).sum(dim=1) / eps**2

                grad_prop = grad_log_target(X_prop, sigma)
                diff_rev = X - X_prop - 0.5 * eps**2 * grad_prop
                log_q_rev = -0.5 * (diff_rev**2).sum(dim=1) / eps**2

                log_accept_ratio = (logp_prop + log_q_rev) - (logp_curr + log_q_fwd)
                accept = (log_u < log_accept_ratio).unsqueeze(1)

                # update
                X = torch.where(accept, X_prop, X)
        return X

    def run(self, init_sampler, init_logp, log_target, grad_log_target, debug_logger=None):
        """
        Runs the full adaptive‐SMC‐with‐MALA algorithm, returns final particles X.
        """
        # 1. initialize
        self.initialize(init_sampler, init_logp)

        progress = tqdm(range(100), desc=f"Running SMC (ESS = {self.N})")

        # we want to display progress in log-sigma space
        def convert_sigma_to_index(sigma):
            return 100 - int(np.log(sigma / self.sigma_target) / np.log(self.sigma_0 / self.sigma_target) * 100)
        
        curr_index = convert_sigma_to_index(self.sigma_0)

        t = 0
        # 2. sequential tempering
        while self.sigma_prev is None or self.sigma_prev > self.sigma_target + self.ess_tol:
            # 2a. adapt σ_t
            sigma_t = self.adapt_sigma(log_target, init_logp)

            print(f"sigma_{t}: {sigma_t}")

            # 2b. weight‐update
            if self.sigma_prev is not None:
                delta = log_target(self.X, sigma_t) - log_target(self.X, self.sigma_prev)
            else:
                delta = log_target(self.X, sigma_t) - init_logp(self.X)
            self.logw += delta

            # 2c. check ESS → maybe resample
            ess = self.compute_ess(self.logw)
            # print(f"current ESS: {ess}")
            progress.set_description(f"Running SMC (ESS = {ess:.4e})")
            if ess < self.alpha * self.N or self.sigma_prev == sigma_t:
                # print("resampling...")
                self.resample()

            # 2d. mutate with MALA
            self.X = self.mala_kernel(self.X, sigma_t, log_target, grad_log_target)

            # 2e. advance
            self.sigma_prev = sigma_t
            t += 1

            # update progress bar
            new_index = convert_sigma_to_index(self.sigma_prev)
            print(f"new_index: {new_index}, curr_index: {curr_index}")
            if new_index > curr_index:
                progress.update(new_index - curr_index)
                curr_index = new_index

            if debug_logger is not None:
                debug_logger(self.X, self.sigma_prev)

        return self.X