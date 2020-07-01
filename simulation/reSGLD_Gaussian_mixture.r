set.seed(1)


# case 1: t-oracles; case 2: normal oracles.
case = 1
# energy function for the Gaussian mixture distribution
if (case == 1) {
    mean1 = -4
    mean2 = 3
    f = function(x) return(-log(0.4 * dnorm(x, mean=mean1, sd=0.7) + 0.6 * dnorm(x, mean=mean2, sd=0.5)))
    noisy_f = function(x) return(f(x) + rt(1, df=5))
} else {
    mean1 = -3
    mean2 = 2
    f = function(x) return(-log(0.4 * dnorm(x, mean=mean1, sd=0.7) + 0.6 * dnorm(x, mean=mean2, sd=0.5)))
    noisy_f = function(x) return(f(x) + rnorm(1, 0, 2))
}
# target Gaussian mixture distribution
density = function(x) exp(-f(x))


# Important hyperparameters
lr = 0.03
T_high = 10
T_low = 1

# Other hyperparameters
x_low = 0
x_high = 0
part1 = 100
grad_std = 0.1
thinning = 100

# You can set a larger value to obtain more stable results.
total = 100000

################################################### SGLD ###################################################
samples_sgld = c()
x_low = 0
for (i in 1:part1) {
    samples_part = c()
    for (j in 1:(total/part1)) {
        x_low  = x_low  - lr * (numDeriv::grad(f, x_low) + rnorm(1, 0, grad_std))  + sqrt(2 * lr * T_low)  * rnorm(1, 0, 1)
    
        if (j %% (thinning) == 0) {samples_part = c(samples_part, x_low)}
    }
    print(i)
    samples_sgld = c(samples_sgld, samples_part)
}



################################################### Naive reSGLD ###################################################
samples_naive = c()
x_low = 0
x_high = 0
for (i in 1:part1) {
    samples_part = c()
    for (j in 1:(total/part1)) {
        x_low  = x_low  - lr * (numDeriv::grad(f, x_low) + rnorm(1, 0, grad_std))  + sqrt(2 * lr * T_low)  * rnorm(1, 0, 1)
        x_high = x_high - lr * (numDeriv::grad(f, x_high) + rnorm(1, 0, grad_std)) + sqrt(2 * lr * T_high) * rnorm(1, 0, 1)
        
        integrand_corrected = min(1, exp((1 / T_high - 1 / T_low) * (noisy_f(x_high) - noisy_f(x_low))))
        
        if (runif(1) < integrand_corrected) {
            tmp = x_low
            x_low = x_high
            x_high = tmp
        }
        if (j %% thinning == 0) {samples_part = c(samples_part, x_low)}
    }
    print(i)
    samples_naive = c(samples_naive, samples_part)
}


################################################### reSGLD ###################################################
samples_ptsgld = c()
hat_var = 10
counter = 1
x_low = 0
x_high = 0
for (i in 1:part1) {
    samples_part = c()
    for (j in 1:(total/part1)) {
        x_low  = x_low  - lr * (numDeriv::grad(f, x_low) + rnorm(1, 0, grad_std))  + sqrt(2 * lr * T_low)  * rnorm(1, 0, 1)
        x_high = x_high - lr * (numDeriv::grad(f, x_high) + rnorm(1, 0, grad_std)) + sqrt(2 * lr * T_high) * rnorm(1, 0, 1)
        
        if (j %% 20 == 0) {
            loss_stat = c()
            for (jjj in 1:10) {
                loss_stat = c(loss_stat, noisy_f(x_low))
            }
            unbiased_var = var(loss_stat)
            hat_var = (1 - 1 / counter) * hat_var + (1 / counter) * unbiased_var
            counter = counter + 1
        }
        
        integrand_corrected = min(1, exp((1 / T_high - 1 / T_low) * (noisy_f(x_high) - noisy_f(x_low) - (1 / T_high - 1 / T_low) * hat_var)))
        
        if (runif(1) < integrand_corrected) {
            tmp = x_low
            x_low = x_high
            x_high = tmp
        }
        if (j %% thinning == 0) {samples_part = c(samples_part, x_low)}
    }
    print(i)
    samples_ptsgld = c(samples_ptsgld, samples_part)
}


real_samples = c(rnorm(length(samples_ptsgld)*0.4, mean=mean1, sd=0.7), rnorm(length(samples_ptsgld)*0.6, mean=mean2, sd=0.5))

wdata = data.frame(
        Type = factor(rep(c("Ground truth", "SGLD", "Naive reSGLD", "reSGLD"), each=length(samples_ptsgld))),
        weight = c(real_samples, samples_sgld, samples_naive, samples_ptsgld)
        )

library(ggplot2)

p=ggplot(wdata, aes(x = weight)) +
    stat_density(aes(x=weight, colour=Type, linetype=Type), size=2, geom="line", position="identity") +
    scale_linetype_manual(values=c("solid", "dotdash", "longdash", "twodash"))+
    scale_color_manual(values = c("#666666", "#7570B3", "#E7298A", "#1B9E77")) +
    scale_x_continuous(name="X") +
    scale_y_continuous(name="Density") +
    theme(
        legend.position = c(0.3, 0.8), 
        legend.title =element_blank(), 
        legend.text = element_text(colour="grey15", size=24),
        legend.key.size = unit(2,"line"),
        legend.key = element_blank(),
        legend.background=element_rect(fill=alpha('grey', 0.1)),
        axis.title=element_text(size=24),
        axis.text.y = element_text(size=24),
        axis.text.x = element_text(size=24)
        )
p







