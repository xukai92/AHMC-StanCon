@model LR(x, y, Nd, Nc) = begin
    B ~ MvNormal(zeros(Nc), 10)
    B0 ~ Normal(0, 10)
    sigma ~ Truncated(Cauchy(0, 5), 0, Inf)
    mu = B0 .+ x * B
    y ~ MvNormal(mu, sigma)
end
x, y, Nd, Nc = ... # load data
chain = sample(LR(x, y, Nd, Nc), NUTS(2_000, 1_000, 0.8))