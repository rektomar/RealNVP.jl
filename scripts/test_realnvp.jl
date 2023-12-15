using DrWatson
@quickactivate

using RealNVP
using Flux
using Plots
using Random
using Statistics
import Flux.Optimise: ADAM, update!
Random.seed!(1234)

function sample_banana(n; c=[1f0, 4f0])
    z = randn(Float32, 2, n)
    x = zeros(Float32, 2, n)
    x[1,:] = z[1,:] ./ c[1]
    x[2,:] = z[2,:].*c[1] + c[1].*c[2].*(z[1,:].^2 .+ c[1]^2)
    return x
end

loss(m::RealNVPModel, x::Matrix) = -mean(logpdf(m, x))

ndata = 500
x = sample_banana(ndata)

nepoc = 100
idim = 2
hdim = 4
nlayers = 2
nflows = 6

m = RealNVPModel(idim, hdim; nlayers=nlayers, nflows=nflows)
ps = Flux.params(m)
lr = 1f-2
opt = Adam(lr)

data = Flux.Data.DataLoader(x, batchsize=64)

println("e: 0, loss: $(loss(m, x))")
for e in 1:nepoc
    for xb in data
        gs = gradient(()->loss(m, xb), ps)
        Flux.Optimise.update!(opt, ps, gs)
    end
    println("e: $e, loss: $(loss(m, x))")
end

z_data = forward(m, x)
z_new = randn(Float32, 2, 200)
x_new = inverse(m, z_new)

p1 = scatter(x[1, :], x[2, :], label="x_data")
scatter!(p1, x_new[1, :], x_new[2, :], label="x_new = f^{-1}(z_new)")


p2 = scatter(z_data[1, :], z_data[2, :], label="z_data = f(x_data)")
scatter!(p2, z_new[1, :], z_new[2, :], label="z_new ~ N(0, 1)")


plot(p1, p2)