module RealNVP

using Flux
using Random
using ChainRulesCore
import SumProductSet: Distribution, logpdf

include("base.jl")
include("coupling.jl")

struct RealNVPModel{T} <: Distribution
    flows::T
end

Flux.@functor RealNVPModel
Flux.trainable(m::RealNVPModel) = (m.flows)

function RealNVPModel(idim, hdim; nflows=2, nlayers=2, sact=tanh, tact=relu)
    pattern = mod.(1:nflows, 2) .== 1
    flows = [CouplingLayer(idim, hdim; nlayers=nlayers, sact=sact, tact=tact, even=pattern[i]) for i in 1:nflows]
    RealNVPModel(Chain(flows...))
end

function logpdf(m::RealNVPModel, x::Matrix{T}) where {T<:Real} 
    logJ = zeros(T, 1, size(x, 2))
    input = (x, logJ)
    z, logJz = m.flows(input)

    logpdf_base(z) + logJz
end

function forward(m::RealNVPModel, x::Matrix{T}) where {T<:Real}
    logJ = zeros(T, 1, size(x, 2))
    input = (x, logJ)
    z, _ = m.flows(input)
    return z
end

function inverse(m::RealNVPModel, z::Matrix{T}) where {T<:Real}
    logJ = zeros(T, 1, size(z, 2))
    input = (z, logJ)
    x, _ = inverse_flow(reverse(m.flows.layers), input)
    return x
end

inverse_flow(::Tuple{}, input) = input
inverse_flow(fs::Tuple, input) = inverse_flow(Base.tail(fs), inverse(first(fs), input))

Base.length(m::RealNVPModel) = m.flows[1].mask |> length
Base.rand(m::RealNVPModel, n::Int) = inverse(m, randn(Float32, length(m), n))
Base.rand(m::RealNVPModel) = rand(m, 1)


export RealNVPModel, CouplingLayer
export forward, inverse, logpdf

end # module RealNVP
