
struct CouplingLayer{S, T}
    mask::BitVector
    s::S
    t::T
end

Flux.@functor CouplingLayer
Flux.trainable(m::CouplingLayer) = (m.s, m.t)

function dense_builder(ni, no, nh, nl, act) 
    @assert nl >= 2
    return Chain(Dense(ni, nh, act), map(_->Dense(nh, nh, act), 1:nl-2)..., Dense(nh, no, identity))
end

# function dense_builder(ni, no, nh, nl, act) 
#     @assert nl >= 2
#         return Chain(
#             Dense(ni, nh, act), BatchNorm(nh),
#             mapreduce(_->[Dense(nh, nh, act), BatchNorm(nh)], vcat, 1:nl-2; init=[])..., 
#             Dense(nh, no, identity))
# end


function CouplingLayer(xdim, hdim; nlayers=2, sact=tanh, tact=relu, even=true)
    mask = even ? (mod.(1:xdim, 2) .== 0) : (mod.(1:xdim, 2) .== 1)

    idim = sum(mask)
    odim = length(mask) - idim
    s = dense_builder(idim, odim, hdim, nlayers, sact)
    t = dense_builder(idim, odim, hdim, nlayers, tact)
    CouplingLayer(mask, s, t)
end

function _cat_with_mask(x1, x2, mask)
	M1, N = size(x1)
	M2, _ = size(x2)
	Y = similar(x1, M1+M2, N)
	Y[mask,:] .= x1
	Y[.~mask, :] .= x2
	Y
end

function _cat_with_mask_pullback(x1, x2, mask, Δy)
	Δy[mask,:], Δy[.~mask, :], NoTangent()
end

function ChainRulesCore.rrule(::typeof(_cat_with_mask), args...)
    y = _cat_with_mask(args...)
    pullback(Δy) = (NoTangent(), _cat_with_mask_pullback(args..., Δy)...)
    y, pullback
end

function forward(m::CouplingLayer, input)
    x, logJ = input
    x1 = x[m.mask, :]
    x2 = x[.~m.mask, :]
    sx_1 = m.s(x1)
    z2 = x2 .* exp.(sx_1) .+ m.t(x1)
    z =_cat_with_mask(x1, z2, m.mask)
    logJz = logJ .+ sum(sx_1, dims=1)
    return z, logJz 
end

function inverse(m::CouplingLayer, input)
    z, logJ = input
    z1 = z[m.mask, :]
    z2 = z[.~m.mask, :]
    sz_1 = m.s(z1)
    x2 = (z2 .-  m.t(z1)) .* exp.(-sz_1)
    x =_cat_with_mask(z1, x2, m.mask)
    logJx = logJ .- sum(sz_1, dims=1)
    return x, logJx 
end

(m::CouplingLayer)(input) = forward(m, input)
