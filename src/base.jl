
function logpdf_base(x::Matrix{T}) where {T<:Real}
    k = size(x, 1)
    -sum(x.^2, dims=1)/2 .- log(2*T(pi))*k/2
end

