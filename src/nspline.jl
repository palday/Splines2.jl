struct NSplineBasis{T<:Real} <: AbstractSplineBasis{T}
    b_spline_basis::BSplineBasis{T}
    qmat::Array{T,2}
    tl0::Array{T,1}
    tl1::Array{T,1}
    tr0::Array{T,1}
    tr1::Array{T,1}
end


function NSplineBasis(boundary_knots::Tuple{T,T},
                      interior_knots::Union{Array{T,1},Nothing}=nothing,
                      order::Int=4,
                      intercept::Bool=false)::NSplineBasis{T} where {T<:Real}
    bs = BSplineBasis(boundary_knots, interior_knots, order, intercept)
    co = basis(bs, [bs.boundary_knots[1], bs.boundary_knots[2]], 2)
    qmat_ = LinearAlgebra.qr(transpose(co)).Q
    qmat = collect(transpose(qmat_[:, 3:size(qmat_, 2)]))
    tl0 = qmat * basis(bs, boundary_knots[1])
    tl1 = qmat * basis(bs, boundary_knots[1], 1)
    tr0 = qmat * basis(bs, boundary_knots[2])
    tr1 = qmat * basis(bs, boundary_knots[2], 1)
    return NSplineBasis(bs,
                        qmat,
                        tl0, tl1, tr0, tr1)
end

function basis(ns::NSplineBasis{T}, x::T, ders::Int=0) where {T<:Real}
    if (x < ns.b_spline_basis.boundary_knots[1])
        if (ders == 0)
            vec = ns.tl0 + (x - ns.b_spline_basis.boundary_knots[1]) * ns.tl1
        elseif (ders == 1)
            vec = ns.tl1
        else
            vec = ns.tl1 .* T(0)
        end
    elseif (x > ns.b_spline_basis.boundary_knots[2])
        if (ders == 0)
            vec = ns.tr0 + (x - ns.b_spline_basis.boundary_knots[2]) * ns.tr1
        elseif (ders == 1)
            vsc = ns.tr1
        else
            vec = ns.tr1 .* T(0)
        end
    else
        vec = ns.qmat * basis(ns.b_spline_basis, x, ders)
    end
    return vec
end

"""
    ns_(x :: Array{T,1}; <keyword arguments>) where T<:Real

Calculate a basis for natural B-splines and return a function with signature
`(x:: Array{T,1}; ders :: Int = 0)` for evaluation of `ders`
derivative for the splines at `x`.

The keyword arguments include one of:
1. `df`, possibly in combination with `intercept`
2. `boundary_knots` and `interior_knots`
3. `knots`

# Arguments
- `boundary_knots :: Union{Tuple{T,T},Nothing} = nothing`: boundary knots
- `interior_knots :: Union{Array{T,1},Nothing} = nothing`: interior knots
- `order :: Int = 4`: order of the spline
- `intercept :: Bool = false`: bool for whether to include an intercept
- `df :: Int = order - 1 + Int(intercept)`: degrees of freedom
- `knots :: Union{Array{T,1}, Nothing} = nothing`: full set of knots (excluding repeats)
- `centre :: Union{T,Nothing} = nothing)`: value to centre the splines

# Examples
```jldoctest
julia> Splines2.ns_(collect(0.0:0.2:1.0), df=3)(collect(0.0:0.2:1.0))
6×3 Array{Float64,2}:
  0.0       0.0        0.0     
 -0.100444  0.409332  -0.272888
  0.102383  0.540852  -0.359235
  0.501759  0.386722  -0.172481
  0.418872  0.327383   0.217745
 -0.142857  0.428571   0.714286
```
"""
function ns_(x::Array{T,1};
             boundary_knots::Union{Tuple{T,T},Nothing}=nothing,
             interior_knots::Union{Array{T,1},Nothing}=nothing,
             order::Int=4,
             intercept::Bool=false,
             df::Int=order - 3 + Int(intercept),
             knots::Union{Array{T,1},Nothing}=nothing,
             centre::Union{T,Nothing}=nothing) where {T<:Real}
    (boundary_knots, interior_knots) = spline_args(x; boundary_knots=boundary_knots,
                                                   interior_knots=interior_knots,
                                                   order=order, intercept=intercept, df=df,
                                                   knots=knots, knots_offset=2)
    spline = NSplineBasis(boundary_knots, interior_knots, order, intercept)
    function eval(x::Array{T,1}; ders::Int=0)
        b = basis(spline, x, ders)
        if (centre != nothing && ders == 0)
            bc = basis(spline, centre, ders)
            for i in 1:size(b, 1)
                b[i, :] -= bc
            end
        end
        return b
    end
    return eval
end

"""
    ns(x :: Array{T,1}; <keyword arguments>) where T<:Real

Calculate a basis for natural B-splines. 

The keyword arguments include one of:
1. `df`, possibly in combination with `intercept`
2. `boundary_knots` and `interior_knots`
3. `knots`

# Arguments
- `boundary_knots :: Union{Tuple{T,T},Nothing} = nothing`: boundary knots
- `interior_knots :: Union{Array{T,1},Nothing} = nothing`: interior knots
- `order :: Int = 4`: order of the spline
- `intercept :: Bool = false`: bool for whether to include an intercept
- `df :: Int = order - 1 + Int(intercept)`: degrees of freedom
- `knots :: Union{Array{T,1}, Nothing} = nothing`: full set of knots (excluding repeats)
- `centre :: Union{T,Nothing} = nothing)`: value to centre the splines
- `ders :: Int = 0`: derivatives of the splines

# Examples
```jldoctest
julia> Splines2.ns(collect(0.0:0.2:1.0), df=3)
6×3 Array{Float64,2}:
  0.0       0.0        0.0     
 -0.100444  0.409332  -0.272888
  0.102383  0.540852  -0.359235
  0.501759  0.386722  -0.172481
  0.418872  0.327383   0.217745
 -0.142857  0.428571   0.714286
```
"""
function ns(x::Array{T,1}; ders::Int=0, kwargs...) where {T<:Real}
    return ns_(x; kwargs...)(x; ders=ders)
end
