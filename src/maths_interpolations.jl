# Most credit goes to the package DataInterpolations.jl: https://github.com/PumasAI/DataInterpolations.jl
# LICENSE as of 2022_06_23: https://github.com/PumasAI/DataInterpolations.jl/blob/master/LICENSE.md

struct CubicSpline{uType,tType,hType,zType,FT,T}
    t::tType
    u::uType
    h::hType
    z::zType
    CubicSpline{FT}(t,u,h,z) where FT = new{typeof(t),typeof(u),typeof(h),typeof(z),FT,eltype(u)}(t,u,h,z)
end

(A::CubicSpline)(t) = _interpolate(A, t)
  
function CubicSpline(t, u::uType) where {uType<:AbstractVector{<:Number}}
    n = length(t) - 1
    h = vcat(0, map(k -> t[k+1] - t[k], 1:length(t)-1), 0)
    dl = h[2:n+1]
    d_tmp = 2 .* (h[1:n+1] .+ h[2:n+2])
    du = h[2:n+1]
    tA = Tridiagonal(dl,d_tmp,du)
    d = map(i -> i == 1 || i == n + 1 ? 0 : 6(u[i+1] - u[i]) / h[i+1] - 6(u[i] - u[i-1]) / h[i], 1:n+1)
    z = tA\d
    CubicSpline{true}(t,u,h[1:n+1],z)
end

function cspline_interp(x, y, xnew)
    good = findall(isfinite.(x) .&& isfinite.(y))
    A = CubicSpline(x[good], y[good])
    good = findall(xnew .>= x[good[1]] .&& xnew .<= x[good[end]])
    ynew = fill(NaN, length(xnew))
    for i ∈ eachindex(good)
        ynew[good[i]] = _interpolate(A, xnew[good[i]])
    end
    return ynew
end

# CubicSpline Interpolation
function _interpolate(A::CubicSpline{<:AbstractVector}, t::Number)
    i = max(1, min(searchsortedlast(A.t, t), length(A.t) - 1))
    I = A.z[i] * (A.t[i+1] - t)^3 / (6A.h[i+1]) + A.z[i+1] * (t - A.t[i])^3 / (6A.h[i+1])
    C = (A.u[i+1]/A.h[i+1] - A.z[i+1]*A.h[i+1]/6)*(t - A.t[i])
    D = (A.u[i]/A.h[i+1] - A.z[i]*A.h[i+1]/6)*(A.t[i+1] - t)
    I + C + D
end


struct LinearInterpolation{uType,tType,FT,T}
    t::tType
    u::uType
    LinearInterpolation{FT}(t, u) where FT = new{typeof(t), typeof(u),FT,eltype(u)}(t,u)
end

(A::LinearInterpolation)(t) = _interpolate(A, t)
  
function LinearInterpolation(t, u)
    LinearInterpolation{true}(t, u)
end

function _interpolate(A::LinearInterpolation{<:AbstractVector}, t::Number)
    if isnan(t)
        # For correct derivative with NaN
        t1 = t2 = one(eltype(A.t))
        u1 = u2 = one(eltype(A.u))
    else
        idx = max(1, min(searchsortedlast(A.t, t), length(A.t) - 1))
        t1, t2 = A.t[idx], A.t[idx+1]
        u1, u2 = A.u[idx], A.u[idx+1]
    end
    θ = (t - t1)/(t2 - t1)
    val = (1 - θ)*u1 + θ*u2
    # Note: The following is limited to when val is NaN as to not change the derivative of exact points.
    t == t1 && isnan(val) && return oftype(val, u1) # Return exact value if no interpolation needed (eg when NaN at t2)
    t == t2 && isnan(val) && return oftype(val, u2) # ... (eg when NaN at t1)
    val
end



function derivative(A::LinearInterpolation{<:AbstractVector}, t::Number)
    idx = searchsortedfirst(A.t, t)
    if A.t[idx] >= t
      idx -= 1
    end
    idx == 0 ? idx += 1 : nothing
    θ = 1 / (A.t[idx+1] - A.t[idx])
    (A.u[idx+1] - A.u[idx]) / (A.t[idx+1] - A.t[idx])
end


# CubicSpline Interpolation
function derivative(A::CubicSpline{<:AbstractVector}, t::Number)
    i = searchsortedfirst(A.t, t)
    isnothing(i) ? i = length(A.t) - 1 : i -= 1
    i == 0 ? i += 1 : nothing
    dI = -3A.z[i] * (A.t[i + 1] - t)^2 / (6A.h[i + 1]) + 3A.z[i + 1] * (t - A.t[i])^2 / (6A.h[i + 1])
    dC = A.u[i + 1] / A.h[i + 1] - A.z[i + 1] * A.h[i + 1] / 6
    dD = -(A.u[i] / A.h[i + 1] - A.z[i] * A.h[i + 1] / 6)
    dI + dC + dD
end

function lin_interp(x, y, xnew)
    good = findall(isfinite.(x) .&& isfinite.(y))
    A = LinearInterpolation(float.(x[good]), float.(y[good]))
    good = findall(xnew .>= x[good[1]] .&& xnew .<= x[good[end]])
    ynew = fill(NaN, length(xnew))
    for i ∈ eachindex(good)
        ynew[good[i]] = _interpolate(A, xnew[good[i]])
    end
    return ynew
end