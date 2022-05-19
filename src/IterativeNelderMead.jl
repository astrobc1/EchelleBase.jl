module IterativeNelderMead

using Statistics, LinearAlgebra
using DataStructures
using Infiltrator

using EchelleBase

export IterativeNelderMeadOptimizer, optimize

struct IterativeNelderMeadOptimizer
end

# Pn = (P - Pl) / Δ
# P = Pn * Δ + Pl
function normalize_parameters(pars)
    parsn = Parameters()
    for par ∈ values(pars)
        parsn[par.name] = normalize_parameter(par)
    end
    return parsn
end

function normalize_parameter(par)
    if isfinite(par.lower_bound) && isfinite(par.upper_bound) && par.lower_bound != par.upper_bound
        r = par.upper_bound - par.lower_bound
        return Parameter(name=par.name, value=(par.value - par.lower_bound) / r, lower_bound=0.0, upper_bound=1.0)
    else
        return Parameter(name=par.name, value=par.value, lower_bound=par.lower_bound, upper_bound=par.upper_bound)
    end
end


function denormalize_parameters(parsn, pars)
    pars_out = Parameters()
    for (parn, par) ∈ zip(values(parsn), values(pars))
        pars_out[par.name] = denormalize_parameter(parn, par)
    end
    return pars_out
end

function denormalize_parameter(parn, par)
    if isfinite(par.lower_bound) && isfinite(par.upper_bound)
        r = par.upper_bound - par.lower_bound
        return Parameter(name=par.name, value=parn.value * r + par.lower_bound, lower_bound=par.lower_bound, upper_bound=par.upper_bound)
    else
        return Parameter(name=par.name, value=parn.value, lower_bound=par.lower_bound, upper_bound=par.upper_bound)
    end
end

function optimize(optimizer::IterativeNelderMeadOptimizer, p0, obj; ftol_rel=1E-6)

    # Varied parameters
    p0v = Parameters()
    for par ∈ values(p0)
        if !(par.lower_bound == par.upper_bound == par.value)
            p0v[par.name] = par
        end
    end

    if length(p0v) == 0
        fbest = obj(p0)
        return (;pbest=p0, fbest=fbest, fcalls=0)
    end

    # Number of iterations
    n_iterations = length(p0v)

    # Max f evals
    max_f_evals = 1200 * length(p0v)

    # Subspaces
    subspaces = []
    pnames = collect(keys(p0))
    pnamesv = collect(keys(p0v))
    vi = [i for (i, par) ∈ enumerate(values(p0)) if par.name ∈ pnamesv]
    full_subspace = (;names=pnamesv, index=nothing, indices=vi, indicesv=[1:length(p0v);])
    if length(p0v) > 2
        for i=1:length(p0v)-1
            k1 = vi[i]
            k2 = vi[i+1]
            push!(subspaces, (;names=[pnamesv[i], pnamesv[i+1]], index=i, indicesv=[i, i+1], indices=[k1, k2]))
        end
        k1 = findfirst(pnames .== pnamesv[1])
        k2 = findfirst(pnames .== pnamesv[end])
        push!(subspaces, (;names=[pnamesv[1], pnamesv[end]], index=length(p0v), indicesv=[1, length(p0v)], indices=[k1, k2]))
        k1 = findfirst(pnames .== pnamesv[2])
        k2 = findfirst(pnames .== pnamesv[end-1])
        push!(subspaces, (;names=[pnamesv[2], pnamesv[end-1]], index=length(p0v), indicesv=[2, length(p0v)-1], indices=[k1, k2]))
    end

    # Rescale parameters
    p0n = normalize_parameters(p0)
    p0vn = normalize_parameters(p0v)

    # Initial solution
    pbest = Ref(deepcopy(p0))
    fbest = Ref(float(obj(p0)))

    # Fcalls
    fcalls = Ref(0)

    # Full simplex
    x0vn = [par.value for par ∈ values(p0vn)]
    current_full_simplex = repeat(x0vn, 1, length(x0vn)+1)
    current_full_simplex[:, 1:end-1] .+= diagm(0.5 .* x0vn)
    
    # Loop over iterations
    for iteration=1:n_iterations

        # Perform Ameoba call for all parameters
        optimize_space!(full_subspace, p0, p0v, p0vn, pbest, fbest, fcalls, max_f_evals, current_full_simplex, current_full_simplex, obj, ftol_rel)
        
        # If there's <= 2 params, a three-simplex is the smallest simplex used and only used once.
        if length(p0v) <= 2
            break
        end
        
        # Perform Ameoba call for subspaces
        for subspace ∈ subspaces
            initial_simplex = get_subspace_simplex(subspace, p0, pbest[])
            optimize_space!(subspace, p0, p0v, p0vn, pbest, fbest, fcalls, max_f_evals, current_full_simplex, initial_simplex, obj, ftol_rel)
        end
    end
    
    # Output
    out = (;pbest=pbest[], fbest=fbest[], fcalls=fcalls[])

    return out

end

function get_subspace_simplex(subspace, p0, pbest)
    n = length(subspace.names)
    simplex = zeros(n, n+1)
    p0n = normalize_parameters(p0)
    pbestn = normalize_parameters(pbest)
    xbestn = [par.value for par ∈ values(pbestn) if par.name ∈ subspace.names]
    x0n = [par.value for par ∈ values(p0n) if par.name ∈ subspace.names]
    simplex[:, 1] .= x0n
    simplex[:, 2] .= xbestn
    for i=3:n+1
        simplex[:, i] .= copy(xbestn)
        j = i - 2
        simplex[j, i] = x0n[j]
    end
    return simplex
end

function optimize_space!(subspace, p0::Parameters, p0v::Parameters, p0vn::Parameters, pbest::Ref{Parameters}, fbest::Ref{Float64}, fcalls::Ref{Int}, max_f_evals::Int, current_full_simplex::Matrix, initial_simplex::Matrix, obj, ftol_rel=1E-6)
    
    # Define these as they are used often
    simplex = copy(initial_simplex)
    nx, nxp1 = size(simplex)

    # Initiate storage arrays
    fvals = zeros(nxp1)
    xnp1 = zeros(nx)
    x1 = zeros(nx)
    xn = zeros(nx)
    xr = zeros(nx)
    xbar = zeros(nx)
    xc = zeros(nx)
    xe = zeros(nx)
    xcc = zeros(nx)

    # Test parameters
    pbestn = normalize_parameters(pbest[])
    ptestn = normalize_parameters(pbest[])
    
    # Generate the fvals for the initial simplex
    for i=1:nxp1
        fvals[i] = compute_obj(simplex[:, i], subspace, ptestn, p0, obj, fcalls)
    end

    # Sort the fvals and then simplex
    inds = sortperm(fvals)
    simplex .= simplex[:, inds]
    fvals .= fvals[inds]
    fmin = fvals[1]
    
    # Best fit parameter is now the first column
    xmin = copy(simplex[:, 1])
    
    # Keeps track of the number of times the solver thinks it has converged in a row.
    no_improve_break = 3
    n_converged = 0

    # Hyper parameters
    α = 1.0
    γ = 2.0
    σ = 0.5
    δ = 0.5
    
    # Loop
    while true

        # Sort the vertices according from best to worst
        # Define the worst and best vertex, and f(best vertex)
        xnp1 .= simplex[:, end]
        fnp1 = fvals[end]
        x1 .= simplex[:, 1]
        f1 = fvals[1]
        xn .= simplex[:, end-1]
        fn = fvals[end-1]
            
        # Checks whether or not to shrink if all other checks "fail"
        shrink = false

        # break after max number function calls is reached.
        if fcalls[] >= max_f_evals
            break
        end
            
        # Break if f tolerance has been met
        if compute_df_rel(fmin, fnp1) > ftol_rel
            n_converged = 0
        else
            n_converged += 1
        end
        if n_converged >= no_improve_break
            break
        end

        # Idea of NM: Given a sorted simplex; N + 1 Vectors of N parameters,
        # We want to iteratively replace the worst vector with a better vector.
        
        # The "average" vector, ignoring the worst point
        # We first anchor points off this average Vector
        xbar .= reshape(mean(simplex[:, 1:end-1], dims=2), (nx, 1))
        
        # The reflection point
        xr .= xbar .+ α .* (xbar .- xnp1)
        
        # Update the current testing parameter with xr
        fr = compute_obj(xr, subspace, ptestn, p0, obj, fcalls)

        if fr < f1
            xe .= xbar .+ γ .* (xbar .- xnp1)
            fe = compute_obj(xe, subspace, ptestn, p0, obj, fcalls)
            if fe < fr
                simplex[:, end] .= copy(xe)
                fvals[end] = fe
            else
                simplex[:, end] .= copy(xr)
                fvals[end] = fr
            end
        elseif fr < fn
            simplex[:, end] .= copy(xr)
            fvals[end] = fr
        else
            if fr < fnp1
                xc .= xbar .+ σ .* (xbar .- xnp1)
                fc = compute_obj(xc, subspace, ptestn, p0, obj, fcalls)
                if fc <= fr
                    simplex[:, end] .= copy(xc)
                    fvals[end] = fc
                else
                    shrink = true
                end
            else
                xcc .= xbar .+ σ .* (xnp1 .- xbar)
                fcc = compute_obj(xcc, subspace, ptestn, p0, obj, fcalls)
                if fcc < fvals[end]
                    simplex[:, end] .= copy(xcc)
                    fvals[end] = fcc
                else
                    shrink = true
                end
            end
        end
        if shrink
            for j=2:nxp1
                simplex[:, j] .= x1 .+ δ .* (simplex[:, j] .- x1)
                fvals[j] = compute_obj(simplex[:, j], subspace, ptestn, p0, obj, fcalls)
            end
        end

        inds = sortperm(fvals)
        fvals = fvals[inds]
        simplex .= simplex[:, inds]
        fmin = fvals[1]
        xmin .= copy(simplex[:, 1])
    end

    inds = sortperm(fvals)
    fvals = fvals[inds]
    simplex .= simplex[:, inds]
    fmin = fvals[1]
    xmin .= simplex[:, 1]
    
    # Update the full simplex and best fit parameters
    for (i, pname) ∈ enumerate(subspace.names)
        pbestn[pname].value = xmin[i]
    end
    if !isnothing(subspace.index)
        current_full_simplex[:, subspace.index] .= [par.value for par ∈ values(pbestn) if par.name ∈ keys(p0v)]
    else
        current_full_simplex .= copy(simplex)
    end

    # Denormalize and store
    pbest[] = denormalize_parameters(pbestn, p0)
    fbest[] = fmin
    nothing
end


###################
#### TOLERANCE ####
###################
    
function compute_dx_rel(simplex)
    a = nanminimum(simplex, dims=2)
    b = nanmaximum(simplex, dims=2)
    c = (abs(b) .+ abs(a)) / 2
    bad = findall(c .< 0)
    c[bad] = 1
    r = abs(b .- a) ./ c
    return nanmaximum(r)
end

function compute_df_rel(a, b)
    avg = (abs(a) + abs(b)) / 2
    return abs(a - b) / avg
end



###########################################################################
###########################################################################
###########################################################################


function penalize(f, ptest, names)
    penalty = abs(f) * 10
    for par ∈ values(ptest)
        if par.name ∈ names
            if par.value < par.lower_bound
                f += penalty * abs(par.value - par.lower_bound)
            end
            if par.value > par.upper_bound
                f += penalty * abs(par.value - par.upper_bound)
            end
        end
    end
    return f
end

function compute_obj(x::Vector{Float64}, subspace, ptestn, p0, obj, fcalls::Ref{Int})
    #@infiltrate
    fcalls[] += 1
    for i=1:length(subspace.names)
        ptestn[subspace.names[i]].value = x[i]
    end
    ptest = denormalize_parameters(ptestn, p0)
    f = obj(ptest)
    f = penalize(f, ptestn, subspace.names)
    if !isfinite(f)
        f = 1E6
    end
    return f
end

end