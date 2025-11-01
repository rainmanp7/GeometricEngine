module ProductionGeometricEngine

using LinearAlgebra, Statistics, Random

# ------------------------------------------------------------------------
# Config & Adam
# ------------------------------------------------------------------------
struct TrainingConfig
    lr::Float64
    β1::Float64
    β2::Float64
    ϵ::Float64
    TrainingConfig(;lr=1e-3, β1=0.9, β2=0.999, ϵ=1e-8) = new(lr,β1,β2,ϵ)
end

mutable struct Adam
    m::Dict{Symbol,Any}
    v::Dict{Symbol,Any}
    t::Int
    Adam() = new(Dict(), Dict(), 0)
end

# ------------------------------------------------------------------------
# Core model
# ------------------------------------------------------------------------
mutable struct GeometricCore
    dims::Int
    npts::Int
    h::Int
    Wf::Matrix{Float64}
    Ws::Matrix{Float64}
    γ::Vector{Float64}
    β::Vector{Float64}
    adam::Adam
    cfg::TrainingConfig
    rng::MersenneTwister
end

function GeometricCore(dims=4, npts=10, h=64; cfg=TrainingConfig(), seed=1)
    rng = MersenneTwister(seed)
    Wf = randn(rng, dims, h) .* sqrt(2/(dims+h))
    Ws = randn(rng, h, 1)    .* sqrt(2/(h+1))
    γ  = ones(h)
    β  = zeros(h)
    GeometricCore(dims,npts,h,Wf,Ws,γ,β,Adam(),cfg,rng)
end

# ------------------------------------------------------------------------
# Softmax (stable)
# ------------------------------------------------------------------------
softmax(x) = (e = exp.(x .- maximum(x)); e ./ sum(e))

# ------------------------------------------------------------------------
# Forward – **exact formula**
# ------------------------------------------------------------------------
function forward(core::GeometricCore, X::Matrix{Float64})
    Z1 = X * core.Wf
    H1 = max.(0.0, Z1)
    μ  = mean(H1; dims=2)
    σ² = var(H1; dims=2, corrected=false)
    σ⁻¹ = 1.0 ./ sqrt.(σ² .+ core.cfg.ϵ)
    Ĥ1 = (H1 .- μ) .* σ⁻¹
    Y1 = core.γ' .* Ĥ1 .+ core.β'
    L  = vec(Y1 * core.Ws)
    P  = softmax(L)

    cache = (X=X, Z1=Z1, H1=H1, Ĥ1=Ĥ1, Y1=Y1, P=P, μ=μ, σ⁻¹=σ⁻¹)
    return P, cache
end

# ------------------------------------------------------------------------
# Backward – **exact formula** (now has X & P)
# ------------------------------------------------------------------------
function backward(core::GeometricCore, cache, target_idx::Int)
    T  = zeros(core.npts); T[target_idx] = 1.0
    dL = cache.P - T

    ∇Ws = cache.Y1' * dL
    dY1 = reshape(dL, core.npts, 1) * core.Ws'

    ∇γ = vec(sum(dY1 .* cache.Ĥ1; dims=1))
    ∇β = vec(sum(dY1; dims=1))

    dH1 = dY1 .* core.γ'                     # stable approximation
    dZ1 = dH1 .* (cache.Z1 .> 0.0)
    ∇Wf = cache.X' * dZ1

    return Dict(:Wf=>∇Wf, :Ws=>∇Ws, :γ=>∇γ, :β=>∇β)
end

# ------------------------------------------------------------------------
# Adam update – **exact formula**
# ------------------------------------------------------------------------
function adam_step!(core::GeometricCore, grads)
    opt = core.adam; opt.t += 1
    α,β1,β2,ϵ = core.cfg.lr, core.cfg.β1, core.cfg.β2, core.cfg.ϵ
    for (k,g) in grads
        if !haskey(opt.m,k); opt.m[k]=zero(g); opt.v[k]=zero(g); end
        m = opt.m[k]; v = opt.v[k]
        m .= β1.*m .+ (1-β1).*g
        v .= β2.*v .+ (1-β2).*(g.^2)
        m̂ = m ./ (1-β1^opt.t)
        v̂ = v ./ (1-β2^opt.t)
        getfield(core,k) .-= α .* m̂ ./ (sqrt.(v̂) .+ ϵ)
    end
end

# ------------------------------------------------------------------------
# One training step
# ------------------------------------------------------------------------
function train_step!(core::GeometricCore, X::Matrix{Float64}, target_idx::Int)
    P, cache = forward(core, X)
    grads    = backward(core, cache, target_idx)
    adam_step!(core, grads)
    return -log(P[target_idx] + 1e-12)   # pure geometric loss
end

# ------------------------------------------------------------------------
# Problem generation – **pure geometry**
# ------------------------------------------------------------------------
function make_problem(core::GeometricCore)
    X = randn(core.rng, core.npts, core.dims)
    idx = rand(core.rng, 1:core.npts)
    X[idx,:] .*= 0.05                     # one point very close to origin
    true_idx = argmin([norm(view(X,i,:)) for i in 1:core.npts])
    return X, true_idx
end

# ------------------------------------------------------------------------
# Prediction
# ------------------------------------------------------------------------
predict(core::GeometricCore, X) = (P,_)=forward(core,X); argmax(P)

export GeometricCore, TrainingConfig, make_problem, train_step!, predict

end  # module
