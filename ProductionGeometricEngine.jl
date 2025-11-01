# GeometricEngine.jl (The Final, Corrected Version)

module GeometricEngine

using LinearAlgebra, Statistics, Random

struct TrainingConfig
    learning_rate::Float64
    β1::Float64
    β2::Float64
    ϵ::Float64
    TrainingConfig(; lr=0.001, β1=0.9, β2=0.999, ϵ=1e-8) = new(lr, β1, β2, ϵ)
end

mutable struct AdamOptimizer
    m::Dict{Symbol, Any}
    v::Dict{Symbol, Any}
    t::Int
    AdamOptimizer() = new(Dict(), Dict(), 0)
end

mutable struct GeometricCore
    dimensions::Int
    num_points::Int
    hidden_size::Int
    W_f::Matrix{Float64}
    W_s::Matrix{Float64}
    γ::Vector{Float64}
    β::Vector{Float64}
    optimizer::AdamOptimizer
    config::TrainingConfig
    rng::MersenneTwister
    
    function GeometricCore(dims=4, n_pts=10, h_size=64; config=TrainingConfig(), seed=42)
        rng = MersenneTwister(seed)
        W_f = randn(rng, dims, h_size) * sqrt(2 / (dims + h_size))
        W_s = randn(rng, h_size, 1) * sqrt(2 / (h_size + 1))
        γ = ones(h_size)
        β = zeros(h_size)
        new(dims, n_pts, h_size, W_f, W_s, γ, β, AdamOptimizer(), config, rng)
    end
end

function stable_softmax(x::Vector{Float64})
    x_max = maximum(x)
    exp_x = exp.(x .- x_max)
    exp_x ./ sum(exp_x)
end

function forward_pass(core::GeometricCore, X::Matrix{Float64})
    Z1 = X * core.W_f
    H1 = max.(0.0, Z1)
    μ = mean(H1, dims=2)
    σ² = var(H1, dims=2, corrected=false)
    σ_inv = 1.0 ./ sqrt.(σ² .+ core.config.ϵ)
    Ĥ1 = (H1 .- μ) .* σ_inv
    Y1 = (core.γ' .* Ĥ1) .+ core.β'
    L = vec(Y1 * core.W_s)
    P = stable_softmax(L)
    
    # FIX 1: Add the original input `X` to the cache so the backward pass can see it.
    cache = (X=X, Z1=Z1, Ĥ1=Ĥ1, Y1=Y1, P=P)
    return P, cache
end

function backward_pass(core::GeometricCore, cache, T::Vector{Float64})
    dL = cache.P - T
    ∇W_s = cache.Y1' * dL
    dY1 = reshape(dL, core.num_points, 1) * core.W_s'
    ∇γ = vec(sum(dY1 .* cache.Ĥ1, dims=1))
    ∇β = vec(sum(dY1, dims=1))
    dH1 = dY1 .* core.γ'  # Stable gradient approximation
    dZ1 = dH1 .* (cache.Z1 .> 0.0)
    
    # FIX 2: Use the original input `X` from the cache to calculate the gradient for W_f.
    ∇W_f = cache.X' * dZ1
    
    return Dict(:W_f => ∇W_f, :W_s => ∇W_s, :γ => ∇γ, :β => ∇β)
end

function adam_update!(core::GeometricCore, gradients::Dict)
    opt = core.optimizer
    opt.t += 1
    α = core.config.learning_rate
    β1 = core.config.β1
    β2 = core.config.β2
    ϵ = core.config.ϵ
    for (name, ∇) in gradients
        if !haskey(opt.m, name)
            opt.m[name] = zeros(size(∇))
            opt.v[name] = zeros(size(∇))
        end
        m = opt.m[name]
        v = opt.v[name]
        m .= β1 .* m .+ (1 - β1) .* ∇
        v .= β2 .* v .+ (1 - β2) .* (∇ .^ 2)
        m_hat = m ./ (1 - β1^opt.t)
        v_hat = v ./ (1 - β2^opt.t)
        param = getfield(core, name)
        param .-= α .* m_hat ./ (sqrt.(v_hat) .+ ϵ)
    end
end

function train_step!(core::GeometricCore, X::Matrix{Float64}, target_idx::Int)
    P, cache = forward_pass(core, X)
    T = zeros(core.num_points)
    T[target_idx] = 1.0
    # The cache now correctly contains X, so this will work.
    gradients = backward_pass(core, cache, T) 
    adam_update!(core, gradients)
end

function generate_problem(core::GeometricCore)
    X = randn(core.rng, core.num_points, core.dimensions)
    # The geometric task: find the point closest to the origin.
    target_idx = argmin(vec(sum(X.^2, dims=2)))
    return X, target_idx
end

function predict(core::GeometricCore, X::Matrix{Float64})
    P, _ = forward_pass(core, X)
    argmax(P)
end

export GeometricCore, TrainingConfig, train_step!, generate_problem, predict

end
