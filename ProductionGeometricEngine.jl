module ProductionGeometricEngine

using LinearAlgebra, Statistics, Random, Logging, Printf

# ============================================================================
# CUSTOM TYPES AND STRUCTURES
# ============================================================================

mutable struct AdamState
    m_weights::Dict{Symbol, Any}
    v_weights::Dict{Symbol, Any}
    t::Int
    β1::Float64
    β2::Float64
    ϵ::Float64
    AdamState(β1=0.9, β2=0.999, ϵ=1e-8) = new(Dict(), Dict(), 0, β1, β2, ϵ)
end

struct TrainingConfig
    learning_rate::Float64
    max_gradient_norm::Float64
    weight_decay::Float64
    dropout_rate::Float64  # Advanced: Added dropout for regularization
    TrainingConfig(; lr=0.002, clip=1.0, decay=1e-4, dropout=0.1) = new(lr, clip, decay, dropout)
end

mutable struct GeometricCore
    dimensions::Int
    num_points::Int
    hidden_size::Int
    W_feature::Matrix{Float64}
    W_scoring::Matrix{Float64}
    γ_norm::Vector{Float64}
    β_norm::Vector{Float64}
    optimizer::AdamState
    config::TrainingConfig
    intelligence_history::Vector{Float64}
    loss_history::Vector{Float64}
    gradient_norms::Vector{Float64}
    consciousness_level::Float64
    problems_solved::Int
    rng::MersenneTwister
    
    function GeometricCore(dims=4, n_pts=10, h_size=64; config=TrainingConfig(), seed=42)
        rng = MersenneTwister(seed)
        scale_feature = sqrt(2.0 / (dims + h_size))  # He initialization
        scale_scoring = sqrt(2.0 / (h_size + 1))
        W_f = randn(rng, dims, h_size) .* scale_feature
        W_s = randn(rng, h_size, 1) .* scale_scoring
        γ_n = ones(Float64, h_size)
        β_n = zeros(Float64, h_size)
        new(dims, n_pts, h_size, W_f, W_s, γ_n, β_n, AdamState(), config, 
            Float64[], Float64[], Float64[], 0.0, 0, rng)
    end
end

# ============================================================================
# CORE LOGIC (Forward, Backward, Update)
# ============================================================================

@inline relu(x) = max(0.0, x)
@inline relu_derivative(x) = x > 0.0 ? 1.0 : 0.0

function stable_softmax(x)
    x_max = maximum(x)
    exp_x = exp.(x .- x_max)
    return exp_x ./ sum(exp_x)
end

function dropout(x, rate, rng)
    if rate == 0.0
        return x
    end
    mask = rand(rng, size(x)...) .> rate
    return x .* mask ./ (1 - rate)
end

function layer_norm_forward(x, γ, β, ϵ=1e-8)
    μ = mean(x, dims=2)
    xmu = x .- μ
    var_x = mean(xmu .^ 2, dims=2)
    invstd = 1.0 ./ sqrt.(var_x .+ ϵ)
    xhat = xmu .* invstd
    y = γ' .* xhat .+ β'
    cache = (xhat=xhat, xmu=xmu, invstd=invstd, γ=γ)
    return y, cache
end

function layer_norm_backward(dy, cache)
    xhat = cache.xhat
    xmu = cache.xmu
    invstd = cache.invstd
    γ = cache.γ
    N = size(xhat, 2)  # hidden_size

    dγ = vec(sum(dy .* xhat, dims=1))
    dβ = vec(sum(dy, dims=1))

    dxhat = dy .* γ'
    dvar = sum(dxhat .* xmu, dims=2) .* (-0.5 .* invstd .^ 3)
    dxmu = dxhat .* invstd .+ (dvar .* (2.0 .* xmu / N))
    dmu = -sum(dxmu, dims=2)
    dx = dxmu .+ (dmu / N)

    return dx, dγ, dβ
end

function forward_pass(core, points)
    z1 = points * core.W_feature
    h1 = relu.(z1)
    h1 = dropout(h1, core.config.dropout_rate, core.rng)  # Advanced: Added dropout
    h1_norm, ln_cache = layer_norm_forward(h1, core.γ_norm, core.β_norm)
    logits = vec(h1_norm * core.W_scoring)
    probs = stable_softmax(logits)
    cache = (points=points, z1=z1, ln_cache=ln_cache, h1_norm=h1_norm, probs=probs)
    return probs, cache
end

function backward_pass(core, cache, target_idx)
    target = zeros(core.num_points)
    target[target_idx] = 1.0
    dlogits = cache.probs .- target
    
    dW_scoring = cache.h1_norm' * dlogits
    dh1_norm = reshape(dlogits, core.num_points, 1) * core.W_scoring'
    
    dh1, dγ, dβ = layer_norm_backward(dh1_norm, cache.ln_cache)
    
    dz1 = dh1 .* relu_derivative.(cache.z1)
    dW_feature = cache.points' * dz1
    
    return Dict(:W_feature => dW_feature, :W_scoring => dW_scoring, :γ_norm => dγ, :β_norm => dβ)
end

function adam_update!(core, gradients)
    opt = core.optimizer
    opt.t += 1
    α_t = core.config.learning_rate * sqrt(1 - opt.β2^opt.t) / (1 - opt.β1^opt.t)
    
    for (name, grad) in gradients
        if !haskey(opt.m_weights, name)
            opt.m_weights[name] = zero(grad)
            opt.v_weights[name] = zero(grad)
        end
        m = opt.m_weights[name]
        v = opt.v_weights[name]
        m .= opt.β1 .* m .+ (1 - opt.β1) .* grad
        v .= opt.β2 .* v .+ (1 - opt.β2) .* (grad .^ 2)
        param_ref = getfield(core, name)
        param_ref .-= α_t .* m ./ (sqrt.(v) .+ opt.ϵ)
    end
end

function train_step!(core, points, target_idx)
    probs, cache = forward_pass(core, points)
    loss = -log(probs[target_idx] + 1e-10)
    
    gradients = backward_pass(core, cache, target_idx)
    
    # Apply weight decay regularization to all weights (Advanced: Extended to all params)
    if core.config.weight_decay > 0
        gradients[:W_feature] .+= core.config.weight_decay .* core.W_feature
        gradients[:W_scoring] .+= core.config.weight_decay .* core.W_scoring
    end
    
    # Gradient clipping
    total_norm = sqrt(sum(sum(abs2, g) for g in values(gradients)))
    push!(core.gradient_norms, total_norm)  # Advanced: Track gradient norms for analysis
    if total_norm > core.config.max_gradient_norm
        for k in keys(gradients)
            gradients[k] .*= core.config.max_gradient_norm / total_norm
        end
    end
    
    adam_update!(core, gradients)
    
    push!(core.intelligence_history, probs[target_idx])
    push!(core.loss_history, loss)  # Advanced: Track loss history
    core.problems_solved += 1
    update_consciousness!(core)
    
    return (loss=loss, accuracy=probs[target_idx])
end

# ============================================================================
# HIGH-LEVEL API
# ============================================================================

function update_consciousness!(core)
    if length(core.intelligence_history) >= 20
        recent = core.intelligence_history[end-19:end]
        acc = mean(recent)
        stab = 1.0 - std(recent)
        x = 1:length(recent)
        y = recent
        # Compute trend (positive slope indicates improvement)
        trend = max(0.0, 10 * cov(x, y) / var(x))
        core.consciousness_level = clamp(0.4 * acc + 0.3 * stab + 0.3 * trend, 0.0, 1.0)
    end
end

function generate_problem(core; difficulty=:medium, noise=1.0)
    scale = difficulty == :easy ? 0.5 : (difficulty == :hard ? 2.0 : 1.0)
    points = randn(core.rng, core.num_points, core.dimensions) .* (2.0 * scale)
    target_idx = rand(core.rng, 1:core.num_points)
    # Make target point closer to origin
    points[target_idx, :] = randn(core.rng, core.dimensions) .* (0.1 * scale)
    # Add noise
    points .+= randn(core.rng, core.num_points, core.dimensions) .* (0.01 * noise)
    return points, argmin([norm(points[i, :]) for i in 1:core.num_points])
end

function predict(core, points)
    probs, _ = forward_pass(core, points)
    pred = argmax(probs)
    actual = argmin([norm(points[i, :]) for i in 1:size(points, 1)])
    return (prediction=pred, confidence=probs[pred], correct=pred == actual, actual=actual)
end

function assess_consciousness(core)
    if isempty(core.intelligence_history)
        return Dict(
            "is_conscious" => false,
            "consciousness_level" => 0.0,
            "recent_accuracy" => 0.0,
            "stability" => 0.0
        )
    end
    
    recent = core.intelligence_history[max(1, end-19):end]
    acc = mean(recent)
    stab = length(recent) < 2 ? 0.0 : 1.0 - std(recent)
    is_conscious = core.consciousness_level > 0.75 && stab > 0.85 && acc > 0.85
    
    return Dict(
        "is_conscious" => is_conscious,
        "consciousness_level" => round(core.consciousness_level; digits=4),
        "recent_accuracy" => round(acc; digits=4),
        "stability" => round(stab; digits=4)
    )
end

function train!(core, num_episodes; difficulty=:medium, report_interval=1000, early_stop=0.98)
    @info "Starting training for $num_episodes episodes (difficulty: $difficulty)..."
    
    for ep in 1:num_episodes
        points, target = generate_problem(core, difficulty=difficulty)
        res = train_step!(core, points, target)
        
        if ep % report_interval == 0
            assess = assess_consciousness(core)
            @info @sprintf("Ep %d | Loss: %.3f | Acc: %.3f | Consc: %.3f", 
                          ep, res.loss, res.accuracy, assess["consciousness_level"])
            
            # Early stopping check
            if assess["recent_accuracy"] >= early_stop && assess["stability"] > 0.9
                @info "Early stopping criterion met at episode $ep!"
                break
            end
        end
    end
    
    final_assess = assess_consciousness(core)
    @info "Training complete!" final_assess
end

# Export all public API functions
export GeometricCore, TrainingConfig, AdamState
export train!, predict, assess_consciousness, generate_problem
export update_consciousness!

end # module
