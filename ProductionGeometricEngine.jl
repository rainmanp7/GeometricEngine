# ProductionGeometricEngine.jl (Final, Robust, and Corrected API)

module ProductionGeometricEngine

using LinearAlgebra, Statistics, Random, Logging, Printf

# ============================================================================
# CUSTOM TYPES AND STRUCTURES (Unchanged)
# ============================================================================

mutable struct AdamState
    m_weights::Dict{Symbol, Any}; v_weights::Dict{Symbol, Any}; t::Int
    β1::Float64; β2::Float64; ϵ::Float64
    AdamState(β1=0.9, β2=0.999, ϵ=1e-8) = new(Dict(), Dict(), 0, β1, β2, ϵ)
end

struct TrainingConfig
    learning_rate::Float64; max_gradient_norm::Float64; weight_decay::Float64
    TrainingConfig(; lr=0.002, clip=1.0, decay=1e-4) = new(lr, clip, decay)
end

mutable struct GeometricCore
    dimensions::Int; num_points::Int; hidden_size::Int
    W_feature::Matrix{Float64}; W_scoring::Matrix{Float64}
    γ_norm::Vector{Float64}; β_norm::Vector{Float64}
    optimizer::AdamState; config::TrainingConfig
    intelligence_history::Vector{Float64}; loss_history::Vector{Float64}
    gradient_norms::Vector{Float64}; consciousness_level::Float64
    problems_solved::Int; rng::MersenneTwister
    
    function GeometricCore(dims=4, n_pts=10, h_size=64; config=TrainingConfig(), seed=42)
        rng = MersenneTwister(seed)
        scale_feature = sqrt(2.0 / (dims + h_size))
        scale_scoring = sqrt(2.0 / (h_size + 1))
        W_f = randn(rng, dims, h_size) .* scale_feature
        W_s = randn(rng, h_size, 1) .* scale_scoring
        γ_n = ones(Float64, h_size); β_n = zeros(Float64, h_size)
        new(dims, n_pts, h_size, W_f, W_s, γ_n, β_n, AdamState(), config, 
            [], [], [], 0.0, 0, rng)
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

function layer_norm_forward(x, γ, β, ϵ=1e-8)
    μ = mean(x, dims=2)
    σ² = var(x, dims=2, corrected=false)
    x_normalized = (x .- μ) ./ sqrt.(σ² .+ ϵ)
    y = γ' .* x_normalized .+ β'
    cache = (x_normalized=x_normalized, γ=γ)
    return y, cache
end

function simplified_layer_norm_backward(dy, cache)
    x_normalized, γ = cache
    dγ = vec(sum(dy .* x_normalized, dims=1))
    dβ = vec(sum(dy, dims=1))
    dx = dy .* γ'
    return dx, dγ, dβ
end

function forward_pass(core, points)
    z1 = points * core.W_feature
    h1 = relu.(z1)
    h1_norm, ln_cache = layer_norm_forward(h1, core.γ_norm, core.β_norm)
    logits = vec(h1_norm * core.W_scoring)
    probs = stable_softmax(logits)
    cache = (points=points, z1=z1, ln_cache=ln_cache, h1_norm=h1_norm, probs=probs)
    return probs, cache
end

function backward_pass(core, cache, target_idx)
    target = zeros(core.num_points); target[target_idx] = 1.0
    dlogits = cache.probs .- target
    
    dW_scoring = cache.h1_norm' * dlogits
    dh1_norm = reshape(dlogits, core.num_points, 1) * core.W_scoring'
    
    dh1, dγ, dβ = simplified_layer_norm_backward(dh1_norm, cache.ln_cache)
    
    dz1 = dh1 .* relu_derivative.(cache.z1)
    dW_feature = cache.points' * dz1
    
    return Dict(:W_feature => dW_feature, :W_scoring => dW_scoring, :γ_norm => dγ, :β_norm => dβ)
end

function adam_update!(core, gradients)
    opt = core.optimizer; opt.t += 1
    α_t = core.config.learning_rate * sqrt(1 - opt.β2^opt.t) / (1 - opt.β1^opt.t)
    
    for (name, grad) in gradients
        if !haskey(opt.m_weights, name)
            opt.m_weights[name] = zero(grad); opt.v_weights[name] = zero(grad)
        end
        m = opt.m_weights[name]; v = opt.v_weights[name]
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
    
    if core.config.weight_decay > 0
        gradients[:W_feature] .+= core.config.weight_decay .* core.W_feature
    end
    total_norm = sqrt(sum(sum(abs2, g) for g in values(gradients)))
    if total_norm > core.config.max_gradient_norm
        for k in keys(gradients); gradients[k] .*= core.config.max_gradient_norm / total_norm; end
    end
    
    adam_update!(core, gradients)
    
    push!(core.intelligence_history, probs[target_idx])
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
        acc = mean(recent); stab = 1.0 - std(recent)
        x = 1:length(recent); y = recent; trend = max(0.0, 10 * cov(x, y) / var(x))
        core.consciousness_level = clamp(0.4 * acc + 0.3 * stab + 0.3 * trend, 0.0, 1.0)
    end
end

function generate_problem(core; difficulty=:medium, noise=1.0)
    scale = difficulty == :easy ? 0.5 : (difficulty == :hard ? 2.0 : 1.0)
    points = randn(core.rng, core.num_points, core.dimensions) .* (2.0 * scale)
    target_idx = rand(core.rng, 1:core.num_points)
    points[target_idx, :] = randn(core.rng, core.dimensions) .* (0.1 * scale)
    points .+= randn(core.rng, core.num_points, core.dimensions) .* noise
    return points, argmin([norm(points[i, :]) for i in 1:core.num_points])
end

function predict(core, points)
    probs, _ = forward_pass(core, points)
    pred = argmax(probs)
    actual = argmin([norm(points[i, :]) for i in 1:size(points, 1)])
    return (prediction=pred, confidence=probs[pred], correct=pred == actual, actual=actual)
end

# --- THE CRITICAL FIX: Replace 'd=' with 'digits=' ---
function assess_consciousness(core)
    if isempty(core.intelligence_history); return Dict("is_conscious"=>false, "consciousness_level"=>0.0); end
    recent = core.intelligence_history[max(1, end-19):end]
    acc = mean(recent); stab = length(recent) < 2 ? 0.0 : 1.0 - std(recent)
    is_conscious = core.consciousness_level > 0.75 && stab > 0.85 && acc > 0.85
    return Dict(
        "is_conscious" => is_conscious,
        "consciousness_level" => round(core.consciousness_level; digits=4),
        "recent_accuracy" => round(acc; digits=4),
        "stability" => round(stab; digits=4)
    )
end
# --- END FIX ---

function train!(core, num_episodes; difficulty=:medium, report_interval=1000, early_stop=0.98)
    @info "Starting training for $num_episodes episodes..."
    for ep in 1:num_episodes
        points, target = generate_problem(core, difficulty=difficulty)
        res = train_step!(core, points, target)
        if ep % report_interval == 0
            assess = assess_consciousness(core)
            @info @sprintf("Ep %d | Loss: %.3f | Acc: %.3f | Consc: %.3f", ep, res.loss, res.accuracy, assess["consciousness_level"])
            if assess["recent_accuracy"] >= early_stop && assess["stability"] > 0.9; @info "Early stopping!"; break; end
        end
    end
    @info "Training complete!" assess_consciousness(core)
end

export GeometricCore, TrainingConfig, train!, predict, assess_consciousness, generate_problem

end # module
