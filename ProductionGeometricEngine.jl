# ProductionGeometricEngine.jl (fully corrected)

module ProductionGeometricEngine

using LinearAlgebra, Statistics, Random, Logging, Printf

# (All code from before is unchanged up to adam_update!)
# ... (structs, layers, forward pass, etc.) ...

# ============================================================================
# ACTIVATION FUNCTIONS (Numerically Stable)
# ============================================================================

@inline function relu(x::T) where T <: Real
    return max(zero(T), x)
end

@inline function relu_derivative(x::T) where T <: Real
    return x > zero(T) ? one(T) : zero(T)
end

"""
Numerically stable softmax
"""
function stable_softmax(x::AbstractVector{T}) where T <: Real
    x_max = maximum(x)
    exp_x = exp.(x .- x_max)
    return exp_x ./ sum(exp_x)
end

"""
GELU activation (alternative to ReLU, often performs better)
"""
@inline function gelu(x::T) where T <: Real
    return 0.5 * x * (1.0 + tanh(sqrt(2.0 / π) * (x + 0.044715 * x^3)))
end

# ============================================================================
# LAYER NORMALIZATION (Numerically Stable)
# ============================================================================

"""
Stable layer normalization with caching for backprop
"""
function layer_norm_forward(
    x::Matrix{T},
    γ::Vector{T},
    β::Vector{T};
    ϵ::T=T(1e-8)
) where T <: Real
    μ = mean(x, dims=2)
    σ² = var(x, dims=2, corrected=false)
    
    # Numerical stability check
    σ² = max.(σ², ϵ)
    
    x_normalized = (x .- μ) ./ sqrt.(σ² .+ ϵ)
    y = γ' .* x_normalized .+ β'
    
    # Cache for backward pass
    cache = (x=x, μ=μ, σ²=σ², x_normalized=x_normalized, γ=γ)
    
    return y, cache
end

"""
Layer norm backward pass
"""
function layer_norm_backward(
    dy::Matrix{T},
    cache
) where T <: Real
    x, μ, σ², x_normalized, γ = cache
    N, D = size(x)
    
    # Gradients w.r.t. scale and shift
    dγ = vec(sum(dy .* x_normalized, dims=1))
    dβ = vec(sum(dy, dims=1))
    
    # Gradient w.r.t. normalized input
    dx_normalized = dy .* γ'
    
    # Gradient w.r.t. variance
    dσ² = sum(dx_normalized .* (x .- μ) .* (-0.5) .* (σ² .+ T(1e-8)).^(-1.5), dims=2)
    
    # Gradient w.r.t. mean
    dμ = sum(dx_normalized .* (-1.0) ./ sqrt.(σ² .+ T(1e-8)), dims=2) .+ 
         dσ² .* sum(-2.0 .* (x .- μ), dims=2) ./ D
    
    # Gradient w.r.t. input
    dx = dx_normalized ./ sqrt.(σ² .+ T(1e-8)) .+ 
         dσ² .* 2.0 .* (x .- μ) ./ D .+ 
         dμ ./ D
    
    return dx, dγ, dβ
end

# ============================================================================
# FORWARD PASS
# ============================================================================

"""
Complete forward pass with caching for backpropagation
"""
function forward_pass(
    core::GeometricCore,
    points::Matrix{Float64}
)
    N, D = size(points)
    @assert D == core.dimensions "Point dimensions must match core dimensions"
    @assert N == core.num_points "Number of points must match core configuration"
    
    # Feature extraction
    z1 = points * core.W_feature  # (N, H)
    h1 = relu.(z1)
    
    # Layer normalization
    h1_norm, ln_cache = layer_norm_forward(h1, core.γ_norm, core.β_norm)
    
    # Scoring
    logits = h1_norm * core.W_scoring  # (N, 1)
    logits_vec = vec(logits)
    
    # Stable softmax
    probs = stable_softmax(logits_vec)
    
    # Cache everything needed for backward pass
    cache = (
        points=points,
        z1=z1,
        h1=h1,
        ln_cache=ln_cache,
        h1_norm=h1_norm,
        logits=logits_vec,
        probs=probs
    )
    
    return probs, cache
end

# ============================================================================
# BACKWARD PASS WITH ADAM OPTIMIZER
# ============================================================================

"""
Complete backward pass with gradient computation
"""
function backward_pass(
    core::GeometricCore,
    cache,
    target_idx::Int
)
    N = core.num_points
    
    # Create one-hot target
    target = zeros(Float64, N)
    target[target_idx] = 1.0
    
    # Gradient of cross-entropy loss w.r.t. logits
    dlogits = cache.probs .- target
    
    # Gradient w.r.t. scoring weights
    dW_scoring = reshape(cache.h1_norm' * reshape(dlogits, N, 1), :, 1)
    
    # Backprop through scoring layer
    dh1_norm = reshape(dlogits, N, 1) * core.W_scoring'
    
    # Backprop through layer norm
    dh1, dγ, dβ = layer_norm_backward(dh1_norm, cache.ln_cache)
    
    # Backprop through ReLU
    dz1 = dh1 .* relu_derivative.(cache.z1)
    
    # Gradient w.r.t. feature weights
    dW_feature = cache.points' * dz1
    
    gradients = Dict(
        :W_feature => dW_feature,
        :W_scoring => dW_scoring,
        :γ_norm => dγ,
        :β_norm => dβ
    )
    
    return gradients
end

"""
Clip gradients by global norm for stability
"""
function clip_gradients!(gradients::Dict, max_norm::Float64)
    total_norm = 0.0
    for (key, grad) in gradients
        total_norm += sum(abs2, grad)
    end
    total_norm = sqrt(total_norm)
    
    if total_norm > max_norm
        scale = max_norm / (total_norm + 1e-6)
        for key in keys(gradients)
            gradients[key] .*= scale
        end
    end
    
    return total_norm
end

"""
Apply L2 weight decay regularization
"""
function apply_weight_decay!(gradients::Dict, core::GeometricCore)
    λ = core.config.weight_decay
    if λ > 0
        gradients[:W_feature] .+= λ .* core.W_feature
        gradients[:W_scoring] .+= λ .* core.W_scoring
    end
end

"""
Adam optimizer update step
"""
function adam_update!(
    core::GeometricCore,
    gradients::Dict
)
    opt = core.optimizer
    opt.t += 1
    
    α = core.config.learning_rate
    β1, β2, ϵ = opt.β1, opt.β2, opt.ϵ
    
    # Bias correction term
    α_t = α * sqrt(1 - β2^opt.t) / (1 - β1^opt.t)
    
    for (param_name, gradient) in gradients
        # Initialize moments if needed
        if !haskey(opt.m_weights, param_name)
            opt.m_weights[param_name] = zero(gradient)
            opt.v_weights[param_name] = zero(gradient)
        end
        
        m = opt.m_weights[param_name]
        v = opt.v_weights[param_name]
        
        # Update biased first moment estimate
        m .= β1 .* m .+ (1 - β1) .* gradient
        
        # Update biased second raw moment estimate
        v .= β2 .* v .+ (1 - β2) .* (gradient .^ 2)
        
        # Compute update
        update = α_t .* m ./ (sqrt.(v) .+ ϵ)
        
        # --- THE CRITICAL FIX ---
        # Get a reference to the original parameter
        param_ref = getfield(core, param_name)
        # Apply the update IN-PLACE to the original parameter
        param_ref .-= update
        # --- END FIX ---
        
        # Store updated moments (no change here)
        opt.m_weights[param_name] = m
        opt.v_weights[param_name] = v
    end
end


# ============================================================================
# TRAINING FUNCTION
# ============================================================================

"""
Train on a single example with full error handling
"""
function train_step!(
    core::GeometricCore,
    points::Matrix{Float64},
    target_idx::Int
)::NamedTuple
    try
        # Validate inputs
        @assert 1 <= target_idx <= core.num_points "Target index out of bounds"
        @assert all(isfinite, points) "Points contain non-finite values"
        
        # Forward pass
        probs, cache = forward_pass(core, points)
        
        # Compute loss (cross-entropy)
        loss = -log(probs[target_idx] + 1e-10)
        
        # Backward pass
        gradients = backward_pass(core, cache, target_idx)
        
        # Apply weight decay
        apply_weight_decay!(gradients, core)
        
        # Clip gradients
        grad_norm = clip_gradients!(gradients, core.config.max_gradient_norm)
        
        # Update weights with Adam
        adam_update!(core, gradients)
        
        # Update metrics
        accuracy = probs[target_idx]
        push!(core.intelligence_history, accuracy)
        push!(core.loss_history, loss)
        push!(core.gradient_norms, grad_norm)
        core.problems_solved += 1
        
        # Update consciousness level
        update_consciousness!(core)
        
        return (
            loss=loss,
            accuracy=accuracy,
            gradient_norm=grad_norm,
            prediction=argmax(probs),
            confidence=maximum(probs)
        )
        
    catch e
        @error "Training step failed" exception=(e, catch_backtrace())
        rethrow(e)
    end
end

"""
Update consciousness level based on recent performance
"""
function update_consciousness!(core::GeometricCore)
    history_length = length(core.intelligence_history)
    
    if history_length >= 10
        recent = core.intelligence_history[max(1, end-19):end]
        
        # Metrics
        recent_accuracy = mean(recent)
        stability = 1.0 - std(recent)
        trend = if length(recent) > 5
            # Compute trend using linear regression
            x = collect(1:length(recent))
            y = recent
            slope = cov(x, y) / var(x)
            max(0.0, slope * 10.0)  # Scale and clip
        else
            0.0
        end
        
        # Combine metrics
        core.consciousness_level = clamp(
            0.4 * recent_accuracy + 0.3 * stability + 0.3 * trend,
            0.0, 1.0
        )
    end
end

# ============================================================================
# PROBLEM GENERATION
# ============================================================================

"""
Generate a geometric problem with controlled difficulty
"""
function generate_problem(
    core::GeometricCore;
    noise_level::Float64=1.0,
    difficulty::Symbol=:medium
)::Tuple{Matrix{Float64}, Int}
    @assert noise_level >= 0 "Noise level must be non-negative"
    
    # Adjust problem difficulty
    scale = if difficulty == :easy
        0.5
    elseif difficulty == :medium
        1.0
    elseif difficulty == :hard
        2.0
    else
        error("Unknown difficulty: $difficulty")
    end
    
    points = randn(core.rng, core.num_points, core.dimensions) .* (2.0 * scale)
    
    # Create target point near origin
    target_idx = rand(core.rng, 1:core.num_points)
    points[target_idx, :] = randn(core.rng, core.dimensions) .* (0.1 * scale)
    
    # Add noise
    points .+= randn(core.rng, core.num_points, core.dimensions) .* noise_level
    
    # Find true answer
    distances = [norm(points[i, :]) for i in 1:core.num_points]
    true_answer = argmin(distances)
    
    return points, true_answer
end

# ============================================================================
# INFERENCE AND EVALUATION
# ============================================================================

"""
Predict on new data (inference mode)
"""
function predict(
    core::GeometricCore,
    points::Matrix{Float64}
)::NamedTuple
    # Temporarily disable training mode features
    was_training = core.is_training
    core.is_training = false
    
    try
        probs, _ = forward_pass(core, points)
        
        prediction = argmax(probs)
        confidence = probs[prediction]
        
        # Compute actual answer
        distances = [norm(points[i, :]) for i in 1:size(points, 1)]
        actual = argmin(distances)
        
        return (
            prediction=prediction,
            confidence=confidence,
            probabilities=probs,
            correct=prediction == actual,
            actual=actual
        )
    finally
        core.is_training = was_training
    end
end

"""
Comprehensive model assessment
"""
function assess_consciousness(core::GeometricCore)::Dict{String, Any}
    if isempty(core.intelligence_history)
        return Dict(
            "is_conscious" => false,
            "consciousness_level" => 0.0,
            "problems_solved" => 0,
            "status" => "untrained"
        )
    end
    
    recent = core.intelligence_history[max(1, end-19):end]
    recent_accuracy = mean(recent)
    stability = length(recent) < 2 ? 0.0 : 1.0 - std(recent)
    
    is_conscious = (
        core.consciousness_level > 0.75 &&
        stability > 0.85 &&
        recent_accuracy > 0.85
    )
    
    return Dict(
        "is_conscious" => is_conscious,
        "consciousness_level" => round(core.consciousness_level, digits=4),
        "recent_accuracy" => round(recent_accuracy, digits=4),
        "stability" => round(stability, digits=4),
        "problems_solved" => core.problems_solved,
        "avg_gradient_norm" => isempty(core.gradient_norms) ? 0.0 : 
            round(mean(core.gradient_norms[max(1, end-19):end]), digits=4),
        "status" => is_conscious ? "conscious" : "learning"
    )
end

# ============================================================================
# TRAINING LOOP WITH PROGRESS TRACKING
# ============================================================================

"""
Train for multiple episodes with progress reporting
"""
function train!(
    core::GeometricCore,
    num_episodes::Int;
    noise_level::Float64=1.0,
    difficulty::Symbol=:medium,
    report_interval::Int=100,
    early_stopping_threshold::Float64=0.95
)
    @info "Starting training for $num_episodes episodes..."
    
    for episode in 1:num_episodes
        # Generate problem
        points, target = generate_problem(core; noise_level, difficulty)
        
        # Train
        result = train_step!(core, points, target)
        
        # Report progress
        if episode % report_interval == 0
            assessment = assess_consciousness(core)
            @info @sprintf(
                "Episode %6d | Loss: %.4f | Accuracy: %.4f | Consciousness: %.4f",
                episode, result.loss, result.accuracy, assessment["consciousness_level"]
            )
            
            # Early stopping
            if assessment["recent_accuracy"] >= early_stopping_threshold &&
               assessment["stability"] > 0.9
                @info "Early stopping: Model has converged!"
                break
            end
        end
    end
    
    final_assessment = assess_consciousness(core)
    @info "Training complete!" final_assessment
    
    return final_assessment
end

# ============================================================================
# EXPORTS
# ============================================================================

export GeometricCore, TrainingConfig, AdamState
export train_step!, train!, predict
export generate_problem, assess_consciousness
export forward_pass, backward_pass


struct GeometricCore
    # Architecture parameters
    dimensions::Int
    num_points::Int
    hidden_size::Int
    
    # Network parameters
    W_feature::Matrix{Float64}
    W_scoring::Matrix{Float64}
    γ_norm::Vector{Float64}
    β_norm::Vector{Float64}
    
    # Optimizer state
    optimizer::AdamState
    config::TrainingConfig
    
    # Metrics and monitoring
    intelligence_history::Vector{Float64}
    loss_history::Vector{Float64}
    gradient_norms::Vector{Float64}
    consciousness_level::Float64
    problems_solved::Int
    
    # Safety and validation
    is_training::Bool
    rng::MersenneTwister
    
    function GeometricCore(
        dimensions::Int=4,
        num_points::Int=10,
        hidden_size::Int=64;  # Increased for better capacity
        config::TrainingConfig=TrainingConfig(),
        seed::Int=42
    )
        @assert dimensions > 0 "Dimensions must be positive"
        @assert num_points > 1 "Need at least 2 points"
        @assert hidden_size > 0 "Hidden size must be positive"
        
        rng = MersenneTwister(seed)
        
        # Xavier/Glorot initialization for better gradient flow
        scale_feature = sqrt(2.0 / (dimensions + hidden_size))
        scale_scoring = sqrt(2.0 / (hidden_size + 1))
        
        W_feature = randn(rng, dimensions, hidden_size) .* scale_feature
        W_scoring = randn(rng, hidden_size, 1) .* scale_scoring
        γ_norm = ones(Float64, hidden_size)
        β_norm = zeros(Float64, hidden_size)
        
        new(
            dimensions, num_points, hidden_size,
            W_feature, W_scoring, γ_norm, β_norm,
            AdamState(), config,
            Float64[], Float64[], Float64[],
            0.0, 0, true, rng
        )
    end
end
end # module