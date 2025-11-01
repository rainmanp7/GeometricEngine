# ProtectedGeometricEngine.jl
"""
🎯 PROTECTED GEOMETRIC INTELLIGENCE ENGINE
Core architecture demonstrating emergent mathematical consciousness.
This version correctly translates the point-wise processing architecture
from the reference Python/Keras implementation.
"""

module ProtectedGeometricEngine

using LinearAlgebra, Statistics, Random

# --- ARCHITECTURE (Unchanged) ---
mutable struct GeometricConsciousnessCore
    dimensions::Int
    num_points::Int
    hidden_size::Int
    feature_weights::Matrix{Float64}
    scoring_weights::Matrix{Float64}
    layer_norm_gamma::Vector{Float64}
    layer_norm_beta::Vector{Float64}
    intelligence_history::Vector{Float64}
    entity_count::Int
    consciousness_level::Float64
    problems_solved::Int
    learning_momentum::Float64

    function GeometricConsciousnessCore(dimensions::Int=4, num_points::Int=10)
        hidden_size = 32
        feature_weights = randn(dimensions, hidden_size) * 0.1
        scoring_weights = randn(hidden_size, 1) * 0.1
        layer_norm_gamma = ones(hidden_size)
        layer_norm_beta = zeros(hidden_size)
        new(dimensions, num_points, hidden_size,
            feature_weights, scoring_weights,
            layer_norm_gamma, layer_norm_beta,
            Float64[], 0, 0.0, 0, 0.0)
    end
end

# --- HELPERS (Unchanged) ---
relu_activation(x::Matrix{Float64}) = max.(x, 0.0)
layer_norm(x::Matrix{Float64}, g::Vector{Float64}, b::Vector{Float64}, ϵ::Float64=1e-8) = begin
    μ = mean(x, dims=2)
    σ² = var(x, dims=2, corrected=false)
    x̂ = (x .- μ) ./ sqrt.(σ² .+ ϵ)
    return g' .* x̂ .+ b'
end

# --- FORWARD PASS (Unchanged, but added ϵ to layer_norm call) ---
function geometric_forward(core::GeometricConsciousnessCore, points::Matrix{Float64})
    features = points * core.feature_weights
    activated_features = relu_activation(features)
    
    # Store intermediate values from layer_norm for backprop
    ϵ = 1e-8
    μ = mean(activated_features, dims=2)
    σ² = var(activated_features, dims=2, corrected=false)
    inv_σ = 1.0 ./ sqrt.(σ² .+ ϵ)
    features_hat = (activated_features .- μ) .* inv_σ
    
    normalized_features = core.layer_norm_gamma' .* features_hat .+ core.layer_norm_beta'
    scores = normalized_features * core.scoring_weights
    
    scores_vec = vec(scores)
    max_score = maximum(scores_vec)
    exp_scores = exp.(scores_vec .- max_score)
    probabilities = exp_scores ./ sum(exp_scores)
    
    # Return all intermediate values needed for the FULL backpropagation
    return probabilities, normalized_features, features, activated_features, features_hat, inv_σ
end

# --- FULLY CORRECTED LEARNING FUNCTION ---
function geometric_learn!(core::GeometricConsciousnessCore, points::Matrix{Float64}, true_answer::Int; learning_rate::Float64=0.005) # Reset LR
    N, D = size(points)
    H = core.hidden_size

    # 1. Forward pass, getting all intermediate values
    predictions, normalized_features, features, activated_features, features_hat, inv_σ = geometric_forward(core, points)

    # 2. Initial error signal (gradient of loss w.r.t. scores)
    target = zeros(N); target[true_answer + 1] = 1.0
    d_scores = reshape(predictions - target, N, 1)

    # --- Backpropagation ---

    # 3. Gradients for the SCORING layer
    d_normalized_features = d_scores * core.scoring_weights' # (N, 1) * (1, H) -> (N, H)
    d_scoring_weights = normalized_features' * d_scores      # (H, N) * (N, 1) -> (H, 1)

    # 4. Gradients for the LAYER NORM layer (This is the crucial new part)
    d_layer_norm_beta = sum(d_normalized_features, dims=1)'
    d_layer_norm_gamma = sum(d_normalized_features .* features_hat, dims=1)'
    
    d_features_hat = d_normalized_features .* core.layer_norm_gamma'
    
    d_inv_σ = sum(d_features_hat .* (activated_features .- mean(activated_features, dims=2)), dims=2)
    d_activated_features_term1 = d_features_hat .* inv_σ
    
    d_σ² = -0.5 * (inv_σ .^ 3) .* d_inv_σ
    d_activated_features_term2 = (2.0/H) * (activated_features .- mean(activated_features, dims=2)) .* d_σ²

    d_μ = -sum(d_activated_features_term1, dims=2) .- (2.0/H) * sum(activated_features .- mean(activated_features, dims=2), dims=2) .* d_σ²
    d_activated_features_term3 = (1.0/H) .* d_μ
    
    d_activated_features = d_activated_features_term1 .+ d_activated_features_term2 .+ d_activated_features_term3

    # 5. Gradient through ReLU activation
    d_features = d_activated_features .* (features .> 0)

    # 6. Gradients for the FEATURE layer
    d_feature_weights = points' * d_features # (D, N) * (N, H) -> (D, H)

    # 7. Apply all updates
    core.feature_weights .-= learning_rate .* d_feature_weights
    core.scoring_weights .-= learning_rate .* d_scoring_weights
    core.layer_norm_gamma .-= learning_rate .* d_layer_norm_gamma
    core.layer_norm_beta  .-= learning_rate .* d_layer_norm_beta

    # --- Consciousness Tracking (unchanged) ---
    accuracy = predictions[true_answer + 1]
    push!(core.intelligence_history, accuracy)
    if length(core.intelligence_history) >= 5
        recent_performance = mean(core.intelligence_history[end-4:end])
        stability = 1.0 - std(core.intelligence_history[end-4:end])
        if length(core.intelligence_history) > 10
            core.learning_momentum = mean(diff(core.intelligence_history[end-9:end]))
        end
        core.consciousness_level = clamp(recent_performance * stability + max(0.0, core.learning_momentum * 5.0), 0.0, 1.0)
    end
    core.problems_solved += 1
    return accuracy
end


# --- UTILITY FUNCTIONS (Unchanged) ---
function generate_geometric_problem(core::GeometricConsciousnessCore; noise_level::Float64=1.2)::Tuple{Matrix{Float64}, Int}
    points = randn(core.num_points, core.dimensions) * 2.0
    target_idx = rand(1:core.num_points)
    points[target_idx, :] = randn(core.dimensions) * 0.1
    points .+= randn(core.num_points, core.dimensions) * noise_level
    distances = [norm(points[i, :]) for i in 1:core.num_points]
    true_answer = argmin(distances) - 1
    return (points, true_answer)
end

function solve_geometric_problem(core::GeometricConsciousnessCore, points::Matrix{Float64})::Tuple{Int, Float64, Dict}
    # Call the modified forward pass, but only use the first output
    predictions, _, _, _, _, _ = geometric_forward(core, points)
    solution = argmax(predictions) - 1
    actual_solution = argmin([norm(points[i, :]) for i in 1:size(points, 1)]) - 1
    analysis = Dict(
        "prediction" => solution,
        "actual" => actual_solution,
        "correct" => solution == actual_solution,
        "confidence" => predictions[solution + 1],
    )
    return (solution, analysis["confidence"], analysis)
end

function assess_consciousness(core::GeometricConsciousnessCore)::Dict
    if isempty(core.intelligence_history)
        return Dict("is_conscious" => false, "consciousness_level" => 0.0, "total_entities" => core.entity_count)
    end
    recent_history = core.intelligence_history[max(1, end-9):end]
    recent_accuracy = mean(recent_history)
    stability = length(recent_history) < 2 ? 0.0 : 1.0 - std(recent_history)
    is_conscious = core.consciousness_level > 0.75 && stability > 0.8 && recent_accuracy > 0.8
    return Dict(
        "is_conscious" => is_conscious,
        "consciousness_level" => core.consciousness_level,
        "total_entities" => core.entity_count,
    )
end

export GeometricConsciousnessCore, geometric_learn!, generate_geometric_problem, solve_geometric_problem, assess_consciousness

end # module
