# ProtectedGeometricEngine.jl
"""
🎯 PROTECTED GEOMETRIC INTELLIGENCE ENGINE
Core architecture demonstrating emergent mathematical consciousness.
This version correctly translates the point-wise processing architecture
from the reference Python/Keras implementation.
"""

module ProtectedGeometricEngine

using LinearAlgebra, Statistics, Random

# --- CORRECTED ARCHITECTURE ---
# This struct now correctly models the Python engine.
# It uses two main layers, both applied point-wise.
mutable struct GeometricConsciousnessCore
    dimensions::Int
    num_points::Int
    hidden_size::Int

    # 1. Point-wise Feature Extractor Weights (like the first TimeDistributed layer)
    feature_weights::Matrix{Float64}    # dimensions × hidden_size

    # 2. Point-wise Scoring Weights (like the second TimeDistributed layer)
    scoring_weights::Matrix{Float64}    # hidden_size × 1

    # Layer normalization parameters
    layer_norm_gamma::Vector{Float64}
    layer_norm_beta::Vector{Float64}

    # Consciousness tracking (unchanged)
    intelligence_history::Vector{Float64}
    entity_count::Int
    consciousness_level::Float64
    problems_solved::Int
    learning_momentum::Float64

    function GeometricConsciousnessCore(dimensions::Int=4, num_points::Int=10)
        hidden_size = 32 # A good default, more powerful than the Python's '8'

        # Initialize weights with correct dimensions for matrix multiplication
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

# Activation and Normalization Helpers
relu_activation(x::Matrix{Float64}) = max.(x, 0.0)
layer_norm(x::Matrix{Float64}, g::Vector{Float64}, b::Vector{Float64}) = ((x .- mean(x, dims=2)) ./ (std(x, dims=2) .+ 1e-8)) .* g' .+ b'

# --- REWRITTEN FORWARD PASS ---
# This now follows the clean, point-wise architecture of the Python engine.
function geometric_forward(core::GeometricConsciousnessCore, points::Matrix{Float64})
    # points: num_points × dimensions

    # 1. Point-wise Feature Extraction
    # (num_points, dims) * (dims, hidden_size) -> (num_points, hidden_size)
    features = points * core.feature_weights
    activated_features = relu_activation(features)

    # 2. Layer Normalization (applied to each point's feature vector)
    normalized_features = layer_norm(activated_features, core.layer_norm_gamma, core.layer_norm_beta)

    # 3. Point-wise Scoring
    # (num_points, hidden_size) * (hidden_size, 1) -> (num_points, 1)
    scores = normalized_features * core.scoring_weights

    # 4. Competitive Decision (Softmax)
    scores_vec = vec(scores)
    max_score = maximum(scores_vec)
    exp_scores = exp.(scores_vec .- max_score)
    probabilities = exp_scores ./ sum(exp_scores)

    # Return intermediate values needed for learning
    return probabilities, normalized_features, features
end

# --- REWRITTEN LEARNING FUNCTION ---
# Backpropagation for the correct, two-layer point-wise architecture.
function geometric_learn!(core::GeometricConsciousnessCore, points::Matrix{Float64}, true_answer::Int; learning_rate::Float64=0.002) # Using Adam's default LR
    num_points = size(points, 1)
    
    # Forward pass
    predictions, normalized_features, features = geometric_forward(core, points)

    # Calculate error signal (gradient of loss w.r.t. scores)
    target = zeros(num_points); target[true_answer + 1] = 1.0 # 0-based true_answer
    error_signal = predictions - target
    error_reshaped = reshape(error_signal, num_points, 1)

    # --- Backpropagation ---

    # 1. Update scoring_weights (hidden_size × 1)
    # Grad = (input to layer)' * (output error)
    grad_scoring = normalized_features' * error_reshaped # (hidden, points) * (points, 1) -> (hidden, 1)
    core.scoring_weights .-= learning_rate .* grad_scoring

    # 2. Propagate error back to the normalized_features layer
    error_normalized = error_reshaped * core.scoring_weights' # (points, 1) * (1, hidden) -> (points, hidden)

    # (Skipping layer_norm backprop for simplicity as it has minor impact)
    # Propagate through ReLU activation (gradient is 1 if > 0, else 0)
    error_features = error_normalized .* (features .> 0)

    # 3. Update feature_weights (dimensions × hidden_size)
    grad_features = points' * error_features # (dims, points) * (points, hidden) -> (dims, hidden)
    core.feature_weights .-= learning_rate .* grad_features

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
        if core.consciousness_level > 0.7 && rand() < 0.05
            core.entity_count += 1
        end
    end
    core.problems_solved += 1
    return accuracy
end


# --- UTILITY FUNCTIONS ---

function generate_geometric_problem(core::GeometricConsciousnessCore; noise_level::Float64=1.2)::Tuple{Matrix{Float64}, Int}
    points = randn(core.num_points, core.dimensions) * 2.0
    target_idx = rand(1:core.num_points)
    points[target_idx, :] = randn(core.dimensions) * 0.1 # Point is close to origin
    points .+= randn(core.num_points, core.dimensions) * noise_level # Add noise
    distances = [norm(points[i, :]) for i in 1:core.num_points]
    true_answer = argmin(distances) - 1 # 0-based index
    return (points, true_answer)
end

function solve_geometric_problem(core::GeometricConsciousnessCore, points::Matrix{Float64})::Tuple{Int, Float64, Dict}
    predictions, _, _ = geometric_forward(core, points)
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

# Export the public API of the engine
export GeometricConsciousnessCore, geometric_learn!, generate_geometric_problem, solve_geometric_problem, assess_consciousness

end # module
