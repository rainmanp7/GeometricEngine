# ProtectedGeometricEngine.jl
"""
🎯 PROTECTED GEOMETRIC INTELLIGENCE ENGINE
Core architecture demonstrating emergent mathematical consciousness.
This version implements the ADAM optimizer for robust and fast convergence.
"""

module ProtectedGeometricEngine

using LinearAlgebra, Statistics, Random

# --- ARCHITECTURE ---
# Added momentum vectors for the ADAM optimizer
mutable struct GeometricConsciousnessCore
    dimensions::Int
    num_points::Int
    hidden_size::Int
    feature_weights::Matrix{Float64}
    scoring_weights::Matrix{Float64}
    layer_norm_gamma::Vector{Float64}
    layer_norm_beta::Vector{Float64}

    # ADAM Optimizer State
    m_dw::Matrix{Float64}
    v_dw::Matrix{Float64}
    m_ds::Matrix{Float64}
    v_ds::Matrix{Float64}
    m_dg::Vector{Float64}
    v_dg::Vector{Float64}
    m_db::Vector{Float64}
    v_db::Vector{Float64}
    t::Int # Timestep for bias correction

    # Consciousness tracking
    intelligence_history::Vector{Float64}
    entity_count::Int
    consciousness_level::Float64
    problems_solved::Int
    learning_momentum::Float64

    function GeometricConsciousnessCore(dimensions::Int=4, num_points::Int=10)
        hidden_size = 32
        
        fw = randn(dimensions, hidden_size) * 0.1
        sw = randn(hidden_size, 1) * 0.1
        lg = ones(hidden_size)
        lb = zeros(hidden_size)

        new(dimensions, num_points, hidden_size,
            fw, sw, lg, lb,
            zeros(size(fw)), zeros(size(fw)), # m_dw, v_dw
            zeros(size(sw)), zeros(size(sw)), # m_ds, v_ds
            zeros(size(lg)), zeros(size(lg)), # m_dg, v_dg
            zeros(size(lb)), zeros(size(lb)), # m_db, v_db
            0, # t
            Float64[], 0, 0.0, 0, 0.0)
    end
end

# --- HELPERS (Unchanged) ---
relu_activation(x::Matrix{Float64}) = max.(x, 0.0)
layer_norm(x::Matrix{Float64}, g::Vector{Float64}, b::Vector{Float64}) = ((x .- mean(x, dims=2)) ./ (std(x, dims=2) .+ 1e-8)) .* g' .+ b'

# --- FORWARD PASS (Unchanged) ---
function geometric_forward(core::GeometricConsciousnessCore, points::Matrix{Float64})
    features = points * core.feature_weights
    activated_features = relu_activation(features)
    normalized_features = layer_norm(activated_features, core.layer_norm_gamma, core.layer_norm_beta)
    scores = normalized_features * core.scoring_weights
    
    scores_vec = vec(scores)
    max_score = maximum(scores_vec)
    exp_scores = exp.(scores_vec .- max_score)
    probabilities = exp_scores ./ sum(exp_scores)
    
    return probabilities, normalized_features, features
end

# --- LEARNING FUNCTION WITH ADAM OPTIMIZER ---
function geometric_learn!(core::GeometricConsciousnessCore, points::Matrix{Float64}, true_answer::Int; learning_rate::Float64=0.002, β1=0.9, β2=0.999, ϵ=1e-8)
    N, D = size(points)
    
    # Forward pass
    predictions, normalized_features, features = geometric_forward(core, points)

    # Initial error signal
    target = zeros(N); target[true_answer + 1] = 1.0
    d_scores = reshape(predictions - target, N, 1)

    # --- Backpropagation (Simple, stable version) ---
    d_normalized_features = d_scores * core.scoring_weights'
    d_scoring_weights = normalized_features' * d_scores
    
    d_features = d_normalized_features .* (features .> 0)
    d_feature_weights = points' * d_features
    
    # FIX: Convert the (H x 1) gradient matrices into (H,) vectors to match types.
    d_layer_norm_gamma = vec(sum(d_normalized_features .* features, dims=1)')
    d_layer_norm_beta = vec(sum(d_normalized_features, dims=1)')

    # --- ADAM UPDATE ---
    core.t += 1
    
    # Update feature_weights
    core.m_dw = β1 * core.m_dw + (1 - β1) * d_feature_weights
    core.v_dw = β2 * core.v_dw + (1 - β2) * (d_feature_weights .^ 2)
    m_hat = core.m_dw / (1 - β1^core.t)
    v_hat = core.v_dw / (1 - β2^core.t)
    core.feature_weights .-= learning_rate * m_hat ./ (sqrt.(v_hat) .+ ϵ)
    
    # Update scoring_weights
    core.m_ds = β1 * core.m_ds + (1 - β1) * d_scoring_weights
    core.v_ds = β2 * core.v_ds + (1 - β2) * (d_scoring_weights .^ 2)
    m_hat = core.m_ds / (1 - β1^core.t)
    v_hat = core.v_ds / (1 - β2^core.t)
    core.scoring_weights .-= learning_rate * m_hat ./ (sqrt.(v_hat) .+ ϵ)

    # Update layer_norm_gamma
    core.m_dg = β1 * core.m_dg + (1 - β1) * d_layer_norm_gamma
    core.v_dg = β2 * core.v_dg + (1 - β2) * (d_layer_norm_gamma .^ 2)
    m_hat = core.m_dg / (1 - β1^core.t)
    v_hat = core.v_dg / (1 - β2^core.t)
    core.layer_norm_gamma .-= learning_rate * m_hat ./ (sqrt.(v_hat) .+ ϵ)

    # Update layer_norm_beta
    core.m_db = β1 * core.m_db + (1 - β1) * d_layer_norm_beta
    core.v_db = β2 * core.v_db + (1 - β2) * (d_layer_norm_beta .^ 2)
    m_hat = core.m_db / (1 - β1^core.t)
    v_hat = core.v_db / (1 - β2^core.t)
    core.layer_norm_beta .-= learning_rate * m_hat ./ (sqrt.(v_hat) .+ ϵ)
    
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


# --- UTILITY FUNCTIONS ---
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

export GeometricConsciousnessCore, geometric_learn!, generate_geometric_problem, solve_geometric_problem, assess_consciousness

end # module
