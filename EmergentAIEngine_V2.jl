# EmergentAIEngine_V2.jl
module EmergentAIEngineV2

using LinearAlgebra, Statistics, Random, JSON3, Dates

# ==================== CORE EMERGENT ENTITIES (NOW WITH REAL WEIGHTS) ====================

abstract type AbstractEmergentEntity end

# This is our core intelligent agent, now loading its "brain" from your JSON file.
struct EmergentGeometricEntity <: AbstractEmergentEntity
    id::String
    feature_weights::Matrix{Float64}
    distance_weights::Vector{Float64}
    biases::Vector{Float64}
    activation_history::Vector{Dict}
    metacognitive_weights::Dict{Symbol, Any}
    training_accuracy::Float64

    function EmergentGeometricEntity(weights_file::String = "real_d_weights.json")
        println("... V2 ENGINE: Loading REAL weights from '$weights_file' ...")
        
        json_string = read(weights_file, String)
        weights_data = JSON3.read(json_string)

        feature_weights = Matrix{Float64}(reduce(hcat, weights_data.feature_weights))'
        distance_weights = Vector{Float64}(weights_data.distance_weights)
        biases = Vector{Float64}(weights_data.biases)
        training_accuracy = Float64(weights_data.training_metadata.accuracy)

        @assert size(feature_weights) == (32, 4) "Loaded feature weights have incorrect dimensions!"
        println("... REAL weights loaded successfully. Training Accuracy: $(training_accuracy)")
        
        id = "v2_geo_entity_$(randstring(8))"
        
        metacognitive_weights = Dict(
            :self_monitoring => [0.45, -0.38, 0.52],
            :confidence_calibration => [0.67, -0.29, 0.58],
            :cross_domain_mapping => [0.31, 0.42, -0.27, 0.53]
        )
        
        new(id, feature_weights, distance_weights, biases, [], metacognitive_weights, training_accuracy)
    end
end

# This entity remains the same. It's a "tool" the main entity can use.
struct ConsciousProcessingEntity <: AbstractEmergentEntity
    id::String
    attention_weights::Matrix{Float64}
    consciousness_level::Float64
    
    function ConsciousProcessingEntity()
        id = "v2_conscious_entity_$(randstring(8))"
        attention_weights = [0.33 -0.27 0.41 -0.19; -0.22 0.38 -0.31 0.25; 0.29 -0.35 0.37 -0.24; -0.26 0.32 -0.28 0.39]
        new(id, attention_weights, 0.75)
    end
end

# ==================== HIGH-LEVEL FUNCTIONS (Now operating on a REAL intelligence) ====================

# This is the core reasoning function, using the proven real-weights logic.
function active_geometric_reasoning(entity::EmergentGeometricEntity, conscious_entity::Union{ConsciousProcessingEntity, Nothing}, points::Matrix{Float64})
    processed_points = !isnothing(conscious_entity) ? (points * conscious_entity.attention_weights) : points
    
    features = processed_points * entity.feature_weights' .+ entity.biases'
    activated_features = @. features / (1 + exp(-features)) # swish
    distance_scores = activated_features * entity.distance_weights
    
    probabilities = exp.(distance_scores .- maximum(distance_scores)) ./ sum(exp.(distance_scores .- maximum(distance_scores)))
    
    closest_idx = argmax(probabilities)
    confidence = probabilities[closest_idx]
    
    problem_type = "real_geometric_closest_$(size(points, 2))D"
    metacognition = assess_own_thinking(entity, problem_type, Dict(:confidence => confidence))
    
    return Dict(
        :solution => closest_idx, 
        :confidence => confidence, 
        :metacognition => metacognition, 
        :consciousness_level => isnothing(conscious_entity) ? 0.0 : conscious_entity.consciousness_level
    )
end

function assess_own_thinking(entity::EmergentGeometricEntity, problem::String, solution::Dict)
    confidence = solution[:confidence]
    novelty = isempty(entity.activation_history) ? 1.0 : (1.0 - count(h -> h[:problem_type] == problem, entity.activation_history) / length(entity.activation_history))
    
    metacognitive_score = (entity.metacognitive_weights[:self_monitoring][1] * confidence) + (entity.metacognitive_weights[:self_monitoring][3] * novelty)
    
    metacognitive_event = Dict(:timestamp => now(), :problem_type => problem, :confidence => confidence, :metacognitive_score => metacognitive_score)
    push!(entity.activation_history, metacognitive_event)
    return metacognitive_event
end

function learn_from_experience(entity::EmergentGeometricEntity, feedback::Dict)
    learning_rate = 0.01
    performance = feedback[:performance] # This will be 1.0 most of the time now.
    
    if performance < 0.95 # Only learn if not perfect
        adjustment = learning_rate * (1 - performance)
        entity.feature_weights .-= adjustment # Weaken patterns that lead to failure
    end
    return entity
end

# Utility to generate test data
function generate_4d_points(n_points)
    points = randn(n_points, 4) .* 2
    closest_idx = rand(1:n_points)
    points[closest_idx, :] .= randn(4) .* 0.1
    return points
end

end # module