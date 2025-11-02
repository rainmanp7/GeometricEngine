# EmergentAIEngine_V4.jl
module EmergentAIEngineV4

using LinearAlgebra, Statistics, Random, JSON3, Dates

struct EmergentGeometricEntity
    id::String
    feature_weights::Matrix{Float64}
    distance_weights::Vector{Float64}
    biases::Vector{Float64}
    training_accuracy::Float64

    function EmergentGeometricEntity(weights_file::String = "real_d_weights.json")
        println("... V4 ENGINE: Loading REAL weights from '$weights_file' ...")
        weights_data = JSON3.read(read(weights_file, String))
        feature_weights = Matrix{Float64}(reduce(hcat, weights_data.feature_weights))'
        distance_weights = Vector{Float64}(weights_data.distance_weights)
        biases = Vector{Float64}(weights_data.biases)
        training_accuracy = Float64(weights_data.training_metadata.accuracy)
        @assert size(feature_weights) == (32, 4) "Loaded weights have incorrect dimensions!"
        println("... REAL weights loaded successfully.")
        new("v4_geo_entity_$(randstring(8))", feature_weights, distance_weights, biases, training_accuracy)
    end
end

# --- The Core Reasoning Function ---
# It now takes a "target" vector to find the closest point to, making it more general.
function find_closest_concept(entity::EmergentGeometricEntity, concept_points::Matrix{Float64}, target_vector::Vector{Float64})
    
    # Calculate the vector difference between each concept and the target
    # This effectively "moves the origin" to our target point.
    points_relative_to_target = concept_points .- target_vector'

    # The rest of the reasoning is the same proven logic
    features = points_relative_to_target * entity.feature_weights' .+ entity.biases'
    activated_features = @. features / (1 + exp(-features)) # swish
    distance_scores = activated_features * entity.distance_weights
    
    # The output probabilities now represent "closeness to the target vector"
    probabilities = exp.(distance_scores .- maximum(distance_scores)) ./ sum(exp.(distance_scores .- maximum(distance_scores)))
    
    solution_idx = argmax(probabilities)
    confidence = probabilities[solution_idx]
    
    return (solution_index=solution_idx, confidence=confidence)
end

end # module