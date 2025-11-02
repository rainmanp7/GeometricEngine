# EmergentAIEngine_V3.jl
module EmergentAIEngineV3

using LinearAlgebra, Statistics, Random, JSON3, Dates

# The core entity is the same as V2. It's already perfect.
struct EmergentGeometricEntity
    id::String
    feature_weights::Matrix{Float64}
    distance_weights::Vector{Float64}
    biases::Vector{Float64}
    training_accuracy::Float64

    function EmergentGeometricEntity(weights_file::String = "real_d_weights.json")
        println("... V3 ENGINE: Loading REAL weights from '$weights_file' ...")
        weights_data = JSON3.read(read(weights_file, String))
        feature_weights = Matrix{Float64}(reduce(hcat, weights_data.feature_weights))'
        distance_weights = Vector{Float64}(weights_data.distance_weights)
        biases = Vector{Float64}(weights_data.biases)
        training_accuracy = Float64(weights_data.training_metadata.accuracy)
        @assert size(feature_weights) == (32, 4) "Loaded weights have incorrect dimensions!"
        println("... REAL weights loaded successfully. Training Accuracy: $(training_accuracy)")
        new("v3_geo_entity_$(randstring(8))", feature_weights, distance_weights, biases, training_accuracy)
    end
end

# --- THE KEY UPGRADE: INSTRUMENTED REASONING ---
# This function now returns its own performance metrics.
function instrumented_reasoning(entity::EmergentGeometricEntity, points::Matrix{Float64})
    # Use @timed macro to capture execution time, memory allocation, and the result.
    stats = @timed begin
        features = points * entity.feature_weights' .+ entity.biases'
        activated_features = @. features / (1 + exp(-features)) # swish
        distance_scores = activated_features * entity.distance_weights
        
        probabilities = exp.(distance_scores .- maximum(distance_scores)) ./ sum(exp.(distance_scores .- maximum(distance_scores)))
        
        solution_idx = argmax(probabilities)
        confidence = probabilities[solution_idx]
        
        # Return the core answer
        (solution=solution_idx, confidence=confidence)
    end

    # The ground truth (what the answer should be)
    actual_closest_idx = argmin([norm(points[i, :]) for i in 1:size(points, 1)])
    
    # Package everything into a neat report
    return (
        result = stats.value,
        correct = stats.value.solution == actual_closest_idx,
        execution_time_ns = stats.time * 1e9, # Convert seconds to nanoseconds
        memory_bytes = stats.bytes,
        gc_time_ns = stats.gctime * 1e9
    )
end

# Utility to generate the familiar spatial data for our baseline test.
function generate_spatial_points(n_points)
    points = randn(n_points, 4) .* 2
    closest_idx = rand(1:n_points)
    points[closest_idx, :] .= randn(4) .* 0.1
    return points
end

end # module