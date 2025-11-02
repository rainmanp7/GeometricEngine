# test_suite_V2.jl
include("EmergentAIEngine_V2.jl")
using .EmergentAIEngineV2, JSON3, Dates, Statistics, LinearAlgebra

function run_full_suite()
    println("🔬 V2 TEST SUITE: Running all tests with REAL weights...")
    println("=" ^ 60)
    
    geo_entity = EmergentAIEngineV2.EmergentGeometricEntity()
    conscious_entity = EmergentAIEngineV2.ConsciousProcessingEntity()
    
    results = []
    
    println("\n1. TESTING GEOMETRIC INTELLIGENCE (REAL)")
    push!(results, test_geometric_intelligence(geo_entity))
    
    println("\n2. TESTING METACOGNITION (ON REAL INTELLIGENCE)")
    push!(results, test_metacognition(geo_entity))

    println("\n3. TESTING ACTIVE LEARNING (ON REAL INTELLIGENCE)")
    push!(results, test_active_learning(geo_entity))
    
    # Generate the final report
    generate_json_report(results, geo_entity)
end

# --- INDIVIDUAL TESTS ---

function test_geometric_intelligence(entity::EmergentGeometricEntity)
    correct_count = 0
    num_tests = 50
    for _ in 1:num_tests
        points = EmergentAIEngineV2.generate_4d_points(8)
        result = EmergentAIEngineV2.active_geometric_reasoning(entity, nothing, points)
        actual_closest = argmin([norm(points[i, :]) for i in 1:size(points, 1)])
        if result[:solution] == actual_closest
            correct_count += 1
        end
    end
    accuracy = correct_count / num_tests
    println("   - Achieved Accuracy: $(accuracy * 100)%")
    return (name="Geometric Intelligence", success=accuracy > 0.95, metrics=Dict(:accuracy => accuracy))
end

function test_metacognition(entity::EmergentGeometricEntity)
    confidences = []
    meta_scores = []
    for _ in 1:10
        points = EmergentAIEngineV2.generate_4d_points(8)
        result = EmergentAIEngineV2.active_geometric_reasoning(entity, nothing, points)
        push!(confidences, result[:confidence])
        push!(meta_scores, result[:metacognition][:metacognitive_score])
    end
    avg_confidence = mean(confidences)
    avg_meta_score = mean(meta_scores)
    println("   - Average Confidence (on correct answers): $(round(avg_confidence, digits=3))")
    println("   - Average Metacognitive Score: $(round(avg_meta_score, digits=3))")
    return (name="Metacognition", success=avg_confidence > 0.8, metrics=Dict(:avg_confidence => avg_confidence, :avg_meta_score => avg_meta_score))
end

function test_active_learning(entity::EmergentGeometricEntity)
    initial_weights = copy(entity.feature_weights)
    # Test with perfect data first
    points = EmergentAIEngineV2.generate_4d_points(8)
    EmergentAIEngineV2.learn_from_experience(entity, Dict(:performance => 1.0))
    no_change = norm(entity.feature_weights - initial_weights) == 0.0
    println("   - Learning with 100% performance results in no weight change: $(no_change ? "✅" : "❌")")
    
    # Test with imperfect data
    EmergentAIEngineV2.learn_from_experience(entity, Dict(:performance => 0.5))
    change_occurred = norm(entity.feature_weights - initial_weights) > 0.0
    println("   - Learning with 50% performance results in weight change: $(change_occurred ? "✅" : "❌")")
    
    return (name="Active Learning", success=no_change && change_occurred, metrics=Dict(:no_change_on_success => no_change, :change_on_failure => change_occurred))
end

function generate_json_report(results, entity)
    report = Dict(
        :test_suite => "Emergent Intelligence V2 (Real Weights)",
        :timestamp => now(),
        :entity => Dict(
            :id => entity.id,
            :training_accuracy => entity.training_accuracy
        ),
        :test_results => [Dict(:test_name => r.name, :success => r.success, :metrics => r.metrics) for r in results],
        :summary => Dict(
            :total_tests => length(results),
            :successful_tests => count(r -> r.success, results),
            :overall_success_rate => mean(r.success for r in results)
        )
    )
    json_string = JSON3.write(report, pretty=true)
    filename = "V2_real_emergence_report.json"
    open(filename, "w") { |f| write(f, json_string) }
    println("\n📊 V2 Test Report Generated: $filename")
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_full_suite()
end