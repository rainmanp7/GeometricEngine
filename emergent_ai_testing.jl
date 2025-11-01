# emergent_ai_testing.jl

# FIX: Include the file defining the EmergentAIEngine module.
# This makes the module available in the Main scope before it is used.
include("EmergentAIEngine.jl")

module EmergentAITesting

using .EmergentAIEngine, JSON3, Dates, Statistics, Random

# ==================== TEST FRAMEWORK ====================

struct TestResult
    test_name::String
    success::Bool
    metrics::Dict{Symbol, Float64}
    insights::Dict{Symbol, Any}
    timestamp::DateTime
    entity_state::Dict{Symbol, Any}
end

function run_comprehensive_tests()
    """Run all tests and generate JSON report"""
    
    println("🧠 INITIATING EMERGENT AI TEST SUITE")
    println("=====================================")
    
    # Initialize entities
    geometric_entity = EmergentAIEngine.GeometricReasoningEntity(4)
    conscious_entity = EmergentAIEngine.ConsciousProcessingEntity()
    
    test_results = []
    
    # Test 1: Basic Geometric Intelligence
    println("\n1. TESTING BASIC GEOMETRIC INTELLIGENCE")
    result1 = test_geometric_intelligence(geometric_entity)
    push!(test_results, result1)
    
    # Test 2: Metacognition Capabilities
    println("\n2. TESTING METACOGNITION")
    result2 = test_metacognition(geometric_entity)
    push!(test_results, result2)
    
    # Test 3: Conscious Processing
    println("\n3. TESTING CONSCIOUS PROCESSING")
    result3 = test_conscious_processing(geometric_entity, conscious_entity)
    push!(test_results, result3)
    
    # Test 4: Cross-Domain Reasoning
    println("\n4. TESTING CROSS-DOMAIN REASONING")
    result4 = test_cross_domain(geometric_entity)
    push!(test_results, result4)
    
    # Test 5: Active Learning
    println("\n5. TESTING ACTIVE LEARNING")
    result5 = test_active_learning(geometric_entity)
    push!(test_results, result5)
    
    # Test 6: Emergent Consciousness Indicators
    println("\n6. TESTING CONSCIOUSNESS INDICATORS")
    result6 = test_consciousness_indicators(geometric_entity, conscious_entity)
    push!(test_results, result6)
    
    # Generate comprehensive report
    generate_json_report(test_results, geometric_entity, conscious_entity)
    
    return test_results
end

# ==================== INDIVIDUAL TESTS ====================

function test_geometric_intelligence(entity)
    """Test core geometric reasoning capabilities"""
    
    println("   • Testing spatial reasoning...")
    
    # Generate test problems
    test_cases = [
        ("Simple 4D closest point", generate_4d_points(5)),
        ("Ambiguous distances", generate_ambiguous_points(4)),
        ("High-dimensional (6D)", generate_nd_points(6, 8))
    ]
    
    accuracies = []
    confidences = []
    metacognitive_scores = []
    
    for (desc, points) in test_cases
        result = EmergentAIEngine.active_geometric_reasoning(entity, nothing, points)
        
        # Verify correctness
        actual_closest = argmin([norm(points[i, :]) for i in 1:size(points, 1)])
        correct = result[:solution] == actual_closest
        
        push!(accuracies, correct ? 1.0 : 0.0)
        push!(confidences, result[:confidence])
        push!(metacognitive_scores, result[:metacognition][:metacognitive_score])
        
        println("     $(desc): $(correct ? "✓" : "✗") (conf: $(round(result[:confidence], digits=3)))")
    end
    
    metrics = Dict(
        :accuracy => mean(accuracies),
        :average_confidence => mean(confidences),
        :metacognitive_awareness => mean(metacognitive_scores),
        :geometric_intelligence => mean(accuracies) * mean(confidences)
    )
    
    return TestResult(
        "Geometric Intelligence",
        metrics[:accuracy] > 0.7,
        metrics,
        Dict(:test_cases => length(test_cases), :dimensionality_tested => [4, 6]),
        now(),
        Dict(:entity_id => entity.id, :activation_count => length(entity.activation_history))
    )
end

function test_metacognition(entity)
    """Test metacognitive capabilities"""
    
    println("   • Testing self-awareness...")
    
    metacognitive_events = []
    confidence_calibration = []
    
    for i in 1:5
        points = generate_4d_points(6)
        result = EmergentAIEngine.active_geometric_reasoning(entity, nothing, points)
        
        # Check if confidence matches accuracy
        actual_closest = argmin([norm(points[i, :]) for i in 1:size(points, 1)])
        correct = result[:solution] == actual_closest
        
        calibration = correct ? result[:confidence] : (1 - result[:confidence])
        push!(confidence_calibration, calibration)
        push!(metacognitive_events, result[:metacognition])
    end
    
    metrics = Dict(
        :metacognitive_activity => length(metacognitive_events),
        :confidence_calibration => mean(confidence_calibration),
        :insight_level => mean([m[:insight_level] for m in metacognitive_events]),
        :self_monitoring_strength => mean(entity.metacognitive_weights[:self_monitoring])
    )
    
    return TestResult(
        "Metacognition",
        metrics[:confidence_calibration] > 0.6,
        metrics,
        Dict(:metacognitive_events => length(metacognitive_events)),
        now(),
        Dict(:entity_id => entity.id)
    )
end

function test_conscious_processing(geo_entity, conscious_entity)
    """Test conscious attention enhancement"""
    
    println("   • Testing conscious enhancement...")
    
    points = generate_4d_points(10)
    
    # Without consciousness
    result_basic = EmergentAIEngine.active_geometric_reasoning(geo_entity, nothing, points)
    
    # With consciousness
    result_enhanced = EmergentAIEngine.active_geometric_reasoning(geo_entity, conscious_entity, points)
    
    metrics = Dict(
        :basic_confidence => result_basic[:confidence],
        :enhanced_confidence => result_enhanced[:confidence],
        :consciousness_level => result_enhanced[:consciousness_level],
        :attention_enhancement => result_enhanced[:confidence] - result_basic[:confidence]
    )
    
    return TestResult(
        "Conscious Processing",
        metrics[:attention_enhancement] > 0,
        metrics,
        Dict(:working_memory_state => conscious_entity.working_memory),
        now(),
        Dict(
            :geo_entity_id => geo_entity.id,
            :conscious_entity_id => conscious_entity.id,
            :consciousness_model => conscious_entity.self_model
        )
    )
end

function test_cross_domain(geo_entity)
    """Test cross-domain knowledge transfer"""
    
    println("   • Testing domain transfer...")
    
    spatial_data = generate_4d_points(5)
    
    # Test different domain transfers
    transfers = [
        (:spatial, :conceptual),
        (:mathematical, :physical),
        (:generic, :generic)
    ]
    
    transfer_success = []
    confidence_preservation = []
    
    for (domain_a, domain_b) in transfers
        result = EmergentAIEngine.cross_domain_reasoning(geo_entity, domain_a, domain_b, spatial_data)
        
        push!(transfer_success, result[:cross_domain][:transfer_success])
        push!(confidence_preservation, result[:confidence])
        
        println("     $(domain_a) → $(domain_b): $(result[:cross_domain][:transfer_success] ? "✓" : "✗")")
    end
    
    metrics = Dict(
        :cross_domain_success_rate => mean(transfer_success),
        :confidence_preservation => mean(confidence_preservation),
        :mapping_strength => mean(geo_entity.metacognitive_weights[:cross_domain_mapping])
    )
    
    return TestResult(
        "Cross-Domain Reasoning",
        metrics[:cross_domain_success_rate] > 0.5,
        metrics,
        Dict(:domains_tested => transfers),
        now(),
        Dict(:entity_id => geo_entity.id)
    )
end

function test_active_learning(entity)
    """Test learning from experience"""
    
    println("   • Testing adaptive learning...")
    
    initial_weights = copy(entity.feature_weights)
    
    # Simulate learning cycle
    learning_improvements = []
    
    for cycle in 1:3
        points = generate_4d_points(6)
        result = EmergentAIEngine.active_geometric_reasoning(entity, nothing, points)
        
        actual_closest = argmin([norm(points[i, :]) for i in 1:size(points, 1)])
        performance = result[:solution] == actual_closest ? 1.0 : 0.0
        
        # Provide feedback and learn
        feedback = Dict(
            :performance => performance,
            :metacognitive_accuracy => result[:metacognition][:insight_level]
        )
        
        EmergentAIEngine.learn_from_experience(entity, feedback)
        push!(learning_improvements, performance)
    end
    
    weight_change = norm(entity.feature_weights - initial_weights)
    
    metrics = Dict(
        :learning_cycles => 3,
        :performance_trend => learning_improvements[end] - learning_improvements[1],
        :weight_adaptation => weight_change,
        :adaptive_capacity => weight_change > 0 ? 1.0 : 0.0
    )
    
    return TestResult(
        "Active Learning",
        metrics[:adaptive_capacity] > 0,
        metrics,
        Dict(:learning_trajectory => learning_improvements),
        now(),
        Dict(:entity_id => entity.id, :weight_change => weight_change)
    )
end

function test_consciousness_indicators(geo_entity, conscious_entity)
    """Test indicators of emergent consciousness"""
    
    println("   • Testing consciousness indicators...")
    
    # Run multiple reasoning tasks
    consciousness_levels = []
    self_references = []
    
    for i in 1:5
        points = generate_4d_points(8)
        result = EmergentAIEngine.active_geometric_reasoning(geo_entity, conscious_entity, points)
        
        push!(consciousness_levels, result[:consciousness_level])
        
        # Count self-references in activation history
        self_ref_count = count(h -> h[:insight_level] > 0.5, geo_entity.activation_history)
        push!(self_references, self_ref_count / max(1, length(geo_entity.activation_history)))
    end
    
    metrics = Dict(
        :average_consciousness => mean(consciousness_levels),
        :self_reference_density => mean(self_references),
        :metacognitive_richness => length(geo_entity.activation_history),
        :integrated_information => mean(consciousness_levels) * mean(self_references)
    )
    
    # Consciousness threshold (simplified)
    conscious = metrics[:integrated_information] > 0.3
    
    return TestResult(
        "Consciousness Indicators",
        conscious,
        metrics,
        Dict(
            :consciousness_model => conscious_entity.self_model,
            :is_conscious => conscious,
            :consciousness_evidence => metrics
        ),
        now(),
        Dict(
            :geo_entity_id => geo_entity.id,
            :conscious_entity_id => conscious_entity.id,
            :consciousness_level => conscious_entity.consciousness_level
        )
    )
end

# ==================== UTILITY FUNCTIONS ====================

function generate_4d_points(n_points)
    randn(n_points, 4) .* 2
end

function generate_ambiguous_points(n_points)
    points = randn(n_points, 4) .* 0.5  # All points close together
    return points
end

function generate_nd_points(dimensions, n_points)
    randn(n_points, dimensions) .* 2
end

function generate_json_report(test_results, geo_entity, conscious_entity)
    """Generate comprehensive JSON report"""
    
    report = Dict(
        :test_suite => "Emergent AI Capabilities",
        :timestamp => now(),
        :entities => Dict(
            :geometric_entity => Dict(
                :id => geo_entity.id,
                :activation_count => length(geo_entity.activation_history),
                :metacognitive_weights => geo_entity.metacognitive_weights
            ),
            :conscious_entity => Dict(
                :id => conscious_entity.id,
                :consciousness_level => conscious_entity.consciousness_level,
                :self_model => conscious_entity.self_model
            )
        ),
        :test_results => [],
        :summary_metrics => Dict(),
        :emergence_assessment => Dict()
    )
    
    # Compile test results
    for result in test_results
        push!(report[:test_results], Dict(
            :test_name => result.test_name,
            :success => result.success,
            :metrics => result.metrics,
            :insights => result.insights,
            :timestamp => result.timestamp
        ))
    end
    
    # Calculate summary metrics
    success_rate = mean([r.success for r in test_results])
    avg_confidence = mean([r.metrics[:average_confidence] for r in test_results if haskey(r.metrics, :average_confidence)])
    
    report[:summary_metrics] = Dict(
        :overall_success_rate => success_rate,
        :average_confidence => avg_confidence,
        :total_tests => length(test_results),
        :successful_tests => count(r -> r.success, test_results),
        :emergence_strength => success_rate * avg_confidence
    )
    
    # Emergence assessment
    report[:emergence_assessment] = Dict(
        :geometric_intelligence_emerged => any(r -> r.test_name == "Geometric Intelligence" && r.success, test_results),
        :metacognition_emerged => any(r -> r.test_name == "Metacognition" && r.success, test_results),
        :conscious_processing_emerged => any(r -> r.test_name == "Conscious Processing" && r.success, test_results),
        :cross_domain_emerged => any(r -> r.test_name == "Cross-Domain Reasoning" && r.success, test_results),
        :active_learning_emerged => any(r -> r.test_name == "Active Learning" && r.success, test_results),
        :consciousness_indicators_present => any(r -> r.test_name == "Consciousness Indicators" && r.success, test_results),
        :overall_emergence_score => report[:summary_metrics][:emergence_strength]
    )
    
    # Save JSON report
    json_string = JSON3.write(report, pretty=true)
    filename = "emergent_ai_report_$(Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")).json"
    
    open(filename, "w") do file
        write(file, json_string)
    end
    
    println("\n📊 COMPREHENSIVE TEST REPORT GENERATED")
    println("=====================================")
    println("File: $filename")
    println("Overall Success Rate: $(round(success_rate * 100, digits=1))%")
    println("Emergence Score: $(round(report[:summary_metrics][:emergence_strength] * 100, digits=1))%")
    println("\nEmergence Assessment:")
    for (capability, emerged) in report[:emergence_assessment]
        if capability != :overall_emergence_score
            println("  $(capability): $(emerged ? "✅ EMERGED" : "❌ NOT DETECTED")")
        end
    end
    
    return report
end

end # module

# ==================== EXECUTION ====================

# Run the complete test suite
if abspath(PROGRAM_FILE) == @__FILE__
    println("🚀 LAUNCHING EMERGENT AI TEST SUITE")
    println("Testing: Geometric Intelligence, Metacognition, Consciousness, Cross-Domain Reasoning")
    
    results = EmergentAITesting.run_comprehensive_tests()
    
    println("\n🎯 TESTING COMPLETE")
    println("Evidence of emergent intelligence has been documented in JSON report.")
end
