# test_suite_V3.jl
include("EmergentAIEngine_V3.jl")
using .EmergentAIEngineV3, JSON3, Dates, Statistics, LinearAlgebra

# --- THE "CONCEPTUAL SPACE" EXPERIMENT ---
function run_conceptual_space_experiment(entity)
    println("\n🔬 EXPERIMENT: Testing Generalization in Conceptual Space...")
    println("   The agent was only trained on spatial data.")
    println("   Can it apply the 'closeness' principle to abstract concepts?")

    # Define our concepts as 4D points (value, temp, speed, size)
    conceptual_points_matrix = [
        -0.2 -0.9  0.0 -0.5;  # "Ice Cube"
         0.6  0.1 -0.1 -0.4;  # "Sleeping Cat"
        -0.8  0.2  0.9 -0.8;  # "Bullet"
         0.1  0.0  0.0 -0.6;  # "A Glass of Water" <- The correct answer
         0.4  0.8  0.2  0.6   # "Bonfire"
    ]
    concept_names = ["Ice Cube", "Sleeping Cat", "Bullet", "A Glass of Water", "Bonfire"]

    # Run the instrumented reasoning on this abstract problem
    conceptual_result = EmergentAIEngineV3.instrumented_reasoning(entity, conceptual_points_matrix)
    
    println("\n   RESULTS (Conceptual Problem):")
    println("   - Predicted 'most neutral' concept: '$(concept_names[conceptual_result.result.solution])'")
    println("   - Correct Answer: 'A Glass of Water'")
    println("   - Succeeded at abstract generalization? $(conceptual_result.correct ? "✅ YES" : "❌ NO")")
    println("   - Confidence in its abstract choice: $(round(conceptual_result.result.confidence * 100, digits=2))%")
    
    return (
        name = "Conceptual Generalization",
        success = conceptual_result.correct,
        metrics = Dict(
            :domain => "Abstract/Conceptual",
            :confidence => conceptual_result.result.confidence,
            :time_ns => conceptual_result.execution_time_ns,
            :memory_bytes => conceptual_result.memory_bytes
        )
    )
end

# --- BASELINE: RESOURCE USAGE ON A FAMILIAR SPATIAL PROBLEM ---
function run_spatial_baseline(entity)
    println("\n📊 BASELINE: Measuring resource use on a known spatial problem...")
    
    spatial_points = EmergentAIEngineV3.generate_spatial_points(5)
    spatial_result = EmergentAIEngineV3.instrumented_reasoning(entity, spatial_points)
    
    println("\n   RESULTS (Spatial Problem):")
    println("   - Succeeded at spatial problem? $(spatial_result.correct ? "✅ YES" : "❌ NO")")
    println("   - Confidence in its spatial choice: $(round(spatial_result.result.confidence * 100, digits=2))%")

    return (
        name = "Spatial Baseline",
        success = spatial_result.correct,
        metrics = Dict(
            :domain => "Geometric/Spatial",
            :confidence => spatial_result.result.confidence,
            :time_ns => spatial_result.execution_time_ns,
            :memory_bytes => spatial_result.memory_bytes
        )
    )
end

# --- MAIN EXECUTION AND REPORTING ---
function run_and_report()
    println("="^60)
    println("V3 Test Suite: Probing for True Generalization and Capacity")
    println("="^60)
    
    entity = EmergentAIEngineV3.EmergentGeometricEntity()
    
    baseline_report = run_spatial_baseline(entity)
    conceptual_report = run_conceptual_space_experiment(entity)
    
    # Compare the resource usage
    time_difference = conceptual_report.metrics[:time_ns] - baseline_report.metrics[:time_ns]
    mem_difference = conceptual_report.metrics[:memory_bytes] - baseline_report.metrics[:memory_bytes]
    
    comparison = Dict(
        :time_difference_ns => time_difference,
        :time_factor => conceptual_report.metrics[:time_ns] / baseline_report.metrics[:time_ns],
        :memory_difference_bytes => mem_difference,
        :interpretation => time_difference > 0 ? "Abstract problem required more processing time." : "Abstract problem was processed as efficiently as a spatial one."
    )

    final_report = Dict(
        :suite => "V3 Generalization and Capacity Probe",
        :timestamp => now(),
        :entity_id => entity.id,
        :core_finding => conceptual_report.success ? "The agent successfully generalized its geometric reasoning to an abstract conceptual domain." : "The agent's intelligence is limited to the spatial domain it was trained on.",
        :results => [baseline_report, conceptual_report],
        :resource_comparison => comparison
    )
    
    json_string = JSON3.write(final_report, pretty=true)
    filename = "V3_generalization_report.json"
    open(filename, "w") do f; write(f, json_string); end
    
    println("\n\n" * "="^60)
    println("📈 FINAL REPORT: V3_generalization_report.json")
    println("   >> Core Finding: $(final_report[:core_finding]) <<")
    println("="^60)
    
end

run_and_report()