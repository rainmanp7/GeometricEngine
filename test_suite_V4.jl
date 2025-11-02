# test_suite_V4.jl
include("EmergentAIEngine_V4.jl")
using .EmergentAIEngineV4, JSON3, Dates, Statistics, LinearAlgebra

# --- THE "CONCEPTUAL MIDPOINT" SYNTHESIS EXPERIMENT ---
function run_midpoint_synthesis_experiment(entity)
    println("\n🔬 V4 EXPERIMENT: Testing Conceptual Midpoint Synthesis...")
    println("   Can the agent synthesize a new concept by finding the midpoint between two others?")

    # Define our conceptual space
    concepts = Dict(
        "Ice Cube" => [-0.2, -0.9, 0.0, -0.5],
        "Bonfire" => [0.4, 0.8, 0.2, 0.6],
        "Sleeping Cat" => [0.6, 0.1, -0.1, -0.4],
        "A Lukewarm Mug of Tea" => [0.1, -0.05, 0.0, -0.6], # This is our expected answer
        "Bullet" => [-0.8, 0.2, 0.9, -0.8]
    )
    
    concept_names = collect(keys(concepts))
    concept_points_matrix = permutedims(reduce(hcat, values(concepts)))

    # --- The Synthesis Task ---
    # 1. Define the two extremes
    point_A = concepts["Ice Cube"]
    point_B = concepts["Bonfire"]
    
    # 2. Calculate the mathematical midpoint vector
    midpoint_vector = (point_A .+ point_B) ./ 2
    println("   - Extreme A (Ice Cube): $(round.(point_A, digits=2))")
    println("   - Extreme B (Bonfire):  $(round.(point_B, digits=2))")
    println("   - Calculated Midpoint:  $(round.(midpoint_vector, digits=2))")

    # 3. Ask the agent to find the concept closest to this calculated midpoint
    result = EmergentAIEngineV4.find_closest_concept(entity, concept_points_matrix, midpoint_vector)
    
    # 4. Determine the ground truth
    distances_to_midpoint = [norm(concept_points_matrix[i,:] .- midpoint_vector) for i in 1:size(concept_points_matrix, 1)]
    actual_closest_idx = argmin(distances_to_midpoint)
    
    # --- Analyze and Report ---
    succeeded = (result.solution_index == actual_closest_idx)
    predicted_concept = concept_names[result.solution_index]
    actual_concept = concept_names[actual_closest_idx]

    println("\n   RESULTS:")
    println("   - Agent's prediction for the midpoint concept: '$(predicted_concept)'")
    println("   - Correct answer: '$(actual_concept)'")
    println("   - Succeeded at synthesis? $(succeeded ? "✅ YES" : "❌ NO")")
    println("   - Confidence in its synthesis: $(round(result.confidence * 100, digits=2))%")
    
    return (
        name = "Conceptual Midpoint Synthesis",
        success = succeeded,
        metrics = Dict(
            :confidence => result.confidence,
            :predicted_concept => predicted_concept,
            :actual_concept => actual_concept
        )
    )
end

# --- MAIN EXECUTION AND REPORTING ---
function run_and_report()
    println("="^60)
    println("V4 Test Suite: Probing Conceptual Synthesis")
    println("="^60)
    
    entity = EmergentAIEngineV4.EmergentGeometricEntity()
    
    synthesis_report = run_midpoint_synthesis_experiment(entity)
    
    # A successful synthesis is a landmark result.
    core_finding = if synthesis_report.success
        "The agent successfully performed conceptual synthesis, proving it understands the relational structure of the conceptual space."
    else
        "The agent failed at synthesis. Its reasoning is limited to finding concepts near the origin, not between arbitrary points."
    end

    final_report = Dict(
        :suite => "V4 Conceptual Synthesis Probe",
        :timestamp => now(),
        :entity_id => entity.id,
        :core_finding => core_finding,
        :experiment_results => synthesis_report
    )
    
    json_string = JSON3.write(final_report, pretty=true)
    filename = "V4_synthesis_report.json"
    open(filename, "w") do f; write(f, json_string); end
    
    println("\n\n" * "="^60)
    println("📈 FINAL REPORT: V4_synthesis_report.json")
    println("   >> Core Finding: $(final_report[:core_finding]) <<")
    println("="^60)
    
end

run_and_report()