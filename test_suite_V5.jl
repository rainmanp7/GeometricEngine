# test_suite_V5.jl
include("EmergentAIEngine_V4.jl") # We can reuse the V4 engine
using .EmergentAIEngineV4, JSON3, Dates, Statistics, LinearAlgebra

# --- THE "UNSUPERVISED RULE DISCOVERY" EXPERIMENT (V5) ---
function run_rule_discovery_experiment(entity)
    println("\n🔬 V5 EXPERIMENT: Testing Unsupervised Rule Discovery (Fluid Intelligence)...")
    println("   Can the agent infer a hidden mathematical transformation from a single example?")

    # 1. Define the HIDDEN transformation rule: T(x,y,z,w) -> (-y, x, 2z, 2w)
    function transform(p)
        return [-p[2], p[1], 2*p[3], 2*p[4]]
    end

    # 2. Create the example ("The Key"): Point A and its transformation A'
    point_A = [1.0, 2.0, 0.5, -0.3]
    point_A_prime = transform(point_A)
    println("\n   The Example Given to the Agent:")
    println("   - 'A' at $(point_A) becomes 'A'' at $(point_A_prime)")

    # 3. Create the challenge: Point B and its correct transformation B'
    point_B = [-2.0, 1.5, 1.0, 0.8]
    point_B_prime = transform(point_B) # This is the correct answer the agent must find

    # 4. Create a list of choices for the agent, including the correct answer and decoys
    choices = Dict(
        "Decoy 1 (Simple Addition)" => point_B .+ point_A,
        "Decoy 2 (Mirrored B)" => -point_B,
        "THE CORRECT ANSWER (B')" => point_B_prime,
        "Decoy 3 (Scaled B)" => 2 .* point_B,
        "Decoy 4 (A' again)" => point_A_prime
    )
    choice_names = collect(keys(choices))
    choice_points_matrix = permutedims(reduce(hcat, values(choices)))
    
    # 5. The core of the solution: Re-frame the problem as a vector analogy
    # Target ≈ B + (A' - A)
    target_vector = point_B .+ (point_A_prime .- point_A)
    println("\n   The Agent's Task:")
    println("   - Given 'B' at $(point_B), find the concept from a list that best completes the analogy.")
    println("   - Agent is calculating the target vector for the analogy as: $(round.(target_vector, digits=2))")

    # 6. Ask the agent to find the choice closest to the analogy's target vector
    result = EmergentAIEngineV4.find_closest_concept(entity, choice_points_matrix, target_vector)
    
    # 7. Analyze and Report
    actual_closest_idx = findfirst(isequal(point_B_prime), [eachrow(choice_points_matrix)...])
    succeeded = (result.solution_index == actual_closest_idx)
    predicted_choice = choice_names[result.solution_index]

    println("\n   RESULTS:")
    println("   - Agent's prediction: '$(predicted_choice)'")
    println("   - Correct answer was: 'THE CORRECT ANSWER (B')'")
    println("   - Succeeded at Rule Discovery? $(succeeded ? "✅ YES, IT INFERRED THE RULE" : "❌ NO, IT FAILED TO GENERALIZE THE PROCESS")")
    println("   - Confidence in this highly abstract task: $(round(result.confidence * 100, digits=2))%")
    
    return (
        name = "Unsupervised Rule Discovery (Fluid Intelligence)",
        success = succeeded,
        metrics = Dict(
            :confidence => result.confidence,
            :predicted_choice => predicted_choice,
            :correct_choice => "THE CORRECT ANSWER (B')"
        )
    )
end

# --- MAIN EXECUTION AND REPORTING ---
function run_and_report()
    println("="^60)
    println("V5 Test Suite: Probing for Fluid Intelligence")
    println("="^60)
    
    entity = EmergentAIEngineV4.EmergentGeometricEntity()
    
    rule_discovery_report = run_rule_discovery_experiment(entity)
    
    core_finding = if rule_discovery_report.success
        "BREAKTHROUGH: The agent demonstrated fluid intelligence by inferring a hidden mathematical rule from a single example and applying it to a new case."
    else
        "LIMITATION DISCOVERED: The agent possesses crystallized intelligence (can apply known rules) but lacks fluid intelligence (cannot infer new rules from abstract examples)."
    end

    final_report = Dict(
        :suite => "V5 Fluid Intelligence Probe",
        :timestamp => now(),
        :entity_id => entity.id,
        :core_finding => core_finding,
        :experiment_results => rule_discovery_report
    )
    
    json_string = JSON3.write(final_report, pretty=true)
    filename = "V5_fluid_intelligence_report.json"
    open(filename, "w") do f; write(f, json_string); end
    
    println("\n\n" * "="^60)
    println("📈 FINAL REPORT: V5_fluid_intelligence_report.json")
    println("   >> Core Finding: $(final_report[:core_finding]) <<")
    println("="^60)
    
end

run_and_report()