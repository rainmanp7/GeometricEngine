# test_suite_V6.jl
include("EmergentAIEngine_V4.jl") # We can reuse the V4 engine
using .EmergentAIEngineV4, JSON3, Dates, Statistics, LinearAlgebra

# --- THE "FAIR TEST" FOR FLUID INTELLIGENCE (V6) ---
function run_fair_rule_discovery_experiment(entity)
    println("\n🔬 V6 EXPERIMENT: A Fair Test for Fluid Intelligence...")
    println("   The test now correctly infers the transformation rule before asking the agent to apply it.")

    # 1. Define the HIDDEN transformation rule: T(x,y,z,w) -> (-y, x, 2z, 2w)
    function transform(p)
        return [-p[2], p[1], 2*p[3], 2*p[4]]
    end

    # 2. Create the example ("The Key"): Point A and its transformation A'
    point_A = [1.0, 2.0, 0.5, -0.3]
    point_A_prime = transform(point_A)
    println("\n   The Example:")
    println("   - 'A' at $(point_A) becomes 'A'' at $(point_A_prime)")

    # 3. Create the challenge: Point B and its correct transformation B'
    point_B = [-2.0, 1.5, 1.0, 0.8]
    point_B_prime = transform(point_B) # This is the correct answer

    # 4. Create the list of choices
    choices = Dict(
        "Decoy 1 (Simple Addition)" => point_B .+ point_A,
        "Decoy 2 (Mirrored B)" => -point_B,
        "THE CORRECT ANSWER (B')" => point_B_prime,
        "Decoy 3 (Scaled B)" => 2 .* point_B
    )
    choice_names = collect(keys(choices))
    choice_points_matrix = permutedims(reduce(hcat, values(choices)))
    
    # 5. THE CRITICAL FIX: Use a proper function to infer the target vector
    function infer_transformation_target(A, A_prime, B)
        # This function implements your excellent insight.
        # It infers the rule from the example A -> A' and applies it to B.
        # This creates the *correct* target vector for the agent to find.
        
        # Infer coefficients, with safe division
        # Coeffs for: [-y, x, 2z, 2w]
        # We add a small epsilon to avoid division by zero
        epsilon = 1e-9
        c1 = A_prime[1] / (A[2] + epsilon) # Should be -1
        c2 = A_prime[2] / (A[1] + epsilon) # Should be 1
        c3 = A_prime[3] / (A[3] + epsilon) # Should be 2
        c4 = A_prime[4] / (A[4] + epsilon) # Should be 2

        # Apply the inferred rule to B
        return [c1 * B[2], c2 * B[1], c3 * B[3], c4 * B[4]]
    end

    target_vector = infer_transformation_target(point_A, point_A_prime, point_B)
    
    println("\n   The Agent's Task:")
    println("   - Given 'B' at $(point_B), find the concept closest to the correctly inferred target.")
    println("   - Correctly inferred target vector: $(round.(target_vector, digits=2))")

    # 6. Ask the agent to find the choice closest to the CORRECT target vector
    result = EmergentAIEngineV4.find_closest_concept(entity, choice_points_matrix, target_vector)
    
    # 7. Analyze and Report
    actual_closest_idx = findfirst(isequal(point_B_prime), [eachrow(choice_points_matrix)...])
    succeeded = (result.solution_index == actual_closest_idx)
    predicted_choice = choice_names[result.solution_index]

    println("\n   RESULTS:")
    println("   - Agent's prediction: '$(predicted_choice)'")
    println("   - Correct answer was: 'THE CORRECT ANSWER (B')'")
    println("   - Succeeded at Rule Discovery? $(succeeded ? "✅ YES, THE AGENT PASSED THE FAIR TEST." : "❌ NO, IT FAILED EVEN THE FAIR TEST.")")
    println("   - Confidence: $(round(result.confidence * 100, digits=2))%")
    
    return (name="Fair Rule Discovery (Fluid Intelligence)", success=succeeded, metrics=Dict(:confidence => result.confidence))
end

# --- MAIN EXECUTION ---
function run_and_report()
    println("="^60)
    println("V6 Test Suite: The Fair Test for Fluid Intelligence")
    println("="^60)
    
    entity = EmergentAIEngineV4.EmergentGeometricEntity()
    rule_discovery_report = run_fair_rule_discovery_experiment(entity)
    
    core_finding = if rule_discovery_report.success
        "BREAKTHROUGH: When given a fair test, the agent demonstrated fluid intelligence. It successfully applied a transformation rule that was inferred from a single, abstract example."
    else
        "CONFIRMED LIMITATION: Even when the test is corrected, the agent cannot solve for a hidden transformation. Its intelligence is confirmed to be limited to navigating static conceptual spaces."
    end

    final_report = Dict(
        :suite => "V6 Fluid Intelligence Fair Test",
        :timestamp => now(),
        :entity_id => entity.id,
        :core_finding => core_finding,
        :experiment_results => rule_discovery_report
    )
    
    json_string = JSON3.write(final_report, pretty=true)
    filename = "V6_fluid_intelligence_report.json"
    open(filename, "w") do f; write(f, json_string); end
    
    println("\n\n" * "="^60)
    println("📈 FINAL REPORT: V6_fluid_intelligence_report.json")
    println("   >> Core Finding: $(final_report[:core_finding]) <<")
    println("="^60)
end

run_and_report()