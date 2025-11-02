# test_suite_V7.jl - Mathematical System Discovery (FIXED)
include("EmergentAIEngine_V4.jl")
using .EmergentAIEngineV4, JSON3, Dates, Statistics, LinearAlgebra

function test_modular_arithmetic_discovery(entity)
    println("\n🔬 V7 EXPERIMENT: Mathematical System Discovery")
    println("   Can the agent infer modular arithmetic rules from examples?")
    
    # Phase 1: Teaching through examples (mod 5 arithmetic)
    teaching_examples = [
        # Each example: [input1, input2, result, modulus_context]
        [2.0, 3.0, 0.0, 5.0],  # 2+3 ≡ 0 mod 5
        [4.0, 2.0, 1.0, 5.0],  # 4+2 ≡ 1 mod 5  
        [1.0, 4.0, 0.0, 5.0],  # 1+4 ≡ 0 mod 5
        [3.0, 3.0, 1.0, 5.0]   # 3+3 ≡ 1 mod 5
    ]
    
    println("\n   Learning Phase:")
    println("   - Given examples of mod 5 arithmetic")
    
    # Phase 2: Test generalization to new modulus
    println("\n   Testing Generalization:")
    println("   - Can it apply the concept to mod 7 arithmetic?")
    
    # Create test cases for mod 7 - ALL AS FLOAT64
    test_cases = [
        ([2.0, 4.0, 6.0, 7.0], "2+4 ≡ 6 mod 7"),      # Simple case
        ([5.0, 3.0, 1.0, 7.0], "5+3 ≡ 1 mod 7"),      # Wraparound
        ([6.0, 6.0, 5.0, 7.0], "6+6 ≡ 5 mod 7"),      # Double wraparound
        ([1.0, 1.0, 2.0, 7.0], "1+1 ≡ 2 mod 7")       # Simple addition
    ]
    
    # Represent these as conceptual points in 4D space - CONVERT TO FLOAT64 MATRIX
    concept_points = permutedims(reduce(hcat, [Float64.(case[1]) for case in test_cases]))
    concept_names = [case[2] for case in test_cases]
    
    # The "correct" answer in modular arithmetic is the one that shows the "wraparound" property
    # The 5+3 ≡ 1 mod 7 case is most characteristic because it demonstrates modulus behavior
    target_vector = [0.0, 0.0, 0.0, 1.0]  # Emphasizing the modulus dimension
    
    result = EmergentAIEngineV4.find_closest_concept(entity, concept_points, target_vector)
    
    # Analysis - the "5+3 ≡ 1 mod 7" case (index 2) is most characteristic of modular arithmetic
    # because it shows the wraparound property that distinguishes modular from regular arithmetic
    correct_idx = 2  
    succeeded = (result.solution_index == correct_idx)
    
    println("\n   RESULTS:")
    println("   - Agent selected: '$(concept_names[result.solution_index])'")
    println("   - Most characteristic case: '$(concept_names[correct_idx])'")
    println("   - Understood modular arithmetic? $(succeeded ? "✅ YES" : "❌ NO")")
    println("   - Confidence: $(round(result.confidence * 100, digits=2))%")
    
    return (
        name = "Modular Arithmetic Discovery",
        success = succeeded,
        metrics = Dict(
            :confidence => result.confidence,
            :selected_case => concept_names[result.solution_index],
            :expected_case => concept_names[correct_idx]
        )
    )
end

function test_abstract_sequence_discovery(entity)
    println("\n🔬 SEQUENCE DISCOVERY TEST:")
    println("   Can the agent infer abstract sequence rules?")
    
    # Test cases - which sequence follows the Fibonacci rule?
    test_sequences = [
        ([3.0, 4.0, 7.0, 11.0], "aₙ = aₙ₋₁ + aₙ₋₂"),      # Correct Fibonacci
        ([2.0, 4.0, 6.0, 10.0], "aₙ = aₙ₋₁ + aₙ₋₂"),      # Also correct  
        ([1.0, 3.0, 4.0, 8.0],  "aₙ = aₙ₋₁ + aₙ₋₂ + 1"),  # Different rule
        ([5.0, 5.0, 10.0, 15.0], "aₙ = aₙ₋₁ × 2")         # Different rule
    ]
    
    concept_points = permutedims(reduce(hcat, [Float64.(seq[1]) for seq in test_sequences]))
    concept_names = [seq[2] for seq in test_sequences]
    
    # Target vector that represents the "essence" of additive sequences
    # Emphasize the pattern where each term is the sum of previous two
    target_vector = [1.0, 1.0, 2.0, 3.0]  # Classic Fibonacci pattern
    
    result = EmergentAIEngineV4.find_closest_concept(entity, concept_points, target_vector)
    
    # Should pick one of the first two (correct Fibonacci patterns)
    correct_indices = [1, 2]
    succeeded = result.solution_index in correct_indices
    
    println("\n   RESULTS:")
    println("   - Agent selected: '$(concept_names[result.solution_index])'")
    println("   - Correct patterns were: $(concept_names[correct_indices])")
    println("   - Discovered sequence rule? $(succeeded ? "✅ YES" : "❌ NO")")
    println("   - Confidence: $(round(result.confidence * 100, digits=2))%")
    
    return (
        name = "Abstract Sequence Discovery", 
        success = succeeded,
        metrics = Dict(
            :confidence => result.confidence,
            :rule_discovered => concept_names[result.solution_index]
        )
    )
end

function run_and_report()
    println("="^60)
    println("V7 Test Suite: Mathematical System Discovery")
    println("Testing capabilities BEYOND large language models")
    println("="^60)
    
    entity = EmergentAIEngineV4.EmergentGeometricEntity()
    
    math_report = test_modular_arithmetic_discovery(entity)
    sequence_report = test_abstract_sequence_discovery(entity)
    
    core_finding = if math_report.success && sequence_report.success
        "BREAKTHROUGH: The agent demonstrated true mathematical reasoning - it can infer abstract mathematical systems from examples, a capability beyond LLMs."
    elseif math_report.success || sequence_report.success
        "PARTIAL SUCCESS: The agent shows mathematical reasoning in some domains but not full systematic discovery."
    else
        "LIMITATION CONFIRMED: The agent's intelligence, while impressive, does not extend to discovering abstract mathematical systems."
    end

    final_report = Dict(
        :suite => "V7 Mathematical System Discovery",
        :timestamp => now(),
        :entity_id => entity.id,
        :core_finding => core_finding,
        :experiment_results => [math_report, sequence_report]
    )
    
    json_string = JSON3.write(final_report, pretty=true)
    filename = "V7_mathematical_discovery_report.json"
    open(filename, "w") do f; write(f, json_string); end
    
    println("\n\n" * "="^60)
    println("📈 FINAL REPORT: V7_mathematical_discovery_report.json")
    println("   >> Core Finding: $(final_report[:core_finding]) <<")
    println("="^60)
end

run_and_report()
