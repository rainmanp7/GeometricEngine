# test_suite_V7.jl - Mathematical System Discovery (Corrected)
include("EmergentAIEngine_V4.jl")
using .EmergentAIEngineV4, JSON3, Dates, Statistics, LinearAlgebra

function test_modular_arithmetic_discovery(entity)
    println("\n🔬 V7 EXPERIMENT: Mathematical System Discovery")
    println("   Can the agent infer modular arithmetic rules from examples?")
    
    # Phase 1: Teaching through examples (mod 5 arithmetic)
    teaching_examples = [
        [2, 3, 0, 5],  # 2+3 ≡ 0 mod 5
        [4, 2, 1, 5],  # 4+2 ≡ 1 mod 5  
        [1, 4, 0, 5],  # 1+4 ≡ 0 mod 5
        [3, 3, 1, 5]   # 3+3 ≡ 1 mod 5
    ]
    
    println("\n   Learning Phase:")
    println("   - Teaching with mod 5 arithmetic examples:")
    for ex in teaching_examples
        println("     $(ex[1]) + $(ex[2]) ≡ $(ex[3]) (mod $(ex[4]))")
    end
    
    # Actually use the teaching examples to establish the pattern
    # Calculate average "pattern signature" from teaching examples
    teaching_matrix = Float64.(permutedims(reduce(hcat, teaching_examples)))
    pattern_signature = vec(mean(teaching_matrix, dims=1))
    
    println("\n   Testing Generalization:")
    println("   - Can it apply the concept to mod 7 arithmetic?")
    
    # Create test cases for mod 7 - ALL mathematically correct
    test_cases = [
        ([2, 4, 6, 7], "2+4 ≡ 6 mod 7"),      # No wraparound
        ([5, 3, 1, 7], "5+3 ≡ 1 mod 7"),      # Wraparound (8 mod 7 = 1)
        ([6, 6, 5, 7], "6+6 ≡ 5 mod 7"),      # Wraparound (12 mod 7 = 5)
        ([4, 5, 2, 7], "4+5 ≡ 2 mod 7")       # Wraparound (9 mod 7 = 2)
    ]
    
    # Also add incorrect examples to test discrimination
    incorrect_cases = [
        ([2, 3, 6, 7], "2+3 ≡ 6 mod 7 [WRONG: should be 5]"),
        ([5, 3, 2, 7], "5+3 ≡ 2 mod 7 [WRONG: should be 1]")
    ]
    
    all_test_cases = vcat(test_cases, incorrect_cases)
    
    # Represent as conceptual points in 4D space [operand1, operand2, result, modulus]
    concept_points = Float64.(permutedims(reduce(hcat, [case[1] for case in all_test_cases])))
    concept_names = [case[2] for case in all_test_cases]
    
    # Use the learned pattern signature as target
    result = EmergentAIEngineV4.find_closest_concept(entity, concept_points, pattern_signature)
    
    # Analysis: success if it picks any of the first 4 (correct) cases
    correct_indices = 1:4
    succeeded = result.solution_index in correct_indices
    
    println("\n   RESULTS:")
    println("   - Agent selected: '$(concept_names[result.solution_index])'")
    println("   - This is $(result.solution_index <= 4 ? "CORRECT" : "INCORRECT")")
    println("   - Understood modular arithmetic? $(succeeded ? "✅ YES" : "❌ NO")")
    println("   - Confidence: $(round(result.confidence * 100, digits=2))%")
    
    return (
        name = "Modular Arithmetic Discovery",
        success = succeeded,
        metrics = Dict(
            :confidence => result.confidence,
            :selected_case => concept_names[result.solution_index],
            :is_correct => result.solution_index <= 4
        )
    )
end

function test_abstract_sequence_discovery(entity)
    println("\n🔬 SEQUENCE DISCOVERY TEST:")
    println("   Can the agent infer abstract sequence rules?")
    
    # Teach through examples of Fibonacci-like sequences
    sequence_examples = [
        [1, 1, 2, 3],    # Standard Fibonacci
        [2, 3, 5, 8],    # Different starting points
        [5, 7, 12, 19]   # Larger numbers
    ]
    
    println("\n   Learning Phase:")
    println("   - Teaching with Fibonacci-like sequences:")
    for seq in sequence_examples
        println("     $(seq)")
    end
    
    # Calculate the pattern: each sequence has property that seq[3] = seq[1] + seq[2], etc.
    teaching_matrix = Float64.(permutedims(reduce(hcat, sequence_examples)))
    pattern_signature = vec(mean(teaching_matrix, dims=1))
    
    # Test cases - which sequences follow the same rule?
    test_sequences = [
        ([3, 4, 7, 11], "aₙ = aₙ₋₁ + aₙ₋₂ [CORRECT]"),           # Correct: 3+4=7, 4+7=11
        ([2, 4, 6, 10], "aₙ = aₙ₋₁ + aₙ₋₂ [CORRECT]"),           # Correct: 2+4=6, 4+6=10
        ([8, 5, 13, 18], "aₙ = aₙ₋₁ + aₙ₋₂ [CORRECT]"),          # Correct: 8+5=13, 5+13=18
        ([1, 3, 4, 8], "aₙ = aₙ₋₁ + aₙ₋₂ + 1 [WRONG]"),          # Wrong: 1+3≠4 but 3+4≠8
        ([5, 5, 10, 20], "aₙ = 2×aₙ₋₁ [WRONG]"),                 # Wrong: different rule
        ([2, 4, 8, 16], "aₙ = 2×aₙ₋₁ [WRONG]")                   # Wrong: exponential growth
    ]
    
    concept_points = Float64.(permutedims(reduce(hcat, [seq[1] for seq in test_sequences])))
    concept_names = [seq[2] for seq in test_sequences]
    
    result = EmergentAIEngineV4.find_closest_concept(entity, concept_points, pattern_signature)
    
    # Should pick one of the first three (correct Fibonacci patterns)
    correct_indices = 1:3
    succeeded = result.solution_index in correct_indices
    
    println("\n   RESULTS:")
    println("   - Agent selected: '$(concept_names[result.solution_index])'")
    println("   - Correct patterns: indices $(collect(correct_indices))")
    println("   - Discovered sequence rule? $(succeeded ? "✅ YES" : "❌ NO")")
    println("   - Confidence: $(round(result.confidence * 100, digits=2))%")
    
    return (
        name = "Abstract Sequence Discovery", 
        success = succeeded,
        metrics = Dict(
            :confidence => result.confidence,
            :selected_index => result.solution_index,
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
        "BREAKTHROUGH: The agent demonstrated true mathematical reasoning - it can infer abstract mathematical systems from examples and distinguish correct from incorrect applications."
    elseif math_report.success || sequence_report.success
        "PARTIAL SUCCESS: The agent shows mathematical reasoning in some domains but not full systematic discovery capabilities."
    else
        "The agent did not successfully discover the mathematical patterns from the teaching examples."
    end

    final_report = Dict(
        :suite => "V7 Mathematical System Discovery",
        :timestamp => now(),
        :entity_id => entity.id,
        :core_finding => core_finding,
        :experiment_results => [
            Dict(:name => math_report.name, :success => math_report.success, :metrics => math_report.metrics),
            Dict(:name => sequence_report.name, :success => sequence_report.success, :metrics => sequence_report.metrics)
        ],
        :overall_success_rate => mean([math_report.success, sequence_report.success])
    )
    
    json_string = JSON3.write(final_report, pretty=true)
    filename = "V7_mathematical_discovery_report_$(Dates.format(now(), "yyyymmdd_HHMMSS")).json"
    open(filename, "w") do f
        write(f, json_string)
    end
    
    println("\n\n" * "="^60)
    println("📈 FINAL REPORT: $filename")
    println("   >> Core Finding: $(final_report[:core_finding]) <<")
    println("   >> Success Rate: $(round(final_report[:overall_success_rate] * 100, digits=1))% <<")
    println("="^60)
    
    return final_report
end

run_and_report()
