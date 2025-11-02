# test_suite_V7.jl - Mathematical System Discovery (Corrected)
include("EmergentAIEngine_V4.jl")
using .EmergentAIEngineV4, JSON3, Dates, Statistics, LinearAlgebra

function test_modular_arithmetic_discovery(entity)
    println("\n🔬 V7 EXPERIMENT: Mathematical System Discovery")
    println("   Can the agent learn modular arithmetic from examples AND classify new cases?")
    
    # Phase 1: Teaching through examples (mod 5 arithmetic)
    # Format: [operand1, operand2, result, modulus]
    teaching_examples = [
        [2, 3, 0, 5],  # 2+3 = 5 ≡ 0 mod 5 ✓
        [4, 2, 1, 5],  # 4+2 = 6 ≡ 1 mod 5 ✓
        [1, 4, 0, 5],  # 1+4 = 5 ≡ 0 mod 5 ✓
        [3, 3, 1, 5],  # 3+3 = 6 ≡ 1 mod 5 ✓
        [4, 4, 3, 5]   # 4+4 = 8 ≡ 3 mod 5 ✓
    ]
    
    println("\n   📚 Learning Phase (Teaching valid mod 5 equations):")
    for ex in teaching_examples
        sum_val = ex[1] + ex[2]
        println("     $(ex[1]) + $(ex[2]) = $sum_val ≡ $(ex[3]) (mod $(ex[4])) ✓")
    end
    
    # Calculate pattern signature from valid examples
    teaching_matrix = Float64.(permutedims(reduce(hcat, teaching_examples)))
    pattern_signature = vec(mean(teaching_matrix, dims=1))
    
    println("\n   🧪 Testing Phase 1: PATTERN RECOGNITION (new modulus)")
    println("   - Can it recognize the pattern in mod 7 arithmetic?")
    
    # Test cases for mod 7 - mix of correct and incorrect
    pattern_test_cases = [
        ([2, 4, 6, 7], "2+4=6 ≡ 6 mod 7", true),      # CORRECT
        ([5, 3, 1, 7], "5+3=8 ≡ 1 mod 7", true),      # CORRECT
        ([6, 6, 5, 7], "6+6=12 ≡ 5 mod 7", true),     # CORRECT
        ([4, 5, 2, 7], "4+5=9 ≡ 2 mod 7", true),      # CORRECT
        ([2, 3, 6, 7], "2+3=5 ≡ 6 mod 7", false),     # WRONG (should be 5)
        ([5, 3, 3, 7], "5+3=8 ≡ 3 mod 7", false),     # WRONG (should be 1)
        ([6, 6, 0, 7], "6+6=12 ≡ 0 mod 7", false)     # WRONG (should be 5)
    ]
    
    # Test pattern recognition
    concept_points = Float64.(permutedims(reduce(hcat, [case[1] for case in pattern_test_cases])))
    concept_names = [case[2] for case in pattern_test_cases]
    is_correct = [case[3] for case in pattern_test_cases]
    
    result1 = EmergentAIEngineV4.find_closest_concept(entity, concept_points, pattern_signature)
    pattern_success = is_correct[result1.solution_index]
    
    println("\n   PATTERN RECOGNITION RESULTS:")
    println("   - Agent selected: '$(concept_names[result1.solution_index])'")
    println("   - This equation is: $(pattern_success ? "✓ CORRECT" : "✗ INCORRECT")")
    println("   - Pattern recognition: $(pattern_success ? "✅ SUCCESS" : "❌ FAILED")")
    println("   - Confidence: $(round(result1.confidence * 100, digits=2))%")
    
    println("\n   🧪 Testing Phase 2: CLASSIFICATION (discriminating valid from invalid)")
    println("   - Can it identify which equations are valid?")
    
    # Create a test where we explicitly ask: which of these follows the rule?
    # Mix correct and incorrect examples in a new modulus (mod 11)
    classification_test_cases = [
        ([7, 8, 4, 11], "7+8=15 ≡ 4 mod 11", true),    # CORRECT
        ([9, 5, 3, 11], "9+5=14 ≡ 3 mod 11", true),    # CORRECT
        ([6, 7, 2, 11], "6+7=13 ≡ 2 mod 11", true),    # CORRECT
        ([7, 8, 5, 11], "7+8=15 ≡ 5 mod 11", false),   # WRONG (should be 4)
        ([9, 5, 4, 11], "9+5=14 ≡ 4 mod 11", false),   # WRONG (should be 3)
        ([10, 10, 9, 11], "10+10=20 ≡ 9 mod 11", false) # WRONG (should be 9... wait this is correct!)
    ]
    
    # Fix the last one
    classification_test_cases[6] = ([10, 10, 8, 11], "10+10=20 ≡ 8 mod 11", false)  # WRONG (should be 9)
    
    concept_points2 = Float64.(permutedims(reduce(hcat, [case[1] for case in classification_test_cases])))
    concept_names2 = [case[2] for case in classification_test_cases]
    is_correct2 = [case[3] for case in classification_test_cases]
    
    result2 = EmergentAIEngineV4.find_closest_concept(entity, concept_points2, pattern_signature)
    classification_success = is_correct2[result2.solution_index]
    
    println("\n   CLASSIFICATION RESULTS:")
    println("   - Agent selected: '$(concept_names2[result2.solution_index])'")
    println("   - This equation is: $(classification_success ? "✓ CORRECT" : "✗ INCORRECT")")
    println("   - Classification ability: $(classification_success ? "✅ SUCCESS" : "❌ FAILED")")
    println("   - Confidence: $(round(result2.confidence * 100, digits=2))%")
    
    overall_success = pattern_success && classification_success
    
    println("\n   📊 OVERALL MODULAR ARITHMETIC TEST:")
    println("   - Pattern Recognition: $(pattern_success ? "✅" : "❌")")
    println("   - Classification: $(classification_success ? "✅" : "❌")")
    println("   - Combined Success: $(overall_success ? "✅ PASSED" : "❌ FAILED")")
    
    return (
        name = "Modular Arithmetic Discovery",
        success = overall_success,
        pattern_recognition = pattern_success,
        classification = classification_success,
        metrics = Dict(
            :pattern_confidence => result1.confidence,
            :classification_confidence => result2.confidence,
            :pattern_selected => concept_names[result1.solution_index],
            :classification_selected => concept_names2[result2.solution_index]
        )
    )
end

function test_abstract_sequence_discovery(entity)
    println("\n🔬 SEQUENCE DISCOVERY TEST:")
    println("   Can the agent learn Fibonacci rules AND classify new sequences?")
    
    # Teach through examples of Fibonacci-like sequences
    sequence_examples = [
        [1, 1, 2, 3],    # 1+1=2, 1+2=3 ✓
        [2, 3, 5, 8],    # 2+3=5, 3+5=8 ✓
        [5, 7, 12, 19],  # 5+7=12, 7+12=19 ✓
        [0, 1, 1, 2]     # 0+1=1, 1+1=2 ✓
    ]
    
    println("\n   📚 Learning Phase (Teaching Fibonacci pattern):")
    for seq in sequence_examples
        println("     $(seq) → Rule: aₙ = aₙ₋₁ + aₙ₋₂")
        println("       Check: $(seq[1])+$(seq[2])=$(seq[3]) ✓, $(seq[2])+$(seq[3])=$(seq[4]) ✓")
    end
    
    teaching_matrix = Float64.(permutedims(reduce(hcat, sequence_examples)))
    pattern_signature = vec(mean(teaching_matrix, dims=1))
    
    println("\n   🧪 Testing Phase 1: PATTERN RECOGNITION")
    println("   - Can it recognize Fibonacci pattern in new sequences?")
    
    # Test sequences - mix of correct Fibonacci and other patterns
    pattern_test_sequences = [
        ([3, 4, 7, 11], "Fibonacci pattern", true),           # 3+4=7, 4+7=11 ✓
        ([8, 5, 13, 18], "Fibonacci pattern", true),          # 8+5=13, 5+13=18 ✓
        ([10, 15, 25, 40], "Fibonacci pattern", true),        # 10+15=25, 15+25=40 ✓
        ([1, 3, 4, 8], "Non-Fibonacci", false),               # 1+3=4 ✓, but 3+4=7≠8 ✗
        ([2, 4, 8, 16], "Exponential (×2)", false),           # 2×2=4, 4×2=8, different rule
        ([5, 10, 15, 25], "Different rule", false)            # 5+10=15 ✓, but 10+15=25 ✗
    ]
    
    concept_points = Float64.(permutedims(reduce(hcat, [seq[1] for seq in pattern_test_sequences])))
    concept_names = [seq[2] for seq in pattern_test_sequences]
    is_correct = [seq[3] for seq in pattern_test_sequences]
    
    result1 = EmergentAIEngineV4.find_closest_concept(entity, concept_points, pattern_signature)
    pattern_success = is_correct[result1.solution_index]
    
    println("\n   PATTERN RECOGNITION RESULTS:")
    println("   - Agent selected: sequence $(result1.solution_index) ($(concept_names[result1.solution_index]))")
    println("   - Follows Fibonacci rule: $(pattern_success ? "✓ YES" : "✗ NO")")
    println("   - Pattern recognition: $(pattern_success ? "✅ SUCCESS" : "❌ FAILED")")
    
    println("\n   🧪 Testing Phase 2: CLASSIFICATION")
    println("   - Can it discriminate Fibonacci from non-Fibonacci?")
    
    classification_test_sequences = [
        ([6, 9, 15, 24], "Valid Fibonacci", true),            # 6+9=15, 9+15=24 ✓
        ([4, 7, 11, 18], "Valid Fibonacci", true),            # 4+7=11, 7+11=18 ✓
        ([2, 5, 10, 15], "Arithmetic (+5)", false),           # 2+5=7≠10 ✗
        ([3, 6, 9, 15], "Different pattern", false),          # 3+6=9 ✓, but 6+9=15 ✗
        ([1, 2, 4, 8], "Powers of 2", false)                  # 1+2=3≠4 ✗
    ]
    
    concept_points2 = Float64.(permutedims(reduce(hcat, [seq[1] for seq in classification_test_sequences])))
    concept_names2 = [seq[2] for seq in classification_test_sequences]
    is_correct2 = [seq[3] for seq in classification_test_sequences]
    
    result2 = EmergentAIEngineV4.find_closest_concept(entity, concept_points2, pattern_signature)
    classification_success = is_correct2[result2.solution_index]
    
    println("\n   CLASSIFICATION RESULTS:")
    println("   - Agent selected: sequence $(result2.solution_index) ($(concept_names2[result2.solution_index]))")
    println("   - Follows Fibonacci rule: $(classification_success ? "✓ YES" : "✗ NO")")
    println("   - Classification ability: $(classification_success ? "✅ SUCCESS" : "❌ FAILED")")
    
    overall_success = pattern_success && classification_success
    
    println("\n   📊 OVERALL SEQUENCE DISCOVERY TEST:")
    println("   - Pattern Recognition: $(pattern_success ? "✅" : "❌")")
    println("   - Classification: $(classification_success ? "✅" : "❌")")
    println("   - Combined Success: $(overall_success ? "✅ PASSED" : "❌ FAILED")")
    
    return (
        name = "Abstract Sequence Discovery", 
        success = overall_success,
        pattern_recognition = pattern_success,
        classification = classification_success,
        metrics = Dict(
            :pattern_confidence => result1.confidence,
            :classification_confidence => result2.confidence,
            :pattern_selected => result1.solution_index,
            :classification_selected => result2.solution_index
        )
    )
end

function run_and_report()
    println("="^60)
    println("V7 Test Suite: Mathematical System Discovery")
    println("Testing DUAL capabilities:")
    println("  1. Pattern Recognition - apply learned rules to new cases")
    println("  2. Classification - distinguish valid from invalid")
    println("="^60)
    
    entity = EmergentAIEngineV4.EmergentGeometricEntity()
    
    math_report = test_modular_arithmetic_discovery(entity)
    sequence_report = test_abstract_sequence_discovery(entity)
    
    # Detailed assessment
    all_passed = math_report.success && sequence_report.success
    pattern_recognition_ok = math_report.pattern_recognition && sequence_report.pattern_recognition
    classification_ok = math_report.classification && sequence_report.classification
    
    core_finding = if all_passed
        "BREAKTHROUGH: The agent demonstrated BOTH pattern recognition AND classification abilities - it can learn mathematical rules from examples and discriminate valid from invalid applications."
    elseif pattern_recognition_ok && !classification_ok
        "PARTIAL SUCCESS: The agent can recognize patterns in new contexts but struggles to classify valid vs invalid examples."
    elseif !pattern_recognition_ok && classification_ok
        "PARTIAL SUCCESS: The agent can classify valid examples but struggles to recognize the pattern in new contexts."
    elseif pattern_recognition_ok || classification_ok
        "LIMITED SUCCESS: The agent shows some mathematical reasoning ability in specific domains."
    else
        "NEGATIVE RESULT: The agent did not successfully demonstrate pattern recognition or classification abilities."
    end

    final_report = Dict(
        :suite => "V7 Mathematical System Discovery (Dual Testing)",
        :timestamp => now(),
        :entity_id => entity.id,
        :core_finding => core_finding,
        :summary => Dict(
            :overall_success => all_passed,
            :pattern_recognition_success => pattern_recognition_ok,
            :classification_success => classification_ok
        ),
        :experiment_results => [
            Dict(
                :name => math_report.name, 
                :success => math_report.success,
                :pattern_recognition => math_report.pattern_recognition,
                :classification => math_report.classification,
                :metrics => math_report.metrics
            ),
            Dict(
                :name => sequence_report.name, 
                :success => sequence_report.success,
                :pattern_recognition => sequence_report.pattern_recognition,
                :classification => sequence_report.classification,
                :metrics => sequence_report.metrics
            )
        ]
    )
    
    json_string = JSON3.write(final_report, pretty=true)
    filename = "V7_mathematical_discovery_report_$(Dates.format(now(), "yyyymmdd_HHMMSS")).json"
    open(filename, "w") do f
        write(f, json_string)
    end
    
    println("\n\n" * "="^60)
    println("📈 FINAL REPORT: $filename")
    println("="^60)
    println(">> Core Finding:")
    println("   $(final_report[:core_finding])")
    println()
    println(">> Capabilities Demonstrated:")
    println("   Pattern Recognition: $(pattern_recognition_ok ? "✅ PASS" : "❌ FAIL")")
    println("   Classification: $(classification_ok ? "✅ PASS" : "❌ FAIL")")
    println("   Overall: $(all_passed ? "✅ PASS" : "❌ FAIL")")
    println("="^60)
    
    return final_report
end

run_and_report()
