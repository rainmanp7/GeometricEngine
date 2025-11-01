# production_proof_suite.jl
#
# REVISED VERSION: This script runs a "proof suite" for the new, ideal
# ProductionGeometricEngine. It verifies that the engine correctly and instantly
# identifies the closest point to the origin.

using JSON3, Random, Statistics, Printf, LinearAlgebra

# Include the new, perfect engine.
include("ProductionGeometricEngine.jl")
using .ProductionGeometricEngine

# --- Parameters for the test problems ---
const NUM_POINTS = 10
const DIMENSIONS = 4
const NUM_TESTS  = 5 # The number of random problems to verify.

"""
    one_run(seed::Int)

Performs a single verification run: generates a problem, runs the engine,
and checks if the engine's prediction matches the ground truth.
"""
function one_run(seed::Int)
    rng = MersenneTwister(seed)

    # 1. Generate a test problem with a known structure.
    points = make_problem(NUM_POINTS, DIMENSIONS; rng=rng)

    # 2. Use the ideal engine to get a prediction.
    result = find_closest_point(points)

    # 3. Independently calculate the ground truth to verify against.
    ground_truth_idx = argmin([norm(view(points, i, :)) for i in 1:NUM_POINTS])
    
    is_correct = result.prediction == ground_truth_idx

    # Return a dictionary with the results of this single test.
    return Dict(
        "seed"               => seed,
        "prediction_correct" => is_correct,
        "predicted_index"    => result.prediction,
        "ground_truth_index" => ground_truth_idx,
        "engine_confidence"  => round(result.probabilities[result.prediction], digits=4),
        "all_distances"      => round.(result.distances, digits=4)
    )
end

"""
    run_suite()

Executes the main proof suite, running multiple tests and summarizing the results.
"""
function run_suite()
    println("Running Production Proof Suite for the Ideal Geometric Engine…")
    
    results = [one_run(i) for i in 1:NUM_TESTS]

    # Since the engine is perfect, all tests should pass.
    all_tests_passed = all(r["prediction_correct"] for r in results)
    
    summary = Dict(
        "total_tests"          => NUM_TESTS,
        "passed_tests"         => count(r["prediction_correct"] for r in results),
        "all_tests_passed"     => all_tests_passed,
        "average_confidence"   => round(mean(r["engine_confidence"] for r in results), digits=4)
    )
    
    # The JSON output now reflects verification, not learning.
    full = Dict(
        "summary"             => summary,
        "runs"                => results,
        "emergent_properties" => Dict(
            "geometric_reasoning_verified" => all_tests_passed,
            "description" => "The engine correctly implements the ideal mathematical formula for finding the closest point to the origin."
        )
    )

    open("proof_results.json", "w") do f
        JSON3.write(f, full, indent=4)
    end
    
    println("\n--- Verification Results ---")
    println(JSON3.write(full))
    
    if all_tests_passed
        println("\nConclusion: All tests passed. The Ideal Geometric Engine is verified as correct.")
    else
        error("FATAL: One or more verification tests failed!")
    end
end

# Execute the suite.
run_suite()
