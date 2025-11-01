"""
🔬 DEBUGGING SCRIPT for `geometric_learn!` in ProtectedGeometricEngine.jl

Purpose:
This script isolates the core learning function (`geometric_learn!`) and tests
its ability to learn the simplest possible task: overfitting to a single,
unchanging data point.

Expected Behavior (if working):
The 'Accuracy' value printed in the loop should steadily climb towards 1.0.
The final test should show that the model correctly predicts the answer.

Likely Behavior (if broken):
The 'Accuracy' will fluctuate randomly, never showing a consistent upward trend.
The final test will likely be incorrect.
"""

using LinearAlgebra # Needed for the `norm` function used internally

# This assumes ProtectedGeometricEngine.jl is in the same directory.
try
    include("ProtectedGeometricEngine.jl")
catch e
    if e isa SystemError
        println("❌ ERROR: Could not find 'ProtectedGeometricEngine.jl'.")
        println("   Please ensure this file is in the same directory.")
        exit(1)
    else
        rethrow(e)
    end
end

function run_learning_debug()
    println("--- STARTING LEARNING DEBUGGER ---")
    println("="^50)

    # 1. Set up the core and a single, fixed problem
    println("1. Creating a 4D Geometric Consciousness Core...")
    core = ProtectedGeometricEngine.GeometricConsciousnessCore(4)
    
    println("2. Generating a SINGLE, FIXED geometric problem...")
    points, true_ans = ProtectedGeometricEngine.generate_geometric_problem(core)
    println("   The correct answer (index) for this problem is: $true_ans")
    println("   The problem has $(size(points, 1)) points, each with $(size(points, 2)) dimensions.")

    # 2. Run the learning loop on that single problem repeatedly
    println("\n3. Starting training loop on this single problem...")
    num_steps = 100
    for step in 1:num_steps
        # The core of the test: attempt to learn the same thing over and over
        accuracy = ProtectedGeometricEngine.geometric_learn!(core, points, true_ans)
        
        # Print the accuracy at each step. We want to see this go up!
        println("   Step $(rpad(step, 3)): Accuracy = $(accuracy)")

        # Optional: Stop early if it learns perfectly
        if accuracy > 0.999
            println("   🎉 Model has successfully learned the problem!")
            break
        end
    end
    println("\n   Training loop finished.")

    # 3. After training, ask the model to solve the problem one last time
    println("\n4. Final test: Solving the problem after training...")
    solution, confidence, analysis = ProtectedGeometricEngine.solve_geometric_problem(core, points)

    println("   Model's final prediction: $(analysis["prediction"])")
    println("   Actual correct answer:    $(analysis["actual"])")
    println("   Was the model correct?    $(analysis["correct"] ? "✅ YES" : "❌ NO")")
    println("   Final confidence:         $(confidence)")

    println("\n--- DEBUGGING COMPLETE ---")
end

# Execute the debug script
run_learning_debug()