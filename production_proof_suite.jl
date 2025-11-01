using JSON, Random
include("ProductionGeometricEngine.jl")
using .GeometricEngine
using JSON3, Statistics, LinearAlgebra, Random, Dates, Printf

function run_tests()
    config = TrainingConfig()
    core = GeometricCore(config=config, seed=42)
    
    results = Dict(
        "pre_train_accuracy" => Float64[],
        "post_train_accuracy" => Float64[],
        "emergent_properties" => Dict("geometric_reasoning" => false)
    )
    
    # Pre-train test: should be random (~1/num_points accuracy)
    for _ in 1:50
        X, target = generate_problem(core)
        pred = predict(core, X)
        push!(results["pre_train_accuracy"], pred == target ? 1.0 : 0.0)
    end
    
    # Train for emergence: geometric learning of norm minimization
    for ep in 1:1000
        X, target = generate_problem(core)
        train_step!(core, X, target)
    end
    
    # Post-train test: should emerge high accuracy in finding closest point
    for _ in 1:50
        X, target = generate_problem(core)
        pred = predict(core, X)
        push!(results["post_train_accuracy"], pred == target ? 1.0 : 0.0)
    end
    
    pre_acc = mean(results["pre_train_accuracy"])
    post_acc = mean(results["post_train_accuracy"])
    
    # Emergent if accuracy improves significantly and geometrically (non-statistical threshold)
    results["emergent_properties"]["geometric_reasoning"] = post_acc > 0.8 && post_acc > pre_acc + 0.5
    
    # JSON output only
    println(JSON.json(results, 4))
end

run_tests()
