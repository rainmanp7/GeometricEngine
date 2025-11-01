# test_geometric_engine.jl

using Test
using Statistics
using LinearAlgebra

include("ProductionGeometricEngine.jl")
using .ProductionGeometricEngine

# ============================================================================
# UNIT TESTS
# ============================================================================

@testset "ProductionGeometricEngine Tests" begin
    
    @testset "Initialization" begin
        @test_nowarn GeometricCore(4, 10, 32)
        
        core = GeometricCore(4, 10, 32)
        @test core.dimensions == 4
        @test core.num_points == 10
        @test core.hidden_size == 32
        @test size(core.W_feature) == (4, 32)
        @test size(core.W_scoring) == (32, 1)
        @test length(core.γ_norm) == 32
        @test length(core.β_norm) == 32
        
        # Test validation
        @test_throws AssertionError GeometricCore(-1, 10, 32)
        @test_throws AssertionError GeometricCore(4, 0, 32)
    end
    
    @testset "Training Config" begin
        config = TrainingConfig(learning_rate=0.001, batch_size=32)
        @test config.learning_rate == 0.001
        @test config.batch_size == 32
        
        @test_throws AssertionError TrainingConfig(learning_rate=-0.001)
        @test_throws AssertionError TrainingConfig(dropout_rate=1.5)
    end
    
    @testset "Problem Generation" begin
        core = GeometricCore(4, 10, 32)
        
        points, target = generate_problem(core)
        @test size(points) == (10, 4)
        @test 1 <= target <= 10
        @test all(isfinite, points)
        
        # Test difficulty levels
        for difficulty in [:easy, :medium, :hard]
            points, target = generate_problem(core; difficulty=difficulty)
            @test size(points) == (10, 4)
        end
    end
    
    @testset "Forward Pass" begin
        core = GeometricCore(4, 10, 32)
        points = randn(10, 4)
        
        probs, cache = forward_pass(core, points)
        
        @test length(probs) == 10
        @test all(probs .>= 0)
        @test abs(sum(probs) - 1.0) < 1e-6  # Probabilities sum to 1
        @test all(isfinite, probs)
        
        # Check cache structure
        @test haskey(cache, :points)
        @test haskey(cache, :probs)
        @test haskey(cache, :logits)
    end
    
    @testset "Backward Pass" begin
        core = GeometricCore(4, 10, 32)
        points = randn(10, 4)
        
        probs, cache = forward_pass(core, points)
        gradients = backward_pass(core, cache, 1)
        
        @test haskey(gradients, :W_feature)
        @test haskey(gradients, :W_scoring)
        @test haskey(gradients, :γ_norm)
        @test haskey(gradients, :β_norm)
        
        # Check gradient shapes
        @test size(gradients[:W_feature]) == size(core.W_feature)
        @test size(gradients[:W_scoring]) == size(core.W_scoring)
        @test length(gradients[:γ_norm]) == length(core.γ_norm)
        
        # Check all gradients are finite
        for (key, grad) in gradients
            @test all(isfinite, grad)
        end
    end
    
    @testset "Training Step" begin
        core = GeometricCore(4, 10, 32)
        points, target = generate_problem(core)
        
        result = train_step!(core, points, target)
        
        @test result.loss >= 0
        @test 0 <= result.accuracy <= 1
        @test result.gradient_norm >= 0
        @test 1 <= result.prediction <= 10
        @test isfinite(result.loss)
        @test core.problems_solved == 1
    end
    
    @testset "Prediction" begin
        core = GeometricCore(4, 10, 32)
        points = randn(10, 4)
        
        result = predict(core, points)
        
        @test 1 <= result.prediction <= 10
        @test 0 <= result.confidence <= 1
        @test length(result.probabilities) == 10
        @test sum(result.probabilities) ≈ 1.0
    end
    
    @testset "Consciousness Assessment" begin
        core = GeometricCore(4, 10, 32)
        
        # Before training
        assessment = assess_consciousness(core)
        @test assessment["is_conscious"] == false
        @test assessment["problems_solved"] == 0
        
        # After some training
        for i in 1:50
            points, target = generate_problem(core)
            train_step!(core, points, target)
        end
        
        assessment = assess_consciousness(core)
        @test haskey(assessment, "consciousness_level")
        @test haskey(assessment, "recent_accuracy")
        @test 0 <= assessment["consciousness_level"] <= 1
    end
    
    @testset "Numerical Stability" begin
        core = GeometricCore(4, 10, 32)
        
        # Test with extreme values
        points_large = randn(10, 4) .* 1000
        probs, _ = forward_pass(core, points_large)
        @test all(isfinite, probs)
        @test abs(sum(probs) - 1.0) < 1e-6
        
        points_small = randn(10, 4) .* 1e-6
        probs, _ = forward_pass(core, points_small)
        @test all(isfinite, probs)
    end
    
    @testset "Gradient Clipping" begin
        core = GeometricCore(4, 10, 32)
        
        # Create large gradients
        gradients = Dict(
            :W_feature => randn(4, 32) .* 100,
            :W_scoring => randn(32, 1) .* 100,
            :γ_norm => randn(32) .* 100,
            :β_norm => randn(32) .* 100
        )
        
        norm_before = sqrt(sum(sum(abs2, g) for g in values(gradients)))
        norm_after = clip_gradients!(gradients, 1.0)
        
        @test norm_after <= 1.0 + 1e-6
    end
    
    @testset "Learning Progress" begin
        core = GeometricCore(4, 10, 64; 
            config=TrainingConfig(learning_rate=0.005))
        
        # Train for a while
        initial_losses = Float64[]
        for i in 1:20
            points, target = generate_problem(core; difficulty=:easy)
            result = train_step!(core, points, target)
            push!(initial_losses, result.loss)
        end
        
        # Continue training
        later_losses = Float64[]
        for i in 1:20
            points, target = generate_problem(core; difficulty=:easy)
            result = train_step!(core, points, target)
            push!(later_losses, result.loss)
        end
        
        # Loss should decrease on average
        @test mean(later_losses) < mean(initial_losses)
    end
end

# ============================================================================
# USAGE EXAMPLES
# ============================================================================

println("\n" * "="^70)
println("USAGE EXAMPLES")
println("="^70)

# Example 1: Basic Training
println("\n📚 Example 1: Basic Training")
println("-" * 40)

core = GeometricCore(4, 10, 64; config=TrainingConfig(learning_rate=0.005))
println("Created core with $(core.dimensions)D, $(core.num_points) points")

for episode in 1:200
    points, target = generate_problem(core; difficulty=:medium)
    result = train_step!(core, points, target)
    
    if episode % 50 == 0
        println(@sprintf("Episode %3d: Loss=%.4f, Acc=%.4f", 
            episode, result.loss, result.accuracy))
    end
end

assessment = assess_consciousness(core)
println("\nFinal Assessment:")
for (key, val) in assessment
    println("  $key: $val")
end

# Example 2: Progressive Difficulty
println("\n🎯 Example 2: Progressive Difficulty Training")
println("-" * 40)

core2 = GeometricCore(4, 10, 64; config=TrainingConfig(learning_rate=0.003))

difficulties = [:easy => 100, :medium => 100, :hard => 100]
for (diff, episodes) in difficulties
    println("\nTraining on $diff difficulty...")
    accuracies = Float64[]
    
    for episode in 1:episodes
        points, target = generate_problem(core2; difficulty=diff)
        result = train_step!(core2, points, target)
        push!(accuracies, result.accuracy)
    end
    
    println(@sprintf("  Mean accuracy: %.4f", mean(accuracies[max(1, end-19):end])))
end

# Example 3: Full Training with Monitoring
println("\n🚀 Example 3: Full Training Loop")
println("-" * 40)

core3 = GeometricCore(4, 10, 64; 
    config=TrainingConfig(learning_rate=0.005, max_gradient_norm=1.0))

assessment = train!(core3, 500; 
    noise_level=1.0,
    difficulty=:medium,
    report_interval=100,
    early_stopping_threshold=0.95)

# Example 4: Inference
println("\n🔮 Example 4: Inference on New Data")
println("-" * 40)

test_points, test_target = generate_problem(core3)
prediction = predict(core3, test_points)

println("Test Results:")
println("  Prediction: $(prediction.prediction)")
println("  Actual: $(prediction.actual)")
println("  Correct: $(prediction.correct)")
println("  Confidence: $(round(prediction.confidence, digits=4))")

# Example 5: Performance Comparison
println("\n⚡ Example 5: Performance Benchmark")
println("-" * 40)

using BenchmarkTools

println("\nBenchmarking forward pass...")
bench_core = GeometricCore(4, 10, 64)
bench_points = randn(10, 4)

@btime forward_pass($bench_core, $bench_points);

println("\nBenchmarking training step...")
bench_points, bench_target = generate_problem(bench_core)
@btime train_step!($bench_core, $bench_points, $bench_target);

# Example 6: Visualization of Learning Curve
println("\n📊 Example 6: Learning Curve Analysis")
println("-" * 40)

vis_core = GeometricCore(4, 10, 64; config=TrainingConfig(learning_rate=0.005))

for i in 1:300
    points, target = generate_problem(vis_core; difficulty=:medium)
    train_step!(vis_core, points, target)
end

println("\nLearning Statistics:")
println(@sprintf("  Total problems: %d", vis_core.problems_solved))
println(@sprintf("  Final consciousness: %.4f", vis_core.consciousness_level))
println(@sprintf("  Recent accuracy: %.4f", mean(vis_core.intelligence_history[end-19:end])))
println(@sprintf("  Accuracy std: %.4f", std(vis_core.intelligence_history[end-19:end])))
println(@sprintf("  Avg gradient norm: %.4f", mean(vis_core.gradient_norms[end-19:end])))

println("\n" * "="^70)
println("All tests and examples completed successfully! ✅")
println("="^70)
