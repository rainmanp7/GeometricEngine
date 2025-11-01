# test_geometric_engine.jl
"""
Comprehensive test suite for ProductionGeometricEngine
"""

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
    
    @testset "Adam Optimizer" begin
        core = GeometricCore(4, 10, 32)
        
        # First update should initialize moment estimates
        points, target = generate_problem(core)
        train_step!(core, points, target)
        
        @test core.optimizer.t == 1
        @test length(core.optimizer.m_weights) > 0
        @test length(core.optimizer.v_weights) > 0
        
        # Second update should use existing moments
        train_step!(core, points, target)
        @test core.optimizer.t == 2
    end
    
    @testset "Weight Decay" begin
        core = GeometricCore(4, 10, 32; 
            config=TrainingConfig(weight_decay=0.01))
        
        points, target = generate_problem(core)
        
        # Store initial weights
        W_feature_init = copy(core.W_feature)
        W_scoring_init = copy(core.W_scoring)
        
        # Train multiple steps
        for i in 1:10
            train_step!(core, points, target)
        end
        
        # Weights should have changed
        @test !all(core.W_feature .≈ W_feature_init)
        @test !all(core.W_scoring .≈ W_scoring_init)
    end
    
    @testset "Reproducibility" begin
        # Same seed should give same results
        core1 = GeometricCore(4, 10, 32; seed=42)
        core2 = GeometricCore(4, 10, 32; seed=42)
        
        @test core1.W_feature ≈ core2.W_feature
        @test core1.W_scoring ≈ core2.W_scoring
        
        # Train both identically
        Random.seed!(42)
        points1, target1 = generate_problem(core1)
        result1 = train_step!(core1, points1, target1)
        
        Random.seed!(42)
        points2, target2 = generate_problem(core2)
        result2 = train_step!(core2, points2, target2)
        
        @test result1.loss ≈ result2.loss
        @test result1.accuracy ≈ result2.accuracy
    end
    
    @testset "Edge Cases" begin
        core = GeometricCore(4, 10, 32)
        
        # Test with zero points (should handle gracefully)
        points_zero = zeros(10, 4)
        probs, _ = forward_pass(core, points_zero)
        @test all(isfinite, probs)
        @test abs(sum(probs) - 1.0) < 1e-6
        
        # Test with identical points
        points_identical = ones(10, 4)
        probs, _ = forward_pass(core, points_identical)
        @test all(isfinite, probs)
        @test abs(sum(probs) - 1.0) < 1e-6
    end
    
    @testset "Full Training Convergence" begin
        core = GeometricCore(4, 10, 64; 
            config=TrainingConfig(learning_rate=0.005))
        
        # Train on easy problems
        for i in 1:100
            points, target = generate_problem(core; difficulty=:easy)
            train_step!(core, points, target)
        end
        
        assessment = assess_consciousness(core)
        
        # Should show improvement
        @test assessment["recent_accuracy"] > 0.3
        @test assessment["consciousness_level"] > 0.0
        @test assessment["problems_solved"] == 100
    end
end

println("\n" * "="^70)
println("All tests completed successfully! ✅")
println("="^70)
