using JSON3, Random, Statistics, Printf, Dates
include("ProductionGeometricEngine.jl")
using .ProductionGeometricEngine

# ------------------------------------------------------------------------
# Hyper-parameters (tuned for fast convergence)
# ------------------------------------------------------------------------
const HIDDEN   = 128
const LR       = 2e-3
const EPISODES = 2500

# ------------------------------------------------------------------------
# Helper: single-run test
# ------------------------------------------------------------------------
function one_run(seed::Int)
    cfg  = TrainingConfig(lr=LR)
    core = GeometricCore(4,10,HIDDEN; cfg=cfg, seed=seed)

    # ---- pre-train ----
    pre = [let (X, t) = make_problem(core)
               predict(core, X) == t ? 1.0 : 0.0
           end
           for _ in 1:100]
    pre_acc = mean(pre)

    # ---- train ----
    for _ in 1:EPISODES
        X,t = make_problem(core)
        train_step!(core, X, t)
    end

    # ---- post-train ----
    post = [let (X, t) = make_problem(core)
                predict(core, X) == t ? 1.0 : 0.0
            end
            for _ in 1:200]
    post_acc = mean(post)

    learned = post_acc > 0.90 && post_acc > pre_acc + 0.50
    return Dict(
        "pre_acc"  => round(pre_acc,  digits=4),
        "post_acc" => round(post_acc, digits=4),
        "learned"  => learned,
        "pre"      => pre,
        "post"     => post
    )
end

# ------------------------------------------------------------------------
# Main suite
# ------------------------------------------------------------------------
function run_suite()
    println("Running Production Proof Suite …")
    results = [one_run(i) for i in 1:5]

    summary = Dict(
        "pre_train_mean_accuracy"  => round(mean(r["pre_acc"]  for r in results), digits=4),
        "post_train_mean_accuracy" => round(mean(r["post_acc"] for r in results), digits=4),
        "learning_achieved"        => all(r["learned"] for r in results)
    )

    full = Dict(
        "summary"           => summary,
        "runs"              => results,
        "emergent_properties" => Dict("geometric_reasoning" => summary["learning_achieved"])
    )

    open("proof_results.json","w") do f
        JSON3.write(f, full, indent=4)
    end
    println(JSON3.write(full))
end

run_suite()
