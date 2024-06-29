using ArgParse
using Random
using DataFrames
using CSV
using StatsBase
using Dates
using Statistics
using Printf
using NNlib: softmax
using MD5
using Plots

include("./onnx_ops.jl")

const ACC_G = 9.81
const FPS = 10
const CONTROL_START_IDX = 100
const COST_END_IDX = 500
const CONTEXT_LENGTH = 20
const VOCAB_SIZE = 1024
const LATACCEL_RANGE = (-5, 5)
const STEER_RANGE = (-2, 2)
const MAX_ACC_DELTA = 0.5
const DEL_T = 0.1
const LAT_ACCEL_COST_MULTIPLIER = 50.0

const FUTURE_PLAN_STEPS = FPS * 5

struct State
    roll_lataccel::Float32
    v_ego::Float32
    a_ego::Float32
end

struct FuturePlan
    lataccel::Vector{Float32}
    roll_lataccel::Vector{Float32}
    v_ego::Vector{Float32}
    a_ego::Vector{Float32}
end

struct LataccelTokenizer
    vocab_size::Int
    bins::Vector{Float32}

    function LataccelTokenizer()
        bins = LinRange(LATACCEL_RANGE[1], LATACCEL_RANGE[2], VOCAB_SIZE)
        new(VOCAB_SIZE, bins)
    end
end

function encode(tokenizer::LataccelTokenizer, value::Union{Float32, Vector{Float32}})
    value = clip(tokenizer, value)
    digitized = searchsortedlast.(Ref(tokenizer.bins), value)
    return digitized
end

function decode(tokenizer::LataccelTokenizer, token::Union{Int, Vector{Int}})
    return tokenizer.bins[token]
end

function clip(tokenizer::LataccelTokenizer, value::Union{Float32, Vector{Float32}})
    return clamp.(value, LATACCEL_RANGE[1], LATACCEL_RANGE[2])
end

const model_path = "./models/tinyphysics.onnx"
onnx_model = let
    b = 3
    A = ones(Float32, 4, 20, b)
    B = ones(Int64, 20, b)
    tape = ONNX.load(model_path, A, B)
    ONNX.compile(tape)
end

struct TinyPhysicsModel
    tokenizer::LataccelTokenizer
    debug::Bool

    function TinyPhysicsModel(; debug=false)
        tokenizer = LataccelTokenizer()
        new(tokenizer, debug)
    end
end

function predict(states::Matrix{Float32}, tokens::Vector{Int64}; temperature=1.0)
    states = permutedims(states)
    states = reshape(states, size(states)..., 1)
    tokens = reshape(tokens, size(tokens)..., 1)
    res = onnx_model(states, tokens)
    res = permutedims(res, [3, 2, 1])
    res = res[1:1, :, :]  # fixme
    probs = softmax(res ./ temperature; dims=size(res, 3))
    @assert size(probs, 1) == 1
    @assert size(probs, 3) == VOCAB_SIZE
    weights = Weights(probs[1, 1, :])
    return sample(1:VOCAB_SIZE, weights)
end

function get_current_lataccel(model::TinyPhysicsModel, sim_states::Vector{State}, actions::Vector{Float32}, past_preds::Vector{Float32})
    tokenized_actions = encode(model.tokenizer, past_preds)
    raw_states::Matrix{Float32} = hcat([getfield.(sim_states, field) for field in fieldnames(State)]...)
    states = hcat(actions, raw_states)
    @show size(states)
    @show size(tokenized_actions)
    return decode(model.tokenizer, predict(states, tokenized_actions))
end

abstract type BaseController end

struct ZeroController <: BaseController
end

function update!(controller::ZeroController, target_lataccel::Float32, current_lataccel::Float32, state::State, futureplan::FuturePlan)
    return 0.0
end

mutable struct PIDController <: BaseController
    kp::Float32
    ki::Float32
    kd::Float32
    integral::Float32
    prev_error::Float32

    function PIDController()
        kp = 0.3
        ki = 0.05
        kd = -0.1
        integral = 0.0
        prev_error = 0.0
        new(kp, ki, kd, integral, prev_error)
    end

    function PIDController(kp::Float32, ki::Float32, kd::Float32)
        new(kp, ki, kd, 0.0, 0.0)
    end
end

function update!(controller::PIDController, target_lataccel::Float32, current_lataccel::Float32, state::State, futureplan::FuturePlan)
    error = target_lataccel - current_lataccel
    controller.integral += error
    derivative = error - controller.prev_error
    action = controller.kp * error + controller.ki * controller.integral + controller.kd * derivative
    controller.prev_error = error
    return action
end

mutable struct TinyPhysicsSimulator
    sim_model::TinyPhysicsModel
    data::DataFrame
    controller::BaseController
    debug::Bool
    step_idx::Int
    state_history::Vector{State}
    action_history::Vector{Float32}
    current_lataccel_history::Vector{Float32}
    target_lataccel_history::Vector{Float32}
    target_future::Union{Nothing, FuturePlan}
    current_lataccel::Float32

    function TinyPhysicsSimulator(model::TinyPhysicsModel, data::DataFrame, controller::BaseController, debug::Bool=false)
        step_idx = CONTEXT_LENGTH
        state_target_futureplans = [get_state_target_futureplan(data, i) for i in 1:step_idx]
        state_history = [x[1] for x in state_target_futureplans]
        action_history = data.steer_command[1:step_idx]
        current_lataccel_history = [x[2] for x in state_target_futureplans]
        target_lataccel_history = [x[2] for x in state_target_futureplans]
        target_future = nothing
        current_lataccel = current_lataccel_history[end]
        seed = parse(Int, bytes2hex(md5(data_path))[24:end], base = 16) % 10^4
        Random.seed!(seed)
        new(model, data, controller, debug, step_idx, state_history, action_history, current_lataccel_history, target_lataccel_history, target_future, current_lataccel)
    end
end

function get_data(data_path::String)
    df = CSV.read(data_path, DataFrame)
    processed_df = DataFrame(
        roll_lataccel = sin.(df.roll) .* ACC_G,
        v_ego = df.vEgo,
        a_ego = df.aEgo,
        target_lataccel = df.targetLateralAcceleration,
        steer_command = df.steerCommand
    )
    return processed_df
end

function sim_step(sim::TinyPhysicsSimulator, step_idx::Int)
    pred = get_current_lataccel(sim.sim_model, sim.state_history[end-CONTEXT_LENGTH+1:end], sim.action_history[end-CONTEXT_LENGTH+1:end], sim.current_lataccel_history[end-CONTEXT_LENGTH+1:end])
    pred = clamp(pred, sim.current_lataccel - MAX_ACC_DELTA, sim.current_lataccel + MAX_ACC_DELTA)
    if step_idx >= CONTROL_START_IDX
        sim.current_lataccel = pred
    else
        sim.current_lataccel = get_state_target_futureplan(sim.data, step_idx)[2]
    end
    push!(sim.current_lataccel_history, sim.current_lataccel)
end

function control_step(sim::TinyPhysicsSimulator, step_idx::Int)
    action = update!(sim.controller, sim.target_lataccel_history[step_idx], sim.current_lataccel, sim.state_history[step_idx], sim.target_future)
    if step_idx < CONTROL_START_IDX
        action = sim.data.steer_command[step_idx]
    end
    action = clamp(action, STEER_RANGE[1], STEER_RANGE[2])
    push!(sim.action_history, action)
end

function get_state_target_futureplan(data::DataFrame, step_idx::Int)
    state = data[step_idx, :]
    upper_idx = min(step_idx + FUTURE_PLAN_STEPS, size(data, 1))
    futureplan = FuturePlan(
        data.target_lataccel[step_idx+1:upper_idx],
        data.roll_lataccel[step_idx+1:upper_idx],
        data.v_ego[step_idx+1:upper_idx],
        data.a_ego[step_idx+1:upper_idx]
    )
    return State(state.roll_lataccel, state.v_ego, state.a_ego), state.target_lataccel, futureplan
end

function step(sim::TinyPhysicsSimulator)
    state, target, futureplan = get_state_target_futureplan(sim.data, sim.step_idx)
    push!(sim.state_history, state)
    push!(sim.target_lataccel_history, target)
    sim.target_future = futureplan
    control_step(sim, sim.step_idx)
    sim_step(sim, sim.step_idx)
    sim.step_idx += 1
end

function compute_cost(sim::TinyPhysicsSimulator)
    target = sim.target_lataccel_history[CONTROL_START_IDX:COST_END_IDX]
    pred = sim.current_lataccel_history[CONTROL_START_IDX:COST_END_IDX]

    lat_accel_cost = mean((target .- pred).^2) * 100
    jerk_cost = mean((diff(pred) ./ DEL_T).^2) * 100
    total_cost = (lat_accel_cost * LAT_ACCEL_COST_MULTIPLIER) + jerk_cost
    return Dict("lataccel_cost" => lat_accel_cost, "jerk_cost" => jerk_cost, "total_cost" => total_cost)
end

function rollout(sim::TinyPhysicsSimulator)
    for _ in CONTEXT_LENGTH:size(sim.data, 1)
        step(sim)
        if sim.debug && sim.step_idx % 10 == 0
            @printf("Step %5d: Current lataccel: %6.2f, Target lataccel: %6.2f\n", sim.step_idx, sim.current_lataccel, sim.target_lataccel_history[end])
            p1 = plot(
                [sim.target_lataccel_history, sim.current_lataccel_history],
                label=["Target lataccel" "Current lataccel"],
                xlabel="Step",
                ylabel="Lateral Acceleration",
                title="Lateral Acceleration"
            )
            p2 = plot(
                [sim.action_history],
                label=["Action"],
                xlabel="Step",
                ylabel="Action",
                title="Action"
            )
            p3 = plot(
                [[state.roll_lataccel for state in sim.state_history]],
                label=["Roll Lateral Acceleration"],
                xlabel="Step",
                ylabel="Lateral Accel due to Road Roll",
                title="Lateral Accel due to Road Roll"
            )
            p4 = plot(
                [[state.v_ego for state in sim.state_history]],
                label=["v_ego"],
                xlabel="Step",
                ylabel="v_ego",
                title="v_ego"
            )
            for pl in [p1, p2, p3, p4]
                vline!(pl, [CONTROL_START_IDX], label="Control Start", color="black", linestyle=:dash)
            end
            plot(p1, p2, p3, p4, layout=(4, 1)) |> display
        end
    end

    return compute_cost(sim)
end

function get_available_controllers()
    return [splitpath(f)[2] for f in readdir("controllers") if isfile(joinpath("controllers", f)) && endswith(f, ".py") && splitpath(f)[2] != "__init__.py"]
end

function run_rollout(data::DataFrame, controller::BaseController; debug=false)
    tinyphysicsmodel = TinyPhysicsModel(
        # debug
    )
    sim = TinyPhysicsSimulator(tinyphysicsmodel, data, controller, debug)
    return rollout(sim), sim.target_lataccel_history, sim.current_lataccel_history
end

const data_path = "./data/00000.csv"
function main()
    # available_controllers = get_available_controllers()

    s = ArgParseSettings()
    @add_arg_table! s begin
        "--data_path" 
            help="Path to the data file or directory" 
            # required=true
            arg_type=String
            default="./data/00000.csv"
        "--num_segs" 
            help="Number of segments" 
            default=100
            arg_type=Int
        "--debug" 
            help="Enable debug mode" 
            # action=:store_true
            default=true
        "--controller" 
            help="Type of controller" 
            # choices=available_controllers 
            default="pid"
            arg_type=String
    end
    args = parse_args(s)

    data_path = args["data_path"]
    if args["controller"] == "zero"
        controller = ZeroController()
    elseif args["controller"] == "pid"
        controller = PIDController()
    else
        error("Invalid controller")
    end
    if isfile(data_path)
        data = get_data(data_path)
        cost, _, _ = run_rollout(data, controller, debug=args["debug"])
        @printf("\nAverage lataccel_cost: %6.4f, average jerk_cost: %6.4f, average total_cost: %6.4f\n", cost["lataccel_cost"], cost["jerk_cost"], cost["total_cost"])
    elseif isdir(data_path)
        run_rollout_partial = (x) -> run_rollout(x, args["controller"], args["model_path"], debug=false)
        files = readdir(data_path)
        data = [get_data(joinpath(data_path, f)) for f in files[1:min(args["num_segs"], end)]]
        results = process_map(run_rollout_partial, data, nworkers=16, batch_size=10)
        costs = [result[1] for result in results]
        costs_df = DataFrame(costs)
        @printf("\nAverage lataccel_cost: %6.4f, average jerk_cost: %6.4f, average total_cost: %6.4f\n", mean(costs_df.lataccel_cost), mean(costs_df.jerk_cost), mean(costs_df.total_cost))

        plt.figure()
        for cost in names(costs_df)
            plt.hist(costs_df[cost], bins=0:10:1000, label=cost, alpha=0.5)
        end
        plt.xlabel("Costs")
        plt.ylabel("Frequency")
        plt.title("Costs Distribution")
        plt.legend()
        plt.show()
    end
end

main()
