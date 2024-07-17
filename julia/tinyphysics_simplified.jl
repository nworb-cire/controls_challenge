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
const STEER_RANGE = (-2, 2)
const MAX_ACC_DELTA = 0.5
const DEL_T = 0.1
const LAT_ACCEL_COST_MULTIPLIER = 50.0
const FUTURE_PLAN_STEPS = FPS * 5

struct State
    roll_lataccel
    v_ego
    a_ego
end

struct FuturePlan
    lataccel::Vector
    roll_lataccel::Vector
    v_ego::Vector
    a_ego::Vector
end

include("./model_runners.jl")

function get_current_lataccel(
        model::Function,
        sim_states::Vector{State},
        actions::Vector,
        past_preds::Vector
    )
    tokenized_actions = encode(past_preds)
    raw_states::Matrix{Float32} = hcat([getfield.(sim_states, field) for field in fieldnames(State)]...)
    states = hcat(actions, raw_states)
    return decode(predict(model, states, tokenized_actions))
end

include("./controllers.jl")

function get_data(data_path::String)
    df = CSV.read(data_path, DataFrame)
    processed_df = DataFrame(
        roll_lataccel = sin.(df.roll) .* ACC_G,
        v_ego = df.vEgo,
        a_ego = df.aEgo,
        target_lataccel = df.targetLateralAcceleration,
        steer_command = -df.steerCommand  # steer commands are logged with left-positive convention but this simulator uses right-positive
    )
    # replace Missing values with nan
    processed_df[!, :steer_command] = coalesce.(processed_df[!, :steer_command], NaN)
    
    processed_df[!, :roll_lataccel] = Float32.(processed_df[!, :roll_lataccel])
    processed_df[!, :v_ego] = Float32.(processed_df[!, :v_ego])
    processed_df[!, :a_ego] = Float32.(processed_df[!, :a_ego])
    processed_df[!, :target_lataccel] = Float32.(processed_df[!, :target_lataccel])
    processed_df[!, :steer_command] = Float32.(processed_df[!, :steer_command])

    return processed_df
end

function get_state_target_futureplan(data::DataFrame, step_idx::Int)
    state = data[step_idx+1, :]
    upper_idx = min(step_idx + FUTURE_PLAN_STEPS, size(data, 1))
    futureplan = FuturePlan(
        data.target_lataccel[step_idx+1:upper_idx],
        data.roll_lataccel[step_idx+1:upper_idx],
        data.v_ego[step_idx+1:upper_idx],
        data.a_ego[step_idx+1:upper_idx]
    )
    return State(state.roll_lataccel, state.v_ego, state.a_ego), state.target_lataccel, futureplan
end

function step(
        model::Function,
        controller::BaseController, 
        data::DataFrame, 
        step_idx::Int, 
        state_history::Vector{State}, 
        action_history::Vector, 
        current_lataccel_history::Vector, 
        target_lataccel_history::Vector, 
        current_lataccel
    )
    state, target, futureplan = get_state_target_futureplan(data, step_idx-1)
    state_history = cat(state_history[2:end], state, dims=1)
    target_lataccel_history = cat(target_lataccel_history[2:end], target, dims=1)
    target_future = futureplan

    # controls step
    action = update!(controller, target_lataccel_history[end], current_lataccel, state_history[end], target_future)
    if step_idx ≤ CONTROL_START_IDX
        action = data.steer_command[step_idx]
    end
    action = Float32(clamp(action, STEER_RANGE[1], STEER_RANGE[2]))
    action_history = cat(action_history[2:end], action, dims=1)

    # sim step
    pred = get_current_lataccel(model, state_history, action_history, current_lataccel_history)
    pred = clamp(pred, current_lataccel - MAX_ACC_DELTA, current_lataccel + MAX_ACC_DELTA)
    if step_idx > CONTROL_START_IDX
        current_lataccel = pred
    else
        current_lataccel = get_state_target_futureplan(data, step_idx)[2]
    end

    return state_history, action_history, current_lataccel_history, target_lataccel_history, current_lataccel
end

function init(data::DataFrame)
    step_idx = CONTEXT_LENGTH
    state_target_futureplans = [get_state_target_futureplan(data, i) for i in 1:step_idx]
    state_history = [x[1] for x in state_target_futureplans]
    action_history = data.steer_command[1:step_idx]
    current_lataccel_history = [x[2] for x in state_target_futureplans]
    target_lataccel_history = [x[2] for x in state_target_futureplans]
    current_lataccel = current_lataccel_history[end]
    
    action_history = Float32.(action_history)
    current_lataccel_history = Float32.(current_lataccel_history)
    target_lataccel_history = Float32.(target_lataccel_history)
    current_lataccel = Float32(current_lataccel)

    return state_history, action_history, current_lataccel_history, target_lataccel_history, current_lataccel
end


function rollout(controller, data, model)
    state_history, action_history, current_lataccel_history, target_lataccel_history, current_lataccel = init(data)
    lat_accel_cost = 0f0
    jerk_cost = 0f0
    for step_idx in CONTEXT_LENGTH:size(data, 1)
        state_history, action_history, current_lataccel_history, target_lataccel_history, current_lataccel = step(model, controller, data, step_idx, state_history, action_history, current_lataccel_history, target_lataccel_history, current_lataccel)
        lat_accel_cost += (target_lataccel_history[end] - current_lataccel)^2
        jerk_cost += (action_history[end] - action_history[end-1])^2
    end

    return 100mean(lat_accel_cost * LAT_ACCEL_COST_MULTIPLIER + jerk_cost)
end

const data_path = "./data/00000.csv"
import Zygote
function main()
    data = get_data(data_path)
    model = load_model()
    # ∇ = Zygote.gradient(pid) do pid
        controller = ZeroController()
        cost = rollout(model, controller, data)
        @show cost
    # end
    # @show ∇
    # controller = PIDController()
    # cost = run_rollout(data, controller)
    # @printf("\nAverage lataccel_cost: %6.4f, average jerk_cost: %6.4f, average total_cost: %6.4f\n", cost["lataccel_cost"], cost["jerk_cost"], cost["total_cost"])
end

main()
