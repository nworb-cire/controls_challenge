abstract type BaseController end

struct ZeroController <: BaseController
end

function update!(controller::ZeroController, target_lataccel::Float32, current_lataccel::Float32, state::State, futureplan::FuturePlan)
    return 0.0
end

mutable struct PIDController <: BaseController
    kp
    ki
    kd
    integral
    prev_error

    function PIDController()
        new(0.3, 0.05, -0.1, 0.0, 0.0)
    end

    function PIDController(kp, ki, kd)
        new(kp, ki, kd, zero(kp), zero(kp))
    end
end

function update!(controller::PIDController, target_lataccel, current_lataccel, state::State, futureplan::FuturePlan)
    error = target_lataccel - current_lataccel
    controller.integral += error
    derivative = error - controller.prev_error
    action = controller.kp * error + controller.ki * controller.integral + controller.kd * derivative
    controller.prev_error = error
    return action
end
