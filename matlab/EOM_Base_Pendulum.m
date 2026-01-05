function dydt = EOM_Base_Pendulum(t, y, qh, qv, k_th, Om, damping_params)
% EOM_Base_Pendulum - Equations of motion for horizontal pendulum with
%                     multiple damping models
%
% Inputs:
%   t             - time
%   y             - state vector [theta; theta_dot]
%   qh            - horizontal excitation amplitude
%   qv            - vertical excitation amplitude
%   k_th          - torsional spring stiffness (normalized)
%   Om            - excitation frequency
%   damping_params - struct with fields:
%       .type     - 'viscous', 'coulomb', 'quadratic', or 'combined'
%       .zeta     - viscous damping ratio (for viscous/combined)
%       .mu_c     - Coulomb friction coefficient (for coulomb/combined)
%       .mu_q     - quadratic damping coefficient (for quadratic/combined)
%
% Damping Models:
%   Viscous:   F_d = 2*zeta*theta_dot           (linear in velocity)
%   Coulomb:   F_d = mu_c*sign(theta_dot)       (constant friction)
%   Quadratic: F_d = mu_q*theta_dot*|theta_dot| (quadratic in velocity)
%   Combined:  F_d = viscous + coulomb + quadratic

thet = y(1);
thet_dt = y(2);

dydt(1,1) = y(2);

% Compute damping force based on type
switch damping_params.type
    case 'viscous'
        % Linear viscous damping: F_d = 2*zeta*theta_dot
        F_damping = 2 * damping_params.zeta * thet_dt;

    case 'coulomb'
        % Coulomb friction: F_d = mu_c * sign(theta_dot)
        % Use smooth approximation to avoid numerical issues at zero velocity
        epsilon = 1e-6;  % small value to smooth sign function
        F_damping = damping_params.mu_c * tanh(thet_dt / epsilon);

    case 'quadratic'
        % Quadratic damping: F_d = mu_q * theta_dot * |theta_dot|
        F_damping = damping_params.mu_q * thet_dt * abs(thet_dt);

    case 'combined'
        % Combined damping: all three types
        epsilon = 1e-6;
        F_viscous = 2 * damping_params.zeta * thet_dt;
        F_coulomb = damping_params.mu_c * tanh(thet_dt / epsilon);
        F_quadratic = damping_params.mu_q * thet_dt * abs(thet_dt);
        F_damping = F_viscous + F_coulomb + F_quadratic;

    otherwise
        error('Unknown damping type: %s. Use viscous, coulomb, quadratic, or combined', damping_params.type);
end

% Equation of motion
A = F_damping + k_th*thet - cos(thet) + Om^2*sin(Om*t)*(qh*sin(thet) - qv*cos(thet));

dydt(2,1) = -A;

end
