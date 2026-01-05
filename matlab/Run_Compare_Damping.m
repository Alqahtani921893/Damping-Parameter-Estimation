clear all; clc; close all;

%%%%%%%%% Plot settings
set(0,'DefaultAxesFontName', 'Times New Roman')
set(0,'DefaultAxesFontSize', 14)
set(0,'DefaultTextFontname', 'Times New Roman')
set(0,'DefaultTextFontSize', 14)
set(0,'defaultlinelinewidth', 2.0)
set(groot, 'defaultAxesTickLabelInterpreter', 'latex');
set(groot, 'defaultLegendInterpreter', 'latex');
set(0,'defaulttextinterpreter', 'latex')

%% ========== DAMPING PARAMETERS ==========
% Define parameters for each damping type
% These are set to produce roughly similar decay rates for comparison

% Viscous
params_viscous.type = 'viscous';
params_viscous.zeta = 0.05;
params_viscous.mu_c = 0;
params_viscous.mu_q = 0;

% Coulomb
params_coulomb.type = 'coulomb';
params_coulomb.zeta = 0;
params_coulomb.mu_c = 0.03;
params_coulomb.mu_q = 0;

% Quadratic
params_quadratic.type = 'quadratic';
params_quadratic.zeta = 0;
params_quadratic.mu_c = 0;
params_quadratic.mu_q = 0.05;

% Combined
params_combined.type = 'combined';
params_combined.zeta = 0.02;
params_combined.mu_c = 0.01;
params_combined.mu_q = 0.02;

%% ========== SYSTEM PARAMETERS ==========
k_th = 20;          % torsional spring stiffness
qv = 0;             % vertical excitation (free response)
qh = 0;             % horizontal excitation (free response)
Om = 5;             % frequency (not used in free response but needed for EOM)

%% ========== INITIAL CONDITIONS ==========
y0 = [120*pi/180, 0];  % 120 degrees initial angle, zero velocity

%% ========== TIME SETTINGS ==========
T = 2*pi/Om;
tf = 200*T;         % Long enough to see decay
t0 = 0;
dt = 0.01;
time_s = t0:dt:tf;

%% ========== ODE SOLVER ==========
options = odeset('RelTol', 1e-9, 'AbsTol', 1e-9);

fprintf('Simulating viscous damping...\n')
[t_v, y_v] = ode15s(@(t,y) EOM_Base_Pendulum(t, y, qh, qv, k_th, Om, params_viscous), ...
                    time_s, y0, options);

fprintf('Simulating Coulomb damping...\n')
[t_c, y_c] = ode15s(@(t,y) EOM_Base_Pendulum(t, y, qh, qv, k_th, Om, params_coulomb), ...
                    time_s, y0, options);

fprintf('Simulating quadratic damping...\n')
[t_q, y_q] = ode15s(@(t,y) EOM_Base_Pendulum(t, y, qh, qv, k_th, Om, params_quadratic), ...
                    time_s, y0, options);

fprintf('Simulating combined damping...\n')
[t_comb, y_comb] = ode15s(@(t,y) EOM_Base_Pendulum(t, y, qh, qv, k_th, Om, params_combined), ...
                          time_s, y0, options);

%% ========== PLOTTING ==========

% Figure 1: Time response comparison
figure('Name', 'Damping Comparison - Time Response', 'Position', [100 100 1000 600])

subplot(2,2,1)
plot(t_v, y_v(:,1)*180/pi, 'b', 'LineWidth', 1.5)
grid on
xlabel('$t$ [s]')
ylabel('$\theta$ [deg]')
title(sprintf('Viscous ($\\zeta$=%.3f)', params_viscous.zeta), 'Interpreter', 'latex')
xlim([0, tf/4])

subplot(2,2,2)
plot(t_c, y_c(:,1)*180/pi, 'r', 'LineWidth', 1.5)
grid on
xlabel('$t$ [s]')
ylabel('$\theta$ [deg]')
title(sprintf('Coulomb ($\\mu_c$=%.3f)', params_coulomb.mu_c), 'Interpreter', 'latex')
xlim([0, tf/4])

subplot(2,2,3)
plot(t_q, y_q(:,1)*180/pi, 'g', 'LineWidth', 1.5)
grid on
xlabel('$t$ [s]')
ylabel('$\theta$ [deg]')
title(sprintf('Quadratic ($\\mu_q$=%.3f)', params_quadratic.mu_q), 'Interpreter', 'latex')
xlim([0, tf/4])

subplot(2,2,4)
plot(t_comb, y_comb(:,1)*180/pi, 'm', 'LineWidth', 1.5)
grid on
xlabel('$t$ [s]')
ylabel('$\theta$ [deg]')
title('Combined', 'Interpreter', 'latex')
xlim([0, tf/4])

sgtitle('Comparison of Damping Models - Free Response', 'Interpreter', 'latex', 'FontSize', 16)

% Figure 2: Overlay comparison
figure('Name', 'Damping Comparison - Overlay', 'Position', [150 150 800 500])
hold on
plot(t_v, y_v(:,1)*180/pi, 'b-', 'LineWidth', 1.5, 'DisplayName', 'Viscous')
plot(t_c, y_c(:,1)*180/pi, 'r--', 'LineWidth', 1.5, 'DisplayName', 'Coulomb')
plot(t_q, y_q(:,1)*180/pi, 'g-.', 'LineWidth', 1.5, 'DisplayName', 'Quadratic')
plot(t_comb, y_comb(:,1)*180/pi, 'm:', 'LineWidth', 2, 'DisplayName', 'Combined')
grid on
xlabel('$t$ [s]')
ylabel('$\theta$ [deg]')
legend('Location', 'best')
title('Comparison of Damping Models', 'Interpreter', 'latex')
xlim([0, tf/4])

% Figure 3: Phase portraits comparison
figure('Name', 'Phase Portraits', 'Position', [200 200 1000 600])

subplot(2,2,1)
plot(y_v(:,1)*180/pi, y_v(:,2)*180/pi, 'b', 'LineWidth', 1)
grid on
xlabel('$\theta$ [deg]')
ylabel('$\dot{\theta}$ [deg/s]')
title('Viscous', 'Interpreter', 'latex')

subplot(2,2,2)
plot(y_c(:,1)*180/pi, y_c(:,2)*180/pi, 'r', 'LineWidth', 1)
grid on
xlabel('$\theta$ [deg]')
ylabel('$\dot{\theta}$ [deg/s]')
title('Coulomb', 'Interpreter', 'latex')

subplot(2,2,3)
plot(y_q(:,1)*180/pi, y_q(:,2)*180/pi, 'g', 'LineWidth', 1)
grid on
xlabel('$\theta$ [deg]')
ylabel('$\dot{\theta}$ [deg/s]')
title('Quadratic', 'Interpreter', 'latex')

subplot(2,2,4)
plot(y_comb(:,1)*180/pi, y_comb(:,2)*180/pi, 'm', 'LineWidth', 1)
grid on
xlabel('$\theta$ [deg]')
ylabel('$\dot{\theta}$ [deg/s]')
title('Combined', 'Interpreter', 'latex')

sgtitle('Phase Portraits - Different Damping Models', 'Interpreter', 'latex', 'FontSize', 16)

% Figure 4: Envelope decay comparison (peak amplitudes)
figure('Name', 'Decay Envelope', 'Position', [250 250 800 500])

% Extract peaks for each damping type
[pks_v, locs_v] = findpeaks(abs(y_v(:,1)));
[pks_c, locs_c] = findpeaks(abs(y_c(:,1)));
[pks_q, locs_q] = findpeaks(abs(y_q(:,1)));
[pks_comb, locs_comb] = findpeaks(abs(y_comb(:,1)));

hold on
plot(t_v(locs_v), pks_v*180/pi, 'b-o', 'MarkerSize', 4, 'DisplayName', 'Viscous')
plot(t_c(locs_c), pks_c*180/pi, 'r-s', 'MarkerSize', 4, 'DisplayName', 'Coulomb')
plot(t_q(locs_q), pks_q*180/pi, 'g-^', 'MarkerSize', 4, 'DisplayName', 'Quadratic')
plot(t_comb(locs_comb), pks_comb*180/pi, 'm-d', 'MarkerSize', 4, 'DisplayName', 'Combined')
grid on
xlabel('$t$ [s]')
ylabel('Peak $|\theta|$ [deg]')
legend('Location', 'best')
title('Amplitude Decay Envelope', 'Interpreter', 'latex')
xlim([0, tf/4])

fprintf('\n========== Simulation Complete ==========\n')
fprintf('Damping parameters used:\n')
fprintf('  Viscous:   zeta = %.3f\n', params_viscous.zeta)
fprintf('  Coulomb:   mu_c = %.3f\n', params_coulomb.mu_c)
fprintf('  Quadratic: mu_q = %.3f\n', params_quadratic.mu_q)
fprintf('  Combined:  zeta=%.3f, mu_c=%.3f, mu_q=%.3f\n', ...
        params_combined.zeta, params_combined.mu_c, params_combined.mu_q)
fprintf('==========================================\n')
