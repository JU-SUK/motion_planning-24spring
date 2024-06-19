function [dxdt,M,C,G] = robotStateFcn(x,u,param)
% robotStateFcn
%
%   [dxdt,M,C,G] = robotStateFcn(x,u,param) calculates the state
%   derivatives of a two-joint robotic arm given the current state x, input
%   torque u, and system parameters param. It also calculates and returns
%   the inertia matrix (M), Coriolis and centrifugal matrix (C), and
%   gravity vector (G) of the robotic arm.
%
%   Inputs:
%       x     - Current state vector [q1; q2; dq1; dq2], where q1 and q2
%               are the joint angles, and dq1 and dq2 are the joint velocities.
%       u     - Input torque vector applied to the joints [tau1; tau2].
%       param - Structure containing system parameters:
%               param.m1 - Mass of link 1
%               param.m2 - Mass of link 2
%               param.r1 - Length of link 1
%               param.r2 - Length of link 2
%               param.J1 - Inertia of link 1
%               param.J2 - Inertia of link 2
%               param.g  - Gravity constant
%
%   Outputs:
%       dxdt - State derivative vector [dq1; dq2; ddq1; ddq2], where ddq1
%              and ddq2 are the joint accelerations. 
%       M    - Inertia matrix of the robotic arm.
%       C    - Coriolis and centrifugal matrix of the robotic arm.
%       G    - Gravity vector of the robotic arm.
%
%   The function also calculates the matrices M, C, and G based on the
%   current state x and system parameters. If only the state derivative
%   dxdt is required, the function uses these matrices to compute the joint
%   accelerations ddq using the equation of motion: M*ddq = u - C*dq - G.

% Copyright 2023 The MathWorks, Inc.

% Initialize output variables.
dxdt = zeros(4,1);
M = zeros(2,2);
C = zeros(2,2);
G = zeros(2,1);

% Extract state variables.
q1 = x(1); % Joint angle 1
q2 = x(2); % Joint angle 2
dq1 = x(3); % Joint velocity 1
dq2 = x(4); % Joint velocity 2

% Extract parameters from the structure.
m1 = param.m1; % Mass 1
m2 = param.m2; % Mass 2
r1 = param.r1; % Link length 1
r2 = param.r2; % Link length 2
J1 = param.J1; % Inertia 1
J2 = param.J2; % Inertia 2
g  = param.g; % Gravity constant

% Compute p1, p2, p3, p4, p5, p6.
p1 = (m1 + m2) * r1^2 + m2 * r2^2 + J1;
p2 = m2 * r1 * r2;
p3 = m2 * r2^2;
p4 = p3 + J2;
p5 = (m1 + m2) * r1 * g;
p6 = m2 * r2 * g;

% Compute M matrix.
M = [p1 + 2 * p2 * cos(q2), p3 + p2 * cos(q2);
    p3 + p2 * cos(q2),      p4];

% Compute C matrix.
C = [-p2 * sin(q2) * dq1, -2 * p2 * sin(q2) * dq1;
    0,                    p2 * sin(q2) * dq2];

% Compute G vector
G = [p5 * cos(q1) + p6 * cos(q1 + q2);
    p6 * cos(q1 + q2)];

if nargout == 1
    % Compute the derivative of the state (dxdt).
    dq = [dq1; dq2];
    ddq = M \ (u - C * dq - G);
    dxdt = [dq; ddq];
end
end