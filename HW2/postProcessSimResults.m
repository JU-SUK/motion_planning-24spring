function [out,ref,lbs] = postProcessSimResults(simout,C)
% postProcessSimResults
%
%   [out,ref,lbs] = postProcessSimResults(simout,C) processes the
%   simulation results from a Simulink model. It extracts the relevant
%   data, organizes it into timetables, and computes the sliding mode
%   control error signals.
%
%   Inputs:
%       simout - Simulation output object
%       C      - Coefficient vector in the sliding surface.
%
%   Outputs:
%       out  - A timetable containing the processed state and input data
%              (joint angles, velocities, and torques) from the simulation.
%       ref  - A timetable containing the reference state data (desired
%              joint angles and velocities) from the simulation.
%       lbs  - Cell array of labels for the data columns in 'out' and 'ref'.

% Copyright 2023 The MathWorks, Inc.

% Define labels for the output data
lbs = { ...
    ["Angular Position"; "Joint 1"]; ...
    ["Angular Position"; "Joint 2"]; ...
    ["Angular Velocity"; "Joint 1"]; ...
    ["Angular Velocity"; "Joint 2"]};

% Extract and process the simulation data for output timetable
out = extractTimetable(simout.logsout);
out = splitvars(out,"x1","NewVariableNames",["q1","q2"]);
out = splitvars(out,"x2","NewVariableNames",["qdot1","qdot2"]);
out = splitvars(out, "u", "NewVariableNames",["Tau1","Tau2"]);
out = removevars(out,["xr1" "xr2"]);

% Extract and process the simulation data for reference timetable
ref = extractTimetable(simout.logsout);
ref = splitvars(ref,"xr1","NewVariableNames",["q1","q2"]);
ref = splitvars(ref,"xr2","NewVariableNames",["qdot1","qdot2"]);
ref = removevars(ref,["x1" "x2" "u"]);

% Compute the sliding mode control error signals and add them to 'out'
out.S1 = C(1)*(ref.q1-out.q1) + (ref.qdot1-out.qdot1);
out.S2 = C(1)*(ref.q2-out.q2) + (ref.qdot2-out.qdot2);
end