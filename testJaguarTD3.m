% Evaluation script for the trained TD3 agent, with EKF pose fusion and online SLAM.
clear; clc; close all;

% --- configuration ---
numTestEpisodes     = 30;
maxSteps            = 450;
renderVisualization = true;
enableSlamMapVis    = true;       % show the SLAM map panel
enableSensorNoise   = true;       % encoder slip and gyro drift
enableSafetyGuardrail = true;     % speed scaling near obstacles
controlMode    = 'deployment_wheel';
odomSourceMode = 'slam';
userModelPath = '"C:\UTM\Sem2\Master\Code\SubnitTest\try-lownoise-randMap2-contNoise2\JaguarAgentLatest.mat"';
% robot limits
v_max         = 0.5;          % m/s
maxWheelSpeed = 15.0;         % rad/s
% guardrail
alphaFilter = 0.3;            % low-pass coefficient (0 < alpha <= 1)
dStop       = 0.25;           % hard stop distance (m)
dWarn       = 0.60;           % start slowing down here (m)
maxLinAcc   = 0.5;            % m/s^2
dtControl   = 0.1;            % s
% EKF (odometry predict, scan-match correct)
odomTransNoiseStd   = 0.03;   % per meter traveled
odomRotNoiseStd     = 0.02;   % per radian turned
odomTransNoiseFloor = 0.005;  % min per-step translation std (m)
odomRotNoiseFloor   = deg2rad(0.3);  % min per-step rotation std (rad)
scanMatchTransNoiseStd = 0.05;       % m
scanMatchRotNoiseStd   = deg2rad(2); % rad
matchRejectTransThresh = 0.5;        % reject match if position jump exceeds this (m)
matchRejectRotThresh   = deg2rad(30);% reject match if heading jump exceeds this (rad)
initPoseVar = [1e-4, 1e-4, deg2rad(0.5)^2]; % initial [x, y, theta] variance
evalMapMode     = 'cycle_pool';  % 'cycle_pool' or 'designed_file'
customMapFile   = fullfile(pwd, 'JaguarDesignedMaps', 'map_1.mat');

% --- environment ---
fprintf('Initializing Environment [Control: %s | Odom: %s | Guardrail: %d | Noise: %d]...\n', ...
        upper(controlMode), upper(odomSourceMode), enableSafetyGuardrail, enableSensorNoise);
env = JaguarSLAMEnv_DynamicGoal();
env.UseSafetyBuffer = true;
if enableSensorNoise
    env.PeriodicNoiseAmp   = 0.05;
    env.EncoderPeriodicAmp = 0.03;
    env.GyroBiasWalkStd    = 0.01;
else
    env.PeriodicNoiseAmp   = 0;
    env.EncoderPeriodicAmp = 0;
    env.GyroBiasWalkStd    = 0;
end
% get the map pool and drop any empty entries
if exist('JaguarMapPool', 'file')
    rawPool = JaguarMapPool();
elseif ismethod(env, 'getMapPool')
    rawPool = env.getMapPool();
else
    rawPool = env.MapPool;
end
cleanPool = {};
for k = 1:numel(rawPool)
    if ~isempty(rawPool{k}) && isstruct(rawPool{k})
        cleanPool{end+1} = rawPool{k}; %#ok<AGROW>
    end
end
if strcmpi(evalMapMode, 'designed_file') && exist(customMapFile, 'file')
    loadedData = load(customMapFile, 'mapDef');
    cleanPool = {loadedData.mapDef};
    env.MapMode = 'designed';
else
    env.MapMode = 'fixed';
end
env.MaxSteps = maxSteps;

% --- load agent ---
if ~isempty(userModelPath) && exist(userModelPath, 'file')
    modelPath = userModelPath;
else
    candidateDirs = { ...
        fullfile(pwd, 'try-lownoise-randMap2-contNoise2'), ...
        fullfile(pwd, 'stable-lownoise-randMap2-contNoise2-use650') ...
    };
    checkpointDir = pwd;
    for i = 1:numel(candidateDirs)
        if exist(candidateDirs{i}, 'dir'), checkpointDir = candidateDirs{i}; break; end
    end
    finalModel = fullfile(checkpointDir, 'JaguarTD3DynamicFinal.mat');
    fallbackModel = fullfile(checkpointDir, 'JaguarTD3Final.mat');
    latestCheckpoint = fullfile(checkpointDir, 'JaguarAgentLatest.mat');
    if exist(finalModel, 'file'), modelPath = finalModel;
    elseif exist(fallbackModel, 'file'), modelPath = fallbackModel;
    elseif exist(latestCheckpoint, 'file'), modelPath = latestCheckpoint;
    else, error('No model file found in: %s', checkpointDir);
    end
end
fprintf('Loaded Model: %s\n', modelPath);
data = load(modelPath);
f = fieldnames(data);
agentFound = false;
for i = 1:numel(f)
    if isa(data.(f{i}), 'rl.agent.AbstractAgent') || contains(class(data.(f{i})), 'Agent')
        agent = data.(f{i}); agentFound = true; break;
    end
end
if ~agentFound, agent = data.(f{1}); end
% no exploration noise at test time
agent.AgentOptions.ExplorationModel.Variance = 0;
agent.AgentOptions.ExplorationModel.VarianceMin = 0;
agent.AgentOptions.ExplorationModel.VarianceDecayRate = 0;

% --- dashboard ---
if renderVisualization
    fig = figure('Name', sprintf('DRL Evaluation Dashboard [%s]', upper(controlMode)), ...
                 'Position', [50, 100, 1400, 450]);
end
totalRewards = zeros(numTestEpisodes, 1);
totalSteps = zeros(numTestEpisodes, 1);
goalsReached = zeros(numTestEpisodes, 1);
collisionsOccurred = zeros(numTestEpisodes, 1);
timeoutsOccurred = zeros(numTestEpisodes, 1);
if isempty(cleanPool)
    error('No valid maps found in JaguarMapPool - cannot run evaluation.');
end

% --- rollouts ---
for ep = 1:numTestEpisodes
    mapIdxForEp = mod(ep - 1, numel(cleanPool)) + 1;
    env.MapPool = cleanPool(mapIdxForEp);
    [obs, ~] = reset(env);
    epReward = 0; stepCount = 0; isDone = false;
    wasGoalReached = false; wasCollision = false; wasTimeout = false;
    prevWL = 0.0; prevWR = 0.0;
    xHistoryTrue = []; yHistoryTrue = [];
    xHistorySlam = []; yHistorySlam = [];
    if isprop(env, 'TrueMap') && ~isempty(env.TrueMap)
        xLim = env.TrueMap.XWorldLimits;
        yLim = env.TrueMap.YWorldLimits;
        width = xLim(2) - xLim(1);
        height = yLim(2) - yLim(1);
        slamMap = occupancyMap(width, height, 20);
        slamMap.GridLocationInWorld = [xLim(1), yLim(1)];
    else
        slamMap = occupancyMap(10.0, 10.0, 20);
        slamMap.GridLocationInWorld = [-5.0, -5.0];
    end
    slamPose = env.RobotPose;
    prevScan = [];
    slamPoseVar = initPoseVar(:);
    calculatedDist = Inf;
    while ~isDone && stepCount < maxSteps
        stepCount = stepCount + 1;
        % 1. SLAM: EKF predict from wheel odometry
        encStart = env.LidarNumBeams + 3 + 3 + 1;
        encVals  = env.FrameBuffer(encStart:encStart+3, end); % [wFL; wRL; wFR; wRR] in rad/s
        wL_enc = mean(encVals(1:2));
        wR_enc = mean(encVals(3:4));
        vL_enc = wL_enc * env.WheelRadius;
        vR_enc = wR_enc * env.WheelRadius;
        v_odom = (vL_enc + vR_enc) / 2;
        w_odom = (vR_enc - vL_enc) / env.TrackWidth;
        deltaLocal = [v_odom * dtControl; 0; w_odom * dtControl];
        predictedSlamPose = composePose(slamPose, deltaLocal);
        distMoved = hypot(deltaLocal(1), deltaLocal(2));
        rotMoved  = abs(deltaLocal(3));
        odomVar = [ ...
            (odomTransNoiseStd*distMoved)^2 + odomTransNoiseFloor^2; ...
            (odomTransNoiseStd*distMoved)^2 + odomTransNoiseFloor^2; ...
            (odomRotNoiseStd*rotMoved)^2   + odomRotNoiseFloor^2 ...
        ];
        predictedSlamPoseVar = slamPoseVar + odomVar;
        % scan match against the previous scan
        ranges = env.LastObsRanges(:);
        angles = env.LastObsAngles(:);
        currScan = lidarScan(ranges, angles);
        measuredSlamPose = predictedSlamPose;
        measuredValid = false;
        if ~isempty(prevScan)
            try
                [relPose, ~] = matchScans(currScan, prevScan, 'InitialPose', deltaLocal');
                measuredSlamPose = composePose(slamPose, relPose(:));
                measuredValid = true;
            catch
                measuredValid = false;
            end
        end
        % reject the match if it disagrees too much with odometry
        if measuredValid
            innov = measuredSlamPose - predictedSlamPose;
            innov(3) = atan2(sin(innov(3)), cos(innov(3)));
            if hypot(innov(1), innov(2)) > matchRejectTransThresh || abs(innov(3)) > matchRejectRotThresh
                measuredValid = false;
            end
        end
        % EKF update
        if measuredValid
            measVar = [scanMatchTransNoiseStd^2; scanMatchTransNoiseStd^2; scanMatchRotNoiseStd^2];
            K = predictedSlamPoseVar ./ (predictedSlamPoseVar + measVar);
            slamPose = predictedSlamPose + K .* innov;
            slamPose(3) = atan2(sin(slamPose(3)), cos(slamPose(3)));
            slamPoseVar = (1 - K) .* predictedSlamPoseVar;
        else
            slamPose = predictedSlamPose;
            slamPoseVar = predictedSlamPoseVar;
        end
        % update the SLAM map with the full-resolution scan
        denseRanges = env.LastRawRanges(:);
        denseAngles = env.LastRawAngles(:);
        denseScan = lidarScan(denseRanges, denseAngles);
        margin = 0.02;
        clampedX = max(min(slamPose(1), slamMap.XWorldLimits(2) - margin), slamMap.XWorldLimits(1) + margin);
        clampedY = max(min(slamPose(2), slamMap.YWorldLimits(2) - margin), slamMap.YWorldLimits(1) + margin);
        safeRayPose = [clampedX, clampedY, slamPose(3)];
        insertRay(slamMap, safeRayPose, denseScan, 5.0);
        prevScan = currScan;
        if strcmpi(odomSourceMode, 'slam')
            currentPose = slamPose;
        else
            currentPose = env.RobotPose;
        end
        % overwrite the goal entries of the observation with the estimated ones
        dx = env.GoalPose(1) - currentPose(1);
        dy = env.GoalPose(2) - currentPose(2);
        calculatedDist = sqrt(dx^2 + dy^2);
        phiGoal = atan2(dy, dx);
        rawTheta = phiGoal - currentPose(3);
        calculatedThetaRel = atan2(sin(rawTheta), cos(rawTheta));
        obs(1:2) = [calculatedDist; calculatedThetaRel];
        % 2. policy
        act = getAction(agent, obs);
        if iscell(act), rawAction = act{1}; else, rawAction = act; end
        % 3. action to wheel speeds
        v_L_raw = rawAction(1) * v_max;
        v_R_raw = rawAction(2) * v_max;
        wL_raw = v_L_raw / env.WheelRadius;
        wR_raw = v_R_raw / env.WheelRadius;
        % 4. smoothing, acceleration limit, guardrail
        wL_smooth = alphaFilter * wL_raw + (1.0 - alphaFilter) * prevWL;
        wR_smooth = alphaFilter * wR_raw + (1.0 - alphaFilter) * prevWR;
        maxWAcc = maxLinAcc / env.WheelRadius;
        maxWChange = maxWAcc * dtControl;
        wL_clamped = max(min(wL_smooth, prevWL + maxWChange), prevWL - maxWChange);
        wR_clamped = max(min(wR_smooth, prevWR + maxWChange), prevWR - maxWChange);
        if enableSafetyGuardrail
            frontBeams = env.FrameBuffer(5:9, end);
            minFrontScan = min(frontBeams);
            % split into forward (avg) and turning (diff) parts
            w_avg  = (wL_clamped + wR_clamped) / 2;
            w_diff = (wR_clamped - wL_clamped) / 2;
            if minFrontScan <= dStop
                w_avg_safe = min(w_avg, 0.0); % no forward motion
            elseif minFrontScan < dWarn
                scale = (minFrontScan - dStop) / (dWarn - dStop);
                if w_avg > 0
                    w_avg_safe = w_avg * scale;
                else
                    w_avg_safe = w_avg;
                end
            else
                w_avg_safe = w_avg;
            end
            % put the turning part back so steering is preserved
            wL_safe = w_avg_safe - w_diff;
            wR_safe = w_avg_safe + w_diff;
        else
            wL_safe = wL_clamped;
            wR_safe = wR_clamped;
        end
        prevWL = wL_safe; prevWR = wR_safe;
        action = [max(min(wL_safe / maxWheelSpeed, 1.0), -1.0); ...
                  max(min(wR_safe / maxWheelSpeed, 1.0), -1.0)];
        [nextObs, reward, isDone, loggedSignals] = step(env, action);
        epReward = epReward + reward; obs = nextObs;
        xHistoryTrue(end+1) = env.RobotPose(1); %#ok<SAGROW>
        yHistoryTrue(end+1) = env.RobotPose(2); %#ok<SAGROW>
        xHistorySlam(end+1) = slamPose(1);      %#ok<SAGROW>
        yHistorySlam(end+1) = slamPose(2);      %#ok<SAGROW>
        if isfield(loggedSignals, 'IsGoalReached') && loggedSignals.IsGoalReached, wasGoalReached = true; end
        if isfield(loggedSignals, 'IsCollision') && loggedSignals.IsCollision, wasCollision = true; end
        if isfield(loggedSignals, 'IsNearObstacleTimeout') && loggedSignals.IsNearObstacleTimeout, wasTimeout = true; end
        % 5. live plots
        if renderVisualization && isvalid(fig)
            set(0, 'CurrentFigure', fig);
            if enableSlamMapVis
                subplot(1, 3, 1);
                show(env.TrueMap); hold on;
                plot(xHistoryTrue, yHistoryTrue, 'g-', 'LineWidth', 2);
                plot(env.RobotPose(1), env.RobotPose(2), 'go', 'MarkerSize', 8, 'MarkerFaceColor', 'g');
                plot(env.GoalPose(1), env.GoalPose(2), 'rp', 'MarkerSize', 14, 'MarkerFaceColor', 'r');
                quiver(env.RobotPose(1), env.RobotPose(2), cos(env.RobotPose(3)), sin(env.RobotPose(3)), 0.6, 'g', 'LineWidth', 2);
                hold off;
                title(sprintf('True Map (Ep %d)', ep));
                grid on;
                subplot(1, 3, 2);
                show(slamMap); hold on;
                plot(xHistorySlam, yHistorySlam, 'b-', 'LineWidth', 2);
                plot(slamPose(1), slamPose(2), 'ro', 'MarkerSize', 8, 'MarkerFaceColor', 'r');
                plot(env.GoalPose(1), env.GoalPose(2), 'gp', 'MarkerSize', 14, 'MarkerFaceColor', 'g');
                quiver(slamPose(1), slamPose(2), cos(slamPose(3)), sin(slamPose(3)), 0.6, 'r', 'LineWidth', 2);
                hold off;
                title(sprintf('Detected SLAM Map (Ep %d)', ep));
                grid on;
                subplot(1, 3, 3);
                polarplot(angles, ranges, 'r.-', 'LineWidth', 1.5);
                rlim([0, 6]);
                title(sprintf('LiDAR Profile | Reward: %.2f', epReward));
            else
                subplot(1, 2, 1);
                show(env.TrueMap); hold on;
                plot(xHistoryTrue, yHistoryTrue, 'g-', 'LineWidth', 2);
                plot(env.RobotPose(1), env.RobotPose(2), 'go', 'MarkerSize', 8, 'MarkerFaceColor', 'g');
                plot(env.GoalPose(1), env.GoalPose(2), 'rp', 'MarkerSize', 14, 'MarkerFaceColor', 'r');
                quiver(env.RobotPose(1), env.RobotPose(2), cos(env.RobotPose(3)), sin(env.RobotPose(3)), 0.6, 'g', 'LineWidth', 2);
                hold off;
                title(sprintf('True Map (Ep %d)', ep));
                grid on;
                subplot(1, 2, 2);
                polarplot(angles, ranges, 'r.-', 'LineWidth', 1.5);
                rlim([0, 6]);
                title(sprintf('LiDAR Profile | Reward: %.2f', epReward));
            end
            drawnow;
        end
    end
    if calculatedDist < 0.2, wasGoalReached = true; end
    goalsReached(ep)       = wasGoalReached;
    collisionsOccurred(ep) = wasCollision;
    timeoutsOccurred(ep)   = wasTimeout;
    totalRewards(ep)       = epReward;
    totalSteps(ep)         = stepCount;
end

% --- summary ---
fprintf('\n================ EVALUATION SUMMARY ================\n');
fprintf('Trained Model File : %s\n', modelPath);
fprintf('Control Mode       : %s\n', upper(controlMode));
fprintf('Odometry Mode      : %s\n', upper(odomSourceMode));
fprintf('Safety Guardrail   : %s\n', string(enableSafetyGuardrail));
fprintf('Terrain Noise      : %s\n', string(enableSensorNoise));
fprintf('Episodes Evaluated : %d\n', numTestEpisodes);
fprintf('Success Rate       : %.1f%%\n', (sum(goalsReached) / numTestEpisodes) * 100);
fprintf('Collision Rate     : %.1f%%\n', (sum(collisionsOccurred) / numTestEpisodes) * 100);
fprintf('Timeout Rate       : %.1f%%\n', (sum(timeoutsOccurred) / numTestEpisodes) * 100);
fprintf('Average Reward     : %.2f\n', mean(totalRewards));
fprintf('Average Steps      : %.1f\n', mean(totalSteps));
fprintf('====================================================\n');

% --- local functions ---
function pose = composePose(basePose, localDelta)
    c = cos(basePose(3)); s = sin(basePose(3));
    pose = zeros(3,1);
    pose(1) = basePose(1) + localDelta(1)*c - localDelta(2)*s;
    pose(2) = basePose(2) + localDelta(1)*s + localDelta(2)*c;
    pose(3) = atan2(sin(basePose(3) + localDelta(3)), cos(basePose(3) + localDelta(3)));
end
