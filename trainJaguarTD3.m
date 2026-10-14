clearvars -except agent; clc;

% stopping criteria
targetConsecutiveEpisodes = 50;        % episodes in a row that must clear the threshold
goalNoCollisionRewardThreshold = 200;  % reward each of those episodes must reach

% hardware / replay memory
experienceBufferLength = 100000;
numParallelWorkers = 5;

% training length
maxEpisodes = 8000;
maxStepsPerEpisode = 450;

% noise and safety
noiseMode = 'low';                % 'off', 'low', 'standard'
enableSafetyBuffer = true;
forceExplorationReset = true;     % reset noise params when loading a checkpoint

% noise decay
targetDecayEpisodes = 1500;       % decay noise over the first 1500 episodes
estimatedAvgStepsPerEp = 150;     % rough average, since many episodes end early
totalDecaySteps = targetDecayEpisodes * estimatedAvgStepsPerEp;

switch lower(string(noiseMode))
    case {'none', 'off', '0'}
        targetVariance    = 0.0;
        targetVarianceMin = 0.0;
        fprintf('=== NOISE MODE: Deterministic / Zero Noise ===\n');

    case {'low', '1'}
        targetVariance    = 0.05;
        targetVarianceMin = 0.01;
        fprintf('=== NOISE MODE: Low Exploration (Fine-Tuning) ===\n');

    case {'standard', 'normal', '2'}
        targetVariance    = 0.30;
        targetVarianceMin = 0.01;
        fprintf('=== NOISE MODE: Standard Exploration with Dynamic Decay ===\n');

    otherwise
        error("Invalid noiseMode specified.");
end

if targetVariance > targetVarianceMin
    varianceDecayRate = (targetVariance - targetVarianceMin) / totalDecaySteps;
else
    varianceDecayRate = 0;
end

% --- parallel pool ---
close all force;
set(0, 'DefaultFigureVisible', 'off');
pool = gcp('nocreate');
if isempty(pool)
    pool = parpool(numParallelWorkers);
elseif pool.NumWorkers ~= numParallelWorkers
    delete(pool);
    pool = parpool(numParallelWorkers);
end
pctRunOnAll('set(0, ''DefaultFigureVisible'', ''off'');');

addAttachedFiles(pool, {'JaguarSLAMEnv_DynamicGoal.m'});

% --- environment ---
env = JaguarSLAMEnv_DynamicGoal();
env.MapMode = 'random';           % 'random' or 'mixed' for generated maps
env.UseSafetyBuffer = enableSafetyBuffer;
env = env.configureEnvironment();
env.MaxSteps = maxStepsPerEpisode;
obsInfo = getObservationInfo(env);
actInfo = getActionInfo(env);
obsDim = obsInfo.Dimension(1);
fprintf('=== IMPLICIT LOCALIZATION: Dynamic Body-Frame Goal Mode (100-D Input) ===\n');

% --- checkpoint folder ---
checkpointDir = fullfile(pwd, 'lownoise-randMap2-contNoise2');
if ~exist(checkpointDir, 'dir')
    mkdir(checkpointDir);
end

targetFilename = 'JaguarAgentLatest.mat';
checkpointFile = '';
if isfile(fullfile(checkpointDir, targetFilename))
    checkpointFile = fullfile(checkpointDir, targetFilename);
end

if ~isempty(checkpointFile)
    fprintf('Resuming training from checkpoint: %s\n', checkpointFile);
    data = load(checkpointFile);
    f = fieldnames(data);
    agent = data.(f{1});

    if forceExplorationReset
        agent.AgentOptions.ExplorationModel.Variance = targetVariance;
        agent.AgentOptions.ExplorationModel.VarianceDecayRate = varianceDecayRate;
        agent.AgentOptions.ExplorationModel.VarianceMin = targetVarianceMin;
    end
else
    fprintf('Building new TD3 Agent for Dynamic Goal Navigation...\n');

    % per-frame normalization: 13 lidar, 3 accel, 3 gyro, 4 encoders, [dGoal, thetaGoal]
    singleMean = [5*ones(1,13), 0, 0, 9.81, 0, 0, 0, zeros(1,4), 2.0, 0.0];
    singleStd  = [3*ones(1,13), 0.1, 0.1, 0.1, 0.15, 0.15, 1.0, 8*ones(1,4), 1.2, 1.0];

    frameObsMean = repmat(singleMean, 1, 4);
    frameObsStd  = repmat(singleStd, 1, 4);

    % actor
    actorPath = [
        featureInputLayer(obsDim, 'Normalization', 'zscore', 'Mean', frameObsMean, 'StandardDeviation', frameObsStd, 'Name', 'obs')
        fullyConnectedLayer(256, 'Name', 'fc1')
        reluLayer('Name', 'relu1')
        fullyConnectedLayer(256, 'Name', 'fc2')
        reluLayer('Name', 'relu2')
        fullyConnectedLayer(actInfo.Dimension(1), 'Name', 'out')
        tanhLayer('Name', 'tanh')
        scalingLayer('Name', 'action', 'Scale', actInfo.UpperLimit)
    ];
    actorNet = dlnetwork(actorPath);
    actor = rlContinuousDeterministicActor(actorNet, obsInfo, actInfo);

    % twin critics
    critic1Net = buildCriticNetwork(obsDim, actInfo.Dimension(1), frameObsMean, frameObsStd);
    critic2Net = buildCriticNetwork(obsDim, actInfo.Dimension(1), frameObsMean, frameObsStd);

    critic1 = rlQValueFunction(critic1Net, obsInfo, actInfo, 'ObservationInputNames', 'obs', 'ActionInputNames', 'act');
    critic2 = rlQValueFunction(critic2Net, obsInfo, actInfo, 'ObservationInputNames', 'obs', 'ActionInputNames', 'act');

    actorOpts  = rlOptimizerOptions('LearnRate', 1e-3);
    criticOpts = rlOptimizerOptions('LearnRate', 1e-3);

    agentOptions = rlTD3AgentOptions(...
        'SampleTime', 0.1, ...
        'TargetSmoothFactor', 1e-3, ...
        'DiscountFactor', 0.99, ...
        'MiniBatchSize', 256, ...
        'ExperienceBufferLength', experienceBufferLength, ...
        'ActorOptimizerOptions', actorOpts, ...
        'CriticOptimizerOptions', criticOpts);

    agentOptions.ExplorationModel.Variance = targetVariance;
    agentOptions.ExplorationModel.VarianceDecayRate = varianceDecayRate;
    agentOptions.ExplorationModel.VarianceMin = targetVarianceMin;

    agent = rlTD3Agent(actor, [critic1, critic2], agentOptions);
end

% --- train ---
set(0, 'DefaultFigureVisible', 'on');
trainOptions = rlTrainingOptions(...
    'MaxEpisodes', maxEpisodes, ...
    'MaxStepsPerEpisode', maxStepsPerEpisode, ...
    'ScoreAveragingWindowLength', 50, ...
    'Verbose', true, ...
    'Plots', 'training-progress', ...
    'StopTrainingCriteria', 'Custom', ...
    'StopTrainingValue', @(stats) numel(stats.EpisodeReward) >= targetConsecutiveEpisodes && ...
    all(stats.EpisodeReward(end-(targetConsecutiveEpisodes-1):end) >= goalNoCollisionRewardThreshold), ...
    'SaveAgentCriteria', 'EpisodeFrequency', ...
    'SaveAgentValue', 50, ...
    'SaveAgentDirectory', checkpointDir, ...
    'UseParallel', true);

trainingStats = train(agent, env, trainOptions);

% --- save ---
save(fullfile(checkpointDir, 'JaguarTD3DynamicFinal.mat'), 'agent');
save(fullfile(checkpointDir, 'JaguarAgentLatest.mat'), 'agent');

function criticNet = buildCriticNetwork(obsDim, actDim, frameObsMean, frameObsStd)
    obsPath = [
        featureInputLayer(obsDim, 'Normalization', 'zscore', 'Mean', frameObsMean, 'StandardDeviation', frameObsStd, 'Name', 'obs')
        fullyConnectedLayer(128, 'Name', 'fcObs')
    ];
    actPath = [
        featureInputLayer(actDim, 'Name', 'act')
        fullyConnectedLayer(128, 'Name', 'fcAct')
    ];
    commonPath = [
        additionLayer(2, 'Name', 'add')
        reluLayer('Name', 'relu1')
        fullyConnectedLayer(256, 'Name', 'fc1')
        reluLayer('Name', 'relu2')
        fullyConnectedLayer(1, 'Name', 'qval')
    ];

    lg = layerGraph();
    lg = addLayers(lg, obsPath);
    lg = addLayers(lg, actPath);
    lg = addLayers(lg, commonPath);
    lg = connectLayers(lg, 'fcObs', 'add/in1');
    lg = connectLayers(lg, 'fcAct', 'add/in2');
    criticNet = dlnetwork(lg);
end
