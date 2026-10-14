classdef JaguarSLAMEnv_DynamicGoal < rl.env.MATLABEnvironment
    properties
        % robot dimensions
        WheelRadius = 0.13;        % m
        WheelBase = 0.36;          % m
        TrackWidth = 0.47;         % m
        LidarForwardOffset = 0.26; % lidar mount distance from robot center (m)
        % robot state
        RobotPose = [0; 0; 0];     % ground truth [x; y; yaw]
        Velocities = [0; 0];       % [v; w]
        TrueMap                    % ground-truth occupancy map
        MapPool                    % list of map structs
        % map selection
        MapMode = 'fixed';         % 'fixed', 'mixed', 'random', 'designed'
        GeneratedMapProbability = 0.7; % chance of a generated map in 'mixed' mode
        % world size
        WorldSize = 6;             % m
        WallMargin = 0.2;          % out-of-bounds distance from the edge (m)
        % sensor and step counters
        LidarSensor
        CurrentStep = 0;
        MaxSteps = 450;
        StagnantSteps = 0;         % consecutive low-speed steps
        % goal tracking
        GoalPose = [4.5; 4.5];
        DynamicRelativeGoal = [0; 0];    % [distance; bearing] in the body frame
        PrevDistToGoal = 0;
        ClosestDistToGoal = Inf;
        % noise and drift
        PeriodicNoiseFreq = 2.0;   % Hz
        PeriodicNoiseAmp  = 0.05;
        EncoderPeriodicAmp = 0.05;
        GyroBias = [0; 0; 0];      % rad/s
        GyroBiasWalkStd = 0.002;
        % lidar
        % The sensor always traces 181 beams. LidarNumBeams (13) of them are
        % picked out for the observation so the 25-D frame size stays fixed.
        LidarNumBeams = 13;
        LidarRawNumBeams = 181;
        LidarFOV = pi;
        LidarBeamSelectIdx         % indices of the 13 chosen beams
        LastRawRanges              % latest 181-beam scan
        LastRawAngles
        LastObsRanges              % latest 13-beam scan used in the observation
        LastObsAngles
        % collision
        UseSafetyBuffer = true;
        CollisionDistanceThreshold = 0.20; % m
        % loitering timeout
        NearObstacleSteps = 0;
        NearObstacleStepLimit = 50;
        NearObstacleDistThreshold = 0.25;  % m
        % observation stacking
        FrameStackSize = 4;
        SingleFrameDim = 25;       % 13 lidar + 3 accel + 3 gyro + 4 encoders + 2 goal
        FrameBuffer                % [25 x 4]
        % pose estimate bookkeeping
        EstimatedPose = [0; 0; 0];
        PrevScanRanges
        PrevScanAngles
        ScanMatchMaxTranslation = 1.0; % m
        ScanMatchMaxRotation = 0.6;    % rad
    end

    methods
        function obj = JaguarSLAMEnv_DynamicGoal()
            singleDim = 25;
            stackSize = 4;
            obsDim = singleDim * stackSize; % 100
            obsInfo = rlNumericSpec([obsDim, 1]);
            actInfo = rlNumericSpec([2, 1], 'LowerLimit', [-1; -1], 'UpperLimit', [1; 1]);
            obj = obj@rl.env.MATLABEnvironment(obsInfo, actInfo);
            obj.SingleFrameDim = singleDim;
            obj.FrameStackSize = stackSize;
            obj.MapPool = obj.getMapPool();
        end

        function obj = configureEnvironment(obj)
            obj.CollisionDistanceThreshold = 0.20;
        end

        % default map pool, with a small fallback if JaguarMapPool.m is missing
        function pool = getMapPool(~)
            if exist('JaguarMapPool', 'file') == 2
                pool = JaguarMapPool();
            else
                map1.StartPose = [1.5; 1.5; 0];
                map1.Obstacles = [3.0, 3.0, 1.5, 2.0; 6.0, 2.0, 1.0, 3.5];
                map2.StartPose = [1.5; 1.5; 0];
                map2.Obstacles = [2.5, 5.0, 3.0, 1.0; 6.5, 5.0, 1.5, 3.0];
                pool = {map1, map2};
            end
        end

        % maps saved under JaguarDesignedMaps/
        function pool = getDesignedMapPool(~)
            pool = {};
            designedFolder = fullfile(pwd, 'JaguarDesignedMaps');
            if exist(designedFolder, 'dir')
                designedFiles = dir(fullfile(designedFolder, 'map_*.mat'));
                for i = 1:numel(designedFiles)
                    loaded = load(fullfile(designedFiles(i).folder, designedFiles(i).name), 'mapDef');
                    pool{end+1} = loaded.mapDef; %#ok<AGROW>
                end
            end
        end

        % grid of points covering a rectangle [x, y, w, h]
        function pts = rectanglePoints(~, rect, resolution)
            step = 1 / resolution;
            xs = rect(1):step:(rect(1) + rect(3));
            ys = rect(2):step:(rect(2) + rect(4));
            [X, Y] = meshgrid(xs, ys);
            pts = [X(:), Y(:)];
        end

        % lidar is mounted ahead of the robot center
        function lidarPose = computeLidarPoseFromPose(obj, pose)
            lidarMountOffset = obj.WheelBase / 2 + obj.LidarForwardOffset;
            rawX = pose(1) + lidarMountOffset * cos(pose(3));
            rawY = pose(2) + lidarMountOffset * sin(pose(3));
            clampedX = min(max(rawX, 0.05), obj.WorldSize - 0.05);
            clampedY = min(max(rawY, 0.05), obj.WorldSize - 0.05);
            lidarPose = [clampedX, clampedY, pose(3)];
        end

        % random rectangles, kept away from the start pose
        function mapDef = generateRandomMap(obj)
            startPose = [1.0; 1.0; 0];
            clearanceRadius = 1.0;
            numObstacles = randi([2, 5]);
            obstacles = zeros(numObstacles, 4);
            placed = 0;
            attempts = 0;
            maxAttempts = 200;
            maxDim = min(1.5, obj.WorldSize / 3);
            while placed < numObstacles && attempts < maxAttempts
                attempts = attempts + 1;
                w = 0.5 + (maxDim - 0.5) * rand();
                h = 0.5 + (maxDim - 0.5) * rand();
                x = 0.5 + (obj.WorldSize - 1.0 - w) * rand();
                y = 0.5 + (obj.WorldSize - 1.0 - h) * rand();
                tooClose = startPose(1) >= (x - clearanceRadius) && ...
                           startPose(1) <= (x + w + clearanceRadius) && ...
                           startPose(2) >= (y - clearanceRadius) && ...
                           startPose(2) <= (y + h + clearanceRadius);
                if tooClose
                    continue;
                end
                placed = placed + 1;
                obstacles(placed, :) = [x, y, w, h];
            end
            mapDef.Obstacles = obstacles(1:placed, :);
            mapDef.StartPose = startPose;
        end

        % builds one 25-D frame and refreshes DynamicRelativeGoal
        function rawFrame = buildRawFrame(obj, ranges, accel_body, gyro_body, encoder_vals)
            dx = obj.GoalPose(1) - obj.RobotPose(1);
            dy = obj.GoalPose(2) - obj.RobotPose(2);
            dGoal = sqrt(dx^2 + dy^2);
            angleToGoalWorld = atan2(dy, dx);
            thetaGoal = atan2(sin(angleToGoalWorld - obj.RobotPose(3)), cos(angleToGoalWorld - obj.RobotPose(3)));
            obj.DynamicRelativeGoal = [dGoal; thetaGoal];
            rawFrame = [ranges(:); accel_body; gyro_body; encoder_vals; dGoal; thetaGoal];
            rawFrame(isnan(rawFrame)) = 6.0; % NaN means no return, treat as max range
        end

        function rwd = calculateReward(obj, currentDistToGoal, ranges)
            if exist('calculateJaguarReward', 'file') == 2
                rwd = calculateJaguarReward(obj, currentDistToGoal, ranges);
            else
                % simple fallback reward
                distReward = (obj.PrevDistToGoal - currentDistToGoal) * 100;
                minLidar = min(ranges);
                collisionPenalty = 0;
                if minLidar < obj.CollisionDistanceThreshold
                    collisionPenalty = -500;
                end
                goalBonus = 0;
                if currentDistToGoal < 0.2
                    goalBonus = 500;
                end
                rwd = distReward + collisionPenalty + goalBonus - 0.1;
            end
        end

        function [nextObs, reward, isDone, loggedSignals] = step(obj, action)
            obj.CurrentStep = obj.CurrentStep + 1;
            % actions are normalized, scale to +/-15 rad/s wheel speed
            maxWheelSpeed = 15.0;
            w_Left = action(1) * maxWheelSpeed;
            w_Right = action(2) * maxWheelSpeed;
            % differential drive with random slip on each side
            v_L_wheel = w_Left * obj.WheelRadius;
            v_R_wheel = w_Right * obj.WheelRadius;
            slip_L = 0.20 * rand();
            slip_R = 0.20 * rand();
            v_L_ground = v_L_wheel * (1 - slip_L);
            v_R_ground = v_R_wheel * (1 - slip_R);
            v = (v_L_ground + v_R_ground) / 2;
            w = (v_R_ground - v_L_ground) / obj.TrackWidth;
            obj.Velocities = [v; w];
            % Euler integration of the true pose
            dt = 0.1;
            obj.RobotPose(1) = obj.RobotPose(1) + v * cos(obj.RobotPose(3)) * dt;
            obj.RobotPose(2) = obj.RobotPose(2) + v * sin(obj.RobotPose(3)) * dt;
            obj.RobotPose(3) = obj.RobotPose(3) + w * dt;
            if abs(v) < 0.05
                obj.StagnantSteps = obj.StagnantSteps + 1;
            else
                obj.StagnantSteps = 0;
            end
            outOfBounds = obj.RobotPose(1) <= obj.WallMargin || obj.RobotPose(1) >= (obj.WorldSize - obj.WallMargin) || ...
                          obj.RobotPose(2) <= obj.WallMargin || obj.RobotPose(2) >= (obj.WorldSize - obj.WallMargin);
            % full 181-beam scan
            lidarPose = obj.computeLidarPoseFromPose(obj.RobotPose);
            [rawRanges, rawAngles] = obj.LidarSensor(lidarPose, obj.TrueMap);
            obj.LastRawRanges = rawRanges;
            obj.LastRawAngles = rawAngles;
            % pick the 13 observation beams
            ranges = rawRanges(obj.LidarBeamSelectIdx);
            angles = rawAngles(obj.LidarBeamSelectIdx);
            obj.LastObsRanges = ranges;
            obj.LastObsAngles = angles;
            minRangeStep = min(ranges);
            if minRangeStep < obj.NearObstacleDistThreshold
                obj.NearObstacleSteps = obj.NearObstacleSteps + 1;
            else
                obj.NearObstacleSteps = 0;
            end
            % IMU: periodic vibration plus gyro bias random walk
            t = obj.CurrentStep * dt;
            accelPeriodicNoise = obj.PeriodicNoiseAmp * sin(2 * pi * obj.PeriodicNoiseFreq * t) * [1; 1; 1];
            gyroPeriodicNoise  = obj.PeriodicNoiseAmp * cos(2 * pi * obj.PeriodicNoiseFreq * t) * [1; 1; 1];
            obj.GyroBias = obj.GyroBias + obj.GyroBiasWalkStd * randn(3,1);
            accel_body = [0; 0; 9.81] + accelPeriodicNoise;
            gyro_body  = [0; 0; w] + gyroPeriodicNoise + obj.GyroBias;
            % encoders with periodic slip
            slip = 1 + obj.EncoderPeriodicAmp * sin(2 * pi * obj.PeriodicNoiseFreq * t + [0; pi/2; pi; 3*pi/2]);
            encoder_vals = [w_Left; w_Left; w_Right; w_Right] .* slip;
            % newest frame goes at the end of the stack
            rawFrame = obj.buildRawFrame(ranges, accel_body, gyro_body, encoder_vals);
            obj.FrameBuffer = [obj.FrameBuffer(:, 2:end), rawFrame];
            nextObs = obj.FrameBuffer(:);
            trueDistToGoal = obj.DynamicRelativeGoal(1);
            reward = obj.calculateReward(trueDistToGoal, ranges);
            obj.PrevDistToGoal = trueDistToGoal;
            reachedGoal = trueDistToGoal < 0.2;
            stagnantTimeout = (obj.StagnantSteps >= 50);
            nearObstacleTimeout = (obj.NearObstacleSteps >= obj.NearObstacleStepLimit);
            hasCollided = (min(ranges) < obj.CollisionDistanceThreshold) || outOfBounds;
            isDone = logical(obj.CurrentStep >= obj.MaxSteps || ...
                             hasCollided || ...
                             reachedGoal || ...
                             stagnantTimeout || ...
                             nearObstacleTimeout);
            loggedSignals = struct();
            loggedSignals.IsGoalReached = reachedGoal;
            loggedSignals.IsCollision = hasCollided;
            loggedSignals.IsNearObstacleTimeout = nearObstacleTimeout;
        end

        function [InitialObservation, LoggedSignals] = reset(obj)
            obj.MapPool = obj.getMapPool();
            % pick a map
            switch lower(obj.MapMode)
                case 'fixed'
                    useGenerated = false;
                    selectedPool = obj.MapPool;
                case 'random'
                    useGenerated = true;
                    selectedPool = {};
                case 'mixed'
                    useGenerated = (rand() < obj.GeneratedMapProbability);
                    selectedPool = obj.MapPool;
                case 'designed'
                    useGenerated = false;
                    selectedPool = obj.getDesignedMapPool();
                otherwise
                    error('Invalid MapMode.');
            end
            if useGenerated
                mapDef = obj.generateRandomMap();
            else
                mapIdx = randi(numel(selectedPool));
                mapDef = selectedPool{mapIdx};
            end
            % reset state
            obj.RobotPose = mapDef.StartPose;
            obj.Velocities = [0; 0];
            obj.CurrentStep = 0;
            obj.StagnantSteps = 0;
            obj.NearObstacleSteps = 0;
            obj.GyroBias = [0; 0; 0];
            obj.EstimatedPose = mapDef.StartPose;
            obj.PrevScanRanges = [];
            obj.PrevScanAngles = [];
            % occupancy map from walls + obstacles
            obj.TrueMap = binaryOccupancyMap(obj.WorldSize, obj.WorldSize, 20);
            ws = obj.WorldSize;
            wallThickness = 0.2;
            outerWalls = [0, 0, ws, wallThickness; ...
                          0, ws - wallThickness, ws, wallThickness; ...
                          0, 0, wallThickness, ws; ...
                          ws - wallThickness, 0, wallThickness, ws];
            allObstacles = [outerWalls; mapDef.Obstacles];
            for i = 1:size(allObstacles, 1)
                % sample 5x finer than the map cells so floating-point
                % drift can't skip a cell and leave a gap in a wall
                pts = obj.rectanglePoints(allObstacles(i, :), obj.TrueMap.Resolution * 5);
                setOccupancy(obj.TrueMap, pts, 1);
            end
            % pick a goal that is free and far enough from the start
            minGoalSeparation = min(1.5, obj.WorldSize / 4);
            while true
                gx = 0.75 + (obj.WorldSize - 1.5) * rand();
                gy = 0.75 + (obj.WorldSize - 1.5) * rand();
                if norm([gx; gy] - obj.RobotPose(1:2)) >= minGoalSeparation && getOccupancy(obj.TrueMap, [gx, gy]) < 0.5
                    obj.GoalPose = [gx; gy];
                    break;
                end
            end
            % lidar at full 181-beam resolution
            angularRes = obj.LidarFOV / (obj.LidarRawNumBeams - 1);
            obj.LidarSensor = rangeSensor('Range', [0.1, 6.0], ...
                                          'HorizontalAngle', [-obj.LidarFOV/2, obj.LidarFOV/2], ...
                                          'HorizontalAngleResolution', angularRes);
            % 13 evenly spaced beams out of the 181, same as in training
            obj.LidarBeamSelectIdx = round(linspace(1, obj.LidarRawNumBeams, obj.LidarNumBeams));
            % first scan
            lidarPose = obj.computeLidarPoseFromPose(obj.RobotPose);
            [rawRanges, rawAngles] = obj.LidarSensor(lidarPose, obj.TrueMap);
            obj.LastRawRanges = rawRanges;
            obj.LastRawAngles = rawAngles;
            ranges = rawRanges(obj.LidarBeamSelectIdx);
            angles = rawAngles(obj.LidarBeamSelectIdx);
            obj.LastObsRanges = ranges;
            obj.LastObsAngles = angles;
            obj.PrevScanRanges = ranges;
            obj.PrevScanAngles = angles;
            accel_body = [0; 0; 9.81];
            gyro_body  = [0; 0; 0];
            encoder_vals = zeros(4,1);
            % fill the whole stack with the first frame
            rawFrame = obj.buildRawFrame(ranges, accel_body, gyro_body, encoder_vals);
            obj.PrevDistToGoal = obj.DynamicRelativeGoal(1);
            obj.ClosestDistToGoal = obj.DynamicRelativeGoal(1);
            obj.FrameBuffer = repmat(rawFrame, 1, obj.FrameStackSize);
            InitialObservation = obj.FrameBuffer(:);
            LoggedSignals = struct();
        end
    end
end
