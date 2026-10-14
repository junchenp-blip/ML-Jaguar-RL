function rwd = calculateJaguarReward(env, currentDistToGoal, ranges)
    v = env.Velocities(1);
    w = env.Velocities(2);
    minRange = min(ranges);

    % heading error to the goal, wrapped to [-pi, pi]
    dx = env.GoalPose(1) - env.RobotPose(1);
    dy = env.GoalPose(2) - env.RobotPose(2);
    angleToGoal = atan2(dy, dx) - env.RobotPose(3);
    angleToGoal = atan2(sin(angleToGoal), cos(angleToGoal));

    % progress toward goal
    r_progress = 18.0 * (env.PrevDistToGoal - currentDistToGoal);

    % bonus every 0.5 m of new progress past the closest point so far
    r_checkpoint = 0;
    checkpointInterval = 0.5;
    progressSinceClosest = env.ClosestDistToGoal - currentDistToGoal;
    if progressSinceClosest >= checkpointInterval
        checkpointsPassed = floor(progressSinceClosest / checkpointInterval);
        r_checkpoint = 20.0 * checkpointsPassed;
        env.ClosestDistToGoal = env.ClosestDistToGoal - checkpointsPassed * checkpointInterval;
    end

    % per-step cost
    r_time = -0.1;

    % reward moving forward while facing the goal
    r_heading = 0;
    if abs(v) >= 0.08
        r_heading = 0.3 * cos(angleToGoal);
    end
    r_forward = 3.0 * v * cos(angleToGoal);

    % discourage sitting still or spinning in place
    r_idle = 0;
    if abs(v) < 0.08
        r_idle = -0.5;
    end

    r_spin = 0;
    if abs(v) < 0.08 && abs(w) > 0.5
        r_spin = -0.5 * abs(w);
    end

    % goal bonus and turning cost
    r_goal = 0;
    if currentDistToGoal < 0.2
        r_goal = 400.0;
    end

    r_turn = -0.1 * abs(w);

    % stay away from obstacles (stricter when the safety buffer is on)
    r_obstacle = 0;
    if isprop(env, 'UseSafetyBuffer') && env.UseSafetyBuffer
        obstacleThreshold = 0.8;
        obstacleWeight    = 5.0;
    else
        obstacleThreshold = 0.6;
        obstacleWeight    = 2.0;
    end
    if minRange < obstacleThreshold
        r_obstacle = -obstacleWeight * (obstacleThreshold - minRange);
    end

    % collision penalty
    collisionDist = 0.20;
    if isprop(env, 'CollisionDistanceThreshold')
        collisionDist = env.CollisionDistanceThreshold;
    end
    r_collision = 0;
    if minRange < collisionDist
        r_collision = -100.0;
    end

    rwd = r_progress + r_checkpoint + r_time + r_heading + r_forward ...
        + r_idle + r_spin + r_goal + r_turn + r_obstacle + r_collision;
end
