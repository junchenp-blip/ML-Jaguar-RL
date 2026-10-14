function maps = JaguarMapPool()
    maps = {};

    % outer walls, each row is [x, y, width, height]
    outerWalls = [
        0.0, 0.0, 6.0, 0.30;  % bottom
        0.0, 5.7, 6.0, 0.30;  % top
        0.0, 0.0, 0.30, 6.0;  % left
        5.7, 0.0, 0.30, 6.0   % right
        ];

    % map 1: single block in the middle
    map1 = struct();
    map1.Obstacles = [
        outerWalls;
        2.5, 2.5, 1.0, 1.0
        ];
    map1.StartPose = [1.0; 1.0; 0];
    maps{end+1} = map1;

    % map 2: L-shaped divider plus a wall segment
    map2 = struct();
    map2.Obstacles = [
        outerWalls;
        1.5, 1.5, 2.0, 0.5;
        1.5, 1.5, 0.5, 2.0;
        4.0, 3.5, 1.5, 0.5
        ];
    map2.StartPose = [0.75; 0.75; 0];
    maps{end+1} = map2;

    % map 3: four pillars
    map3 = struct();
    map3.Obstacles = [
        outerWalls;
        1.5, 1.5, 0.75, 0.75;
        3.5, 1.5, 0.75, 0.75;
        1.5, 3.5, 0.75, 0.75;
        3.75, 3.75, 0.75, 0.75
        ];
    map3.StartPose = [0.75; 3.0; 0];
    maps{end+1} = map3;

    % append any hand-designed maps saved in JaguarDesignedMaps/
    designedFolder = fullfile(pwd, 'JaguarDesignedMaps');
    if exist(designedFolder, 'dir')
        designedFiles = dir(fullfile(designedFolder, 'map_*.mat'));
        for i = 1:numel(designedFiles)
            loaded = load(fullfile(designedFiles(i).folder, designedFiles(i).name), 'mapDef');
            if isfield(loaded, 'mapDef')
                mapDef = loaded.mapDef;
                if isfield(mapDef, 'Obstacles')
                    mapDef.Obstacles = [outerWalls; mapDef.Obstacles];
                else
                    mapDef.Obstacles = outerWalls;
                end
                maps{end+1} = mapDef; %#ok<AGROW>
            end
        end
    end
end
