%% SCRIPT TO PROCESS the STB dataset by Zhang et al., ‘3d Hand Pose Tracking and Estimation Using Stereo Matching’, 2016 into binary format

% SET THIS to where you have the dataset
PATH_TO_DATASET = '/misc/lmbraid19/zimmermc/datasets/StereoHandTracking/';

%% no more changes below %%

sequences = {'B1Counting', 'B1Random'};
cam = 'BB'; % thats the camera the annotations are in
% cam = 'SK';

% chose apropriate K matrix
if strcmp(cam, 'BB')
    fx = 822.79041;
    fy = 822.79041;
    tx = 318.47345;
    ty = 250.31296;
    base = 120.054;

    R_l = zeros(3, 4);
    R_l(1, 1) = 1;
    R_l(2, 2) = 1;
    R_l(3, 3) = 1;
    R_r = R_l;
    R_r(1, 4) = -base;

    K = diag([fx, fy, 1.0]);
    K(1, 3) = tx;
    K(2, 3) = ty;
else
    fx = 607.92271;
    fy = 607.88192;
    tx = 314.78337;
    ty = 236.42484;
    K = diag([fx, fy, 1]);
    K(1, 3) = tx;
    K(2, 3) = ty;
    K = [K zeros(3, 1)];
end

finger_ind = 1:21;


% open binary file
file = fopen(sprintf('./stb_eval.bin'), 'w');
for seq=sequences
    seq_name = seq{1};
    fprintf('Working on %s\n', seq_name)

    % load annotation file
    load(sprintf('%s/labels/%s_%s.mat', PATH_TO_DATASET, seq_name, cam), 'handPara');

    for im_id=0:1499
        % path to images
        img_path_left = sprintf('%s/%s/%s_%s_%d.png', PATH_TO_DATASET, seq_name, cam, 'left', im_id);
        img_path_right = sprintf('%s/%s/%s_%s_%d.png', PATH_TO_DATASET, seq_name, cam, 'right', im_id);

        % load image
        try
            img_l = imread(img_path_left);
            img_r = imread(img_path_right);
        catch
            fprintf('Skipped one file: %s\n', img_path_left)
        end

        % get corresponding annotations
        anno_xyz_l = handPara(:, :, im_id+1);

        % left frame
        anno_uv_l = K * R_l * [anno_xyz_l; ones(1, 21)];
        for k=1:21
            anno_uv_l(:, k) = anno_uv_l(:, k) ./ anno_uv_l(3, k);
        end

        % right frame
        anno_xyz_r = R_r * [anno_xyz_l; ones(1, 21)];
        anno_uv_r = K * anno_xyz_r;
        for k=1:21
            anno_uv_r(:, k) = anno_uv_r(:, k) ./ anno_uv_r(3, k);
        end
        anno_uv_r = anno_uv_r(1:2, :);

        % write db
        write_binary_record(file, img_l, anno_xyz_l, anno_uv_l)
        write_binary_record(file, img_r, anno_xyz_r, anno_uv_r)
    end
end
fclose(file);