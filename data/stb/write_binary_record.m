function [ ] = write_binary_record( file, image, coord_xyz, coord_uv)
%Writes one entry to the binary db

    % write pose: v, uvz (float)
    for i=1:size(coord_xyz, 2)
        fwrite(file, coord_xyz(1, i), 'float'); %x
        fwrite(file, coord_xyz(2, i), 'float'); %y
        fwrite(file, coord_xyz(3, i), 'float'); %z
    end

    % write pose: v, uvz (float)
    for i=1:size(coord_uv, 2)
        fwrite(file, coord_uv(1, i), 'float'); %u
        fwrite(file, coord_uv(2, i), 'float'); %v
        fwrite(file, 1.0, 'float'); %valid
    end

    % write image
    image = permute(image, [3 2 1]);
    fwrite(file, image(:), 'uint8');
end

