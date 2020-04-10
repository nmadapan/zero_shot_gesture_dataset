close all;
clear;
clc;

load('..\sd_data_mturk.mat')

%% Command names
full_cmd_names = cellstr(full_cmd_names);
full_cmd_names = full_cmd_names(1:end-2);
fid = fopen('.\class_labels.txt','w');
for idx = 1 : numel(full_cmd_names)
    fprintf(fid, '%s\n', full_cmd_names{idx});
end
fclose(fid);

%% Full list of semantic descriptors
% Semantic descriptor names
full_sd_names = cellstr(full_sd_names);
fid = fopen('.\full_descriptor_names.txt','w');
for idx = 1 : numel(full_sd_names)
    fprintf(fid, '%s\n', full_sd_names{idx});
end
fclose(fid);

% Binary SD matrix
full_bin_sd_mat = full_bin_sd_mat(1:end-2, :);
dlmwrite('.\full_binary_description_matrix.csv', full_bin_sd_mat)

% Continuous SD matrix
full_con_sd_mat = full_con_sd_mat(1:end-2, :);
dlmwrite('.\full_continuous_description_matrix.csv', full_con_sd_mat)

%% Reduced list of semantic descriptors and classes
% % Class names
% reduced_cmd_names = cellstr(full_cmd_names);
% reduced_cmd_names = reduced_cmd_names(1:26);
% fid = fopen('..\data\reduced_class_labels.txt','w');
% for idx = 1 : numel(reduced_cmd_names)
%     fprintf(fid, '%s\n', reduced_cmd_names{idx});
% end
% fclose(fid);

% Semantic descriptor names
sd_names = cellstr(sd_names);
fid = fopen('.\reduced_descriptor_names.txt','w');
for idx = 1 : numel(sd_names)
    fprintf(fid, '%s\n', sd_names{idx});
end
fclose(fid);

% Binary SD matrix
dlmwrite('.\reduced_binary_description_matrix.csv', bin_sd)

% Continuous SD matrix
dlmwrite('.\reduced_continuous_description_matrix.csv', con_sd)

%% Create masks
% unseen_class_ids = [8, 11, 12, 16, 18];
% seen_class_ids = setdiff(1:26, dstruct.unseen_class_ids);

% %% Saving files
% save('..\sd_data_mturk2.mat', 'bin_sd', 'con_sd', 'full_cmd_names', ...
%     'full_bin_sd_mat', 'full_con_sd_mat', 'full_sd_names', ...
%     'sd_names')