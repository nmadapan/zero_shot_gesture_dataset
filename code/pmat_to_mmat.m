%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% When you save a .mat file using scipy.io.savemat, the list of 
% strings are saved as character arrays and not cell arrays. 
% When such files are read in python using scipy.io.loadmat, 
% the character arrays are distorted and split character wise. 
% This file converts such character arrays back into cell arrays
% so that, when python reads it, it is read properly.
%
% Read all variables in INPUT_FNAME, transform all char arrays in cell
% arrays and right them back to OUTPUT_FNAME. The variables that are not
% char arrays are written to OUTPUT_FNAME as they are. 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

close all;
clear;
clc;

%% Initialization 
BASE_FOLDER = '..\data';
INPUT_FNAME = 'sd_data_mturk.mat';
OUTPUT_FNAME = 'sd_data_mturk2.mat';

%%
input_fpath = fullfile(BASE_FOLDER, INPUT_FNAME);
output_fpath = fullfile(BASE_FOLDER, OUTPUT_FNAME);

load(input_fpath)
var_names = fieldnames(load(input_fpath));

% Convert char arrays to cell arrays. 
for idx = 1 : numel(var_names)
   if(ischar(eval(var_names{idx})))
       temp = cellstr(eval(var_names{idx}));
       assignin('base', var_names{idx}, temp);
   end
end

% Removing the last two commands. Consider only first 26 commands. 
full_cmd_names = full_cmd_names(1:26);
full_bin_sd_mat = full_bin_sd_mat(1:26, :);
full_con_sd_mat = full_con_sd_mat(1:26, :);

%% Saving files
for idx = 1 : numel(var_names)
   if(idx == 1) 
      save(output_fpath, var_names{idx}) 
      continue
   end
   save(output_fpath, var_names{idx}, '-append') 
end