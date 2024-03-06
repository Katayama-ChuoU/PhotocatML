%% Data & parameter import
clear
% close all


% load('00_features.mat') 
% load('00_ML_para.mat')
% load('Ex_1_Hem_inputs.mat')
% load('Ex_2_BVO_inputs.mat')
load('Ex_3_Heterojunction_inputs.mat')



% load('00_note_info.mat')
% note_info.OriginalDataDir(1) = OriDataDir;
% note_info.OriginalDataDir(2) = OriDataDir;
% note_info_ori = note_info;

%% autoscaling of the table
% save("00_ML_para.mat","ML_para")
sample_name = ML_para.SampleInfo.SampleName;
inverse_id = ML_para.SampleInfo.Inverse;

myTbl = features_tbl;
myTbl_matrix = myTbl.Variables;
myTbl_size = size(myTbl_matrix);
myTbl_name = myTbl.Properties.VariableNames;

for i = 1:myTbl_size(2)
    scl_myTbl_matrix(:,i) = (myTbl_matrix(:,i) - mean(myTbl_matrix(:,i)))/std(myTbl_matrix(:,i));
end

scl_myTbl = table;
scl_myTbl.Variables = scl_myTbl_matrix;
scl_myTbl.Properties.VariableNames = myTbl_name;


if strcmp(inverse_id ,"true") == 1
    data_type = append(sample_name," ",num2str(size(scl_myTbl_matrix,1))," samples, ",num2str(size(scl_myTbl_matrix,2)-1)," features with inverse");
else
    data_type = append(sample_name," ",num2str(size(scl_myTbl_matrix,1))," samples, ",num2str(size(scl_myTbl_matrix,2)-1)," features");
end

data_type_ori = data_type;
scl_myTbl_ori = scl_myTbl;
scl_myTbl_matrix_ori = scl_myTbl_matrix;


%% Filter feature selection --remove the zero/low relative standard deviation--
% save("00_note_info.mat","note_info")
% savevar_name.RSTD_filter{1} = '01_inputs.mat';save(savevar_name.RSTD_filter{1})
% auto_input_organizer;

rstd_thrs = ML_para.Filter.RSTD_filter;
scl_myTbl = scl_myTbl_ori;
scl_myTbl_matrix = scl_myTbl_matrix_ori;

rsd = std(features_tbl{:,:})./mean(features_tbl{:,:});
constant_dcp_idx = abs(rsd) < abs(rstd_thrs);
lowsdfeatures = scl_myTbl(:,constant_dcp_idx);
low_rsd = string(round(rsd(constant_dcp_idx),3));
lowsdfeatures_name = (lowsdfeatures.Properties.VariableNames);
lowsd_tbl = table( lowsdfeatures_name',low_rsd');
lowsd_tbl.Properties.VariableNames = {'Feature_Names','rsd'};
scl_myTbl(:,constant_dcp_idx) = [];
scl_myTbl_matrix(:,constant_dcp_idx) = [];
myTbl_matrix(:,constant_dcp_idx) = [];


num_filtered = length(low_rsd);
cos_sim_tbl_low_sd = cos_s_map(lowsdfeatures,1);
cos_sim_tbl_all = cos_s_map(scl_myTbl,2);

scl_myTbl_ori = scl_myTbl;
scl_myTbl_matrix_ori = scl_myTbl_matrix;
scl_myTbl_RSTD = scl_myTbl_ori;
scl_myTbl_matrix_RSTD = scl_myTbl_matrix_ori;

% record
% fig_idx.RSTD_filter{1} = 1;
% fig_idx.RSTD_filter{2} = 2;
% savefig_name.RSTD_filter{1} = strcat('01_lowrstd_cosmap.fig');saveas(figure(fig_idx.RSTD_filter{1}),savefig_name.RSTD_filter{1})
% savefig_name.RSTD_filter{2} = strcat('01_lowrstd_filtered_cosmap.fig');saveas(figure(fig_idx.RSTD_filter{2}),savefig_name.RSTD_filter{2})
% savevar_name.RSTD_filter{2} = strcat('01_rstd_filt_',num2str(rstd_thrs),'.mat');save(savevar_name.RSTD_filter{2})
% 
% [calc_summary_info,calc_folder_last] = matlabnote_save("RSTD_filter",note_info,savefig_name,fig_idx,savevar_name);
% matlabnote_ppt("RSTD_filter",note_info,savefig_name,fig_idx,savevar_name,calc_folder_last);

% %% Filter feature selection -- backward sequential (correlation)
% % % Notice 
% % Descriptor order dependecy exist
% % Descriptor Var1 and Var2 have high correlation, Var1 is removed and Var2 remains. 
% % then, descirptor table is sorted before backward sequential
% % auto_input_organizer;
% % savevar_name.Corr_filter{1} = '02_inputs.mat';save(savevar_name.Corr_filter{1})
% cos_sim_tbl_PEC_abs = rows2vars(cell2table(cos_sim_tbl_all.Properties.VariableNames));
% cos_sim_tbl_PEC_abs.Properties.VariableNames{end} = 'DescriptorName';
% cos_sim_tbl_PEC_abs(:,1) =[];
% cos_sim_tbl_PEC_abs = [cos_sim_tbl_PEC_abs,array2table(abs(cos_sim_tbl_all.PEC))];
% cos_sim_tbl_PEC_abs.Properties.VariableNames{end} = 'PEC_abs';
% [~,sort_idx] = sortrows(cos_sim_tbl_PEC_abs,2);
% 
% feat_tbl = scl_myTbl_ori(:,sort_idx);
% 
% feat = table2array(feat_tbl(:,1:end-1));
% 
% corr(feat,"Rows","pairwise");
% 
% opts = statset("Display","iter");
% [~,history] = sequentialfs(@mycorr,feat, ...
%     "Direction","backward","NullModel",true, ...
%     "CV","none","Options",opts);
% myTbl_name = feat_tbl.Properties.VariableNames;
% fig_posi = size(feat,2);
% idx = NaN(1,fig_posi);
% for i = 1 : fig_posi
%     idx(i) = find(history.In(i,:)~=history.In(i+1,:));
% end
% 
% cutoff_para = ML_para.Filter.corr_cutoff;
% iter_last_exclude = find(history.Crit(2:end)<cutoff_para,1);
% idx_filtered = idx(iter_last_exclude+1:end);
% idx_removed = setdiff(idx, idx_filtered)
% featRmv = string(myTbl_name(idx_removed));
% message1 = append(featRmv, " removed.")
% message2 = append("Features: ", num2str(length(myTbl_name)-1), " to ", num2str(length(idx_filtered)))
% 
% if strcmp(inverse_id,"true") == 1
%     data_type = append(sample_name," ",num2str(size(scl_myTbl_matrix,1))," samples, ",num2str(size(scl_myTbl_matrix,2)-1)," features with inverse, corr_cutoff > ", num2str(cutoff_para));
% else
%     data_type = append(sample_name," ",num2str(size(scl_myTbl_matrix,1))," samples, ",num2str(size(scl_myTbl_matrix,2)-1)," features, corr_cutoff > ", num2str(cutoff_para));
% end
% 
% % Pearson correlation coefficient mapping (after filter)
% 
% % feat_tbl(:,idx_removed) = [];
% % scl_myTbl = feat_tbl;
% % scl_myTbl_matrix = table2array(scl_myTbl);
% % scl_myTbl_matrix = feat_tbl(:,sort([idx_filtered, length(myTbl_name)]));
% scl_myTbl(:,featRmv) = [];
% scl_myTbl_matrix = table2array(scl_myTbl);
% cc = corrcoef(scl_myTbl_matrix);
% heatmap_ax = scl_myTbl.Properties.VariableNames;
% % heatmap_ax = myTbl_name(sort([idx_filtered, length(myTbl_name)]));
% 
% for i = 1:length(heatmap_ax)
%     heatmap_ax{i} = strrep(heatmap_ax{i},'_','-');
% end
% 
% figure(3)
% set(figure(3),'Color', [1 1 1])
% hm = heatmap(heatmap_ax,heatmap_ax,cc);
% hm.Colormap = jet;
% hm.FontSize  = 10;
% 
% scl_myTbl_ori = scl_myTbl;
% scl_myTbl_matrix_ori = scl_myTbl_matrix;
% 
% 
% % % record 
% featRmv_tbl = table(featRmv');
% featRmv_tbl.Properties.VariableNames{1} = 'Descriptor Name';
% % fig_idx.Corr_filter{1} = 3;
% % fig_idx.Corr_filter{2} = 5;
% savefig_name.Corr_filter{1} = strcat('02_cos_s_map_filter.fig');saveas(figure(fig_idx.Corr_filter{1}),savefig_name.Corr_filter{1})
% 
% savevar_name.Corr_filter{2} = strcat('02_corr_filt_',num2str(cutoff_para),'.mat');save(savevar_name.Corr_filter{2})
% 
% [calc_summary_info,calc_folder_last] = matlabnote_save("Corr_filter",note_info,savefig_name,fig_idx,savevar_name);
% matlabnote_ppt("Corr_filter",note_info,savefig_name,fig_idx,savevar_name,calc_folder_last);


% % %%
% % Find the unique values in cluster_idx
% uniqueValues = unique(cluster_idx,'stable');
% 
% % Initialize a cell array to store the indices for each unique value
% indicesCell = cell(size(uniqueValues,1),2);
% Ind_T_idx = zeros(size(uniqueValues,1),1);
% % Iterate over each unique value and find its indices
% for i = 1:numel(uniqueValues)
%     indicesCell{i,1} = find(cluster_idx == uniqueValues(i));
%     indicesCell{i,2} = uniqueValues(i);
%     Ind_T_idx(i,1) = length(indicesCell{i,1}) == 1;
% end
% Ind_T_idx = logical(Ind_T_idx);
% % indicesCell2 = indicesCell;
% % indicesCell2{T_idx,1:2} =[];
% 
% Ind_T = [cell2table(indicesCell),table(Ind_T_idx)];
% Ind_T2 = sortrows(Ind_T,2);
% Ind_T3 = Ind_T2;
% Ind_T3(Ind_T3.T_idx,:) = [];
% overlapping_number_indices2 = table2cell(Ind_T3(:,1));
% % 
% % for i = 1:size(T2,1)
% %     if length(T2(i,1)) == 1
% %         T_idx = true;
% %     else
% %         T_idx = false;
% %     end
% % end
% 
% 
% overlapping_indices2 = [];
% for i = 1:length(uniqueValues)
%     if length(indicesCell{i}) > 1
%         overlapping_indices2 = [overlapping_indices2; indicesCell{i}];
%     end
% end
% 
% for i = 1:numel(overlapping_indices2)
%     overlapping_number2{i} = cluster_idx(overlapping_indices2(i));
%     overlapping_number_indices2{i} = find(cluster_idx == overlapping_number2{i});
%     % disp(['Number ', num2str(overlapping_number), ' - Indices: ', num2str(overlapping_number_indices')]);
% end
%% Fileter feature extraction by concantation of features by clustering
% draw dendrogram and decide threshold by cutoff para
% save("00_note_info.mat","note_info")
% auto_input_organizer;
% savevar_name.Cluster_filter{1} = '02_inputs.mat';save(savevar_name.Cluster_filter{1})

distfunc = ML_para.Filter.dist_func; % distfunc = "cosine_abs" | "cosine" 
cluster_method = ML_para.Filter.cluster_method; % "average" | "complete" | "single"

if strcmp(distfunc,"cosine_abs") == 1
    distfunc = @dist_cos_abs;
end

scl_myTbl = scl_myTbl_ori;
scl_myTbl_matrix = scl_myTbl_matrix_ori;


cutoff_para = ML_para.Filter.cluster_cutoff;


if strcmp(inverse_id,"true") == 1
    data_type = append(sample_name," ",num2str(size(scl_myTbl_matrix,1))," samples, ",num2str(size(scl_myTbl_matrix,2)-1)," features with inverse, cluster_cutoff < ", num2str(cutoff_para));
else
    data_type = append(sample_name," ",num2str(size(scl_myTbl_matrix,1))," samples, ",num2str(size(scl_myTbl_matrix,2)-1)," features, cluster_cutoff < ", num2str(cutoff_para));
end

all_featuresName_tbl = table(scl_myTbl.Properties.VariableNames');
all_featuresName_tbl(end,:) = [];
all_featuresName_str = table2cell(all_featuresName_tbl);

for i = 1:length(all_featuresName_str)
    all_featuresName_str_ax{i} = strrep(all_featuresName_str{i},'_','\_');
end

rng(3)

tree = linkage((scl_myTbl_matrix(:,1:end-1))',cluster_method,distfunc);
clear cluster_idx
cluster_idx = cluster(tree,'cutoff',cutoff_para,'Criterion','distance');
all_featuresNames_labeled = [all_featuresName_tbl,array2table(cluster_idx)];
all_featuresNames_labeled.Properties.VariableNames{1} = 'Features name';
all_featuresNames_lb_sort = sortrows(all_featuresNames_labeled,2);

% Find the unique values in cluster_idx
unique_values = unique(cluster_idx,'stable');

% Initialize a cell array to store the indices for each unique value
indicesCell = cell(size(unique_values,1),2);
Ind_T_idx = zeros(size(unique_values,1),1);
% Iterate over each unique value and find its indices
for i = 1:numel(unique_values)
    indicesCell{i,1} = find(cluster_idx == unique_values(i));
    indicesCell{i,2} = unique_values(i);
    Ind_T_idx(i,1) = length(indicesCell{i,1}) == 1;
end
Ind_T_idx = logical(Ind_T_idx);
% indicesCell2 = indicesCell;
% indicesCell2{T_idx,1:2} =[];

Ind_T = [cell2table(indicesCell),table(Ind_T_idx)];
Ind_T2 = sortrows(Ind_T,2);
Ind_T3 = Ind_T2;
Ind_T3(Ind_T3.Ind_T_idx,:) = [];
overlapping_number_indices = table2cell(Ind_T3(:,1));
overlapping_number = table2cell(Ind_T3(:,2));
% unique_values = unique(cluster_idx);
% overlapping_indices = find(histcounts(cluster_idx) > 1); % bag 
% overlapping_number = cell(numel(overlapping_indices),1);
% overlapping_number_indices =  cell(numel(overlapping_indices),1);
% 
% for i = 1:numel(overlapping_indices)
%     overlapping_number{i} = unique_values(overlapping_indices(i));
%     overlapping_number_indices{i} = find(cluster_idx == overlapping_number{i});
%     % disp(['Number ', num2str(overlapping_number), ' - Indices: ', num2str(overlapping_number_indices')]);
% end

% filp the sign of one of the negatively correlated descriptors before
% taking average descriptors 
scl_myTbl_clust_c = cell(1,length(overlapping_number_indices));
for k = 1:length(overlapping_number_indices)
    scl_myTbl_clust = scl_myTbl(:,(overlapping_number_indices{k}));
    scl_myTbl_clust_ori = scl_myTbl_clust; 
    scl_myTbl_clust_c{k} = scl_myTbl_clust_ori;
    scl_myTbl_clust_varname = scl_myTbl_clust.Properties.VariableNames;
    scl_myTbl_clust_mat = table2array(scl_myTbl_clust);
    corr_map =  corrcoef(scl_myTbl_clust_mat);

    lower_triangular = tril(corr_map, -1);
    negative_trend_descriptors = find(lower_triangular < 0);
    negative_trend_descriptors(negative_trend_descriptors > size(lower_triangular,1)) = [];
    if isempty(negative_trend_descriptors) == 0
    scl_myTbl_clust_mat(:, negative_trend_descriptors) = -1 * scl_myTbl_clust_mat(:, negative_trend_descriptors);
    end
    scl_myTbl_clust = array2table(scl_myTbl_clust_mat,"VariableNames",scl_myTbl_clust_varname);
    scl_myTbl(:,(overlapping_number_indices{k})) = scl_myTbl_clust;
   
end

overlapping_clusters = cell2mat(overlapping_number);
unique_clusters = setdiff(cluster_idx,overlapping_clusters);

scl_myTbl_re = table;
ovrp_count  = 0;

% commonPattern = ["XRD_","Raman_","UV_Vis_","DRS_"];
commonPattern = ML_para.Filter.commonPattern;

for i = 1:length(unique_values)
    if ismember(i,unique_clusters)
        idx_unique = find(cluster_idx == i);

        scl_myTbl_re = [scl_myTbl_re,scl_myTbl(:,idx_unique)];

    else
        clear features4cluster cluster_features
        ovrp_count = ovrp_count + 1;
        features4cluster = scl_myTbl(:,overlapping_number_indices{ovrp_count});
        featuresnames4cluster = cellstr(join(string(features4cluster.Properties.VariableNames),', '));
        featureNames4cluster_short = features4cluster.Properties.VariableNames;
        for m = 1:length(commonPattern)
            TF = contains(featureNames4cluster_short,commonPattern(m));

            for n = 2:length(overlapping_number_indices{ovrp_count})
                if sum(TF) > 1 && sum(TF) == length(featureNames4cluster_short)
                    featureNames4cluster_short{n} = replace(featureNames4cluster_short{n}, commonPattern(m), '');
                end
            end

        end
        featureNames4cluster_short = cellstr(join(string(featureNames4cluster_short),', '));
        cluster_features = array2table(mean(table2array(features4cluster),2));
        tmp_varnames =  char(strcat('Cluster',num2str(ovrp_count),'-(',featureNames4cluster_short,')'));
        if strlength(tmp_varnames) <= 63
            cluster_features.Properties.VariableNames{1} = char(strcat('Cluster',num2str(ovrp_count),'-(',featureNames4cluster_short,')'));
        else
            cluster_features.Properties.VariableNames{1} = char(strcat('Cluster',num2str(ovrp_count)));
            
        end
        % cluster_features.Properties.VariableNames{1} = char(strcat("Cluster",string(num2str(ovrp_count)),"-"+newline+featureNames4cluster_short));
        scl_myTbl_re = [scl_myTbl_re,cluster_features];
        
    end
end

cluster_tbl = table;
tmp_tbl = table;

for i = 1:length(overlapping_number_indices)
    tmp_tbl = all_featuresNames_labeled(overlapping_number_indices{i, 1},1);
    tmp_tbl = [tmp_tbl,array2table(repmat(i,[size(tmp_tbl,1),1]))];
    cluster_tbl = [cluster_tbl;tmp_tbl];
end

cluster_tbl.Properties.VariableNames(2) = "Cluster idx";
num_filtered = size(cluster_tbl,1) - max(cluster_tbl.("Cluster idx"));
cluster_tbl

scl_myTbl_re = [scl_myTbl_re,scl_myTbl(:,end)];

scl_myTbl = normalize(scl_myTbl_re);
scl_myTbl_matrix = table2array(scl_myTbl);
cos_sim_tbl_fil = cos_s_map(scl_myTbl,5);


for i = 1:length(overlapping_number_indices)
    tmp_ax_str = all_featuresName_str_ax(overlapping_number_indices{i});
    for j = 1:length(tmp_ax_str)
        tmp_ax_str{j} = strcat('Cluster',num2str(i),' (',tmp_ax_str{j},')');
    end
    all_featuresName_str_ax(overlapping_number_indices{i}) = tmp_ax_str;
end

figure(4);
fig_posi = get(gcf,'Position');close(4);figure(4);set(gcf,'Position',fig_posi)
ax = gca(4);
% set(groot,"CurrentFigure",4);
[H,Ind_T, outperm] = dendrogram(ax,tree,0,'Labels',all_featuresName_str_ax);

hold on 
cut_plot = cutoff_para*ones(200,1);
adjfig
ax.FontSize = 10;
plot(cut_plot,'LineWidth',2,'Color',[1 0 0])

scl_myTbl_clust_concate = table;
for m = 1:size(scl_myTbl_clust_c,2)
    scl_myTbl_clust_c{m}.Properties.VariableNames = strcat('Cluster',num2str(m),'-',scl_myTbl_clust_c{m}.Properties.VariableNames);
    scl_myTbl_clust_concate = [scl_myTbl_clust_concate,scl_myTbl_clust_c{m}];
end

cos_sim_tbl_clusters = cos_s_map(scl_myTbl_clust_concate,6);

% fig_idx.Cluster_filter{1} = 4;
% fig_idx.Cluster_filter{2} = 5;
% fig_idx.Cluster_filter{3} = 6;
% 
% savefig_name.Cluster_filter{1} = strcat('02_dendrogram.fig');saveas(figure(fig_idx.Cluster_filter{1}),savefig_name.Cluster_filter{1})
% savefig_name.Cluster_filter{2} = strcat('02_cos_s_map_withcluster.fig');saveas(figure(fig_idx.Cluster_filter{2}),savefig_name.Cluster_filter{2})
% savefig_name.Cluster_filter{3} = strcat('02_cos_s_map_incluster.fig');saveas(figure(fig_idx.Cluster_filter{3}),savefig_name.Cluster_filter{3})
% 
% savevar_name.Cluster_filter{2} = strcat('02_cluster_filt_',num2str(cutoff_para),'.mat');save(savevar_name.Cluster_filter{2})
% % 
% [calc_summary_info,calc_folder_last] = matlabnote_save("Cluster_filter",note_info,savefig_name,fig_idx,savevar_name);
% matlabnote_ppt("Cluster_filter",note_info,savefig_name,fig_idx,savevar_name,calc_folder_last);

% %%
% 
% %% flip the sign of negative correlation descriptor
% % using sequentialfs 
% 
% for k = 1:length(overlapping_number_indices)
% feat = table2array(scl_myTbl_ori(:,(overlapping_number_indices{k})));
% opts = statset("Display","iter");
% [~,history_neg_corr] = sequentialfs(@mycorr2,feat, ...
%     "Direction","backward","NullModel",true, ...
%     "CV","none","Options",opts);
% myTbl_name = scl_myTbl_ori(:,(overlapping_number_indices{k})).Properties.VariableNames;
% % p = size(feat,2)-2;
% % idx = NaN(1,p);
% % for i = 1 : p
% %     idx(i) = find(history_neg_corr.In(i,:)~=history_neg_corr.In(i+1,:));
% % end
% 
% threshold = 0.85;
% idx_fliped = find(history_neg_corr.Crit>threshold,1);
% 
% % idx_fliped = idx(iter_last_exclude+1:end);
% % idx_removed = setdiff(idx, idx_fliped);
% featFlip = string(myTbl_name(idx_fliped));
% featFlip_c{k} = featFlip;
% end
% %% 
% commonPattern =['XRD_','DRS_','UV_Vis_','Raman_'];
% for n = 2:2
%     features4cluster.Properties.VariableNames{n} = strrep(features4cluster.Properties.VariableNames{n}, commonPattern, '');
% end
% disp(features4cluster.Properties.VariableNames)
%% Cross validation with descriptor selection by sequential add/remove descriptors (wrapper)
% save("00_note_info.mat","note_info")
% savevar_name.CV{1} = '03_inputs.mat';save(savevar_name.CV{1})
% auto_input_organizer;
                                  
ML_method = ML_para.ML_method;
options = ML_para;

% % reference parameter
% options.SVR.kernel = "linear"; % "linear" (default), "gaussian"
% options.SVR.epsilon = 1e-1; % iqr(Y)/13.49 (default)
% 
% options.GPR.Basis = "linear";
% options.GPR.Kernel ="squaredexponential"; 
% % exponential, squaredexponential, matern32, matern52, rationalquadratic
% % ardsquaredexponential, ardmatern32, ardmatern52, ardrationalquadratic
% options.GPR.FitMethod ="exact";
% options.GPR.PredictMethod = "exact";
% options.GPR.Sigma = 0.1;
% 
% 
% % Tree
% % options.Tree.SplitMethod = 'allsplits'; %"allsplits" (default) | "curvature" | "interaction-curvature
% options.Tree.MaxNumSplits = 30; % size(X,1) -1 (default)
% options.Tree.MinLeafSize = 2;   % 1 (default)
% options.Tree.MinParentSize = 1; % 1 (default) | MinParentSize = max(MinParentSize,2*MinLeafSize)
% 
% % ensemble 
% options.Ensemble.Method = "LSBoost"; % "LSBoost" (default) | "Bag" (random forest ?)
% % options.Ensemble.BagMethod = "TemplateTree"; % TemplateTree | TreeBagger 
% % options.Ensemble.NumTrees = 200; % For TreeBagger 
% 
% options.Ensemble.MaxNumSplits = 30; % size(X,1) -1 (default)
% options.Ensemble.MinLeafSize = 2;   % 1 (default)
% options.Ensemble.MinParentSize = 1; % 1 (default) | MinParentSize = max(MinParentSize,2*MinLeafSize)
% options.Ensemble.Tolerance = 1e-6; % 1e-6 (default)
% options.Ensemble.Regularization ="off"; % "on" (LASSO to shirink tree) | "off" (general setting) 
% options.Ensemble.NumCyle = 200; % 100 (defalut) 
% options.Ensemble.LearnRate = 0.1; % 1 (default), only if Method is "LSBoost" 
% 
% % simple_stepwise linear
% % cannot use AIC or BIC. cannot use upper, lower, PEnter, and PRemove
% options.simple_stepwise.modelspec = "linear"; %'constant' | 'linear' | 'interactions' | 'purequadratic' | 'quadratic'

% 
% if strcmp(inverse_id,"true") == 1
%     data_type = append(sample_name," ",num2str(size(scl_myTbl_matrix,1))," samples, ",num2str(size(scl_myTbl_matrix,2)-1)," features with inverse, cluster_cutoff < ", num2str(cutoff_para));
% else
%     data_type = append(sample_name," ",num2str(size(scl_myTbl_matrix,1))," samples, ",num2str(size(scl_myTbl_matrix,2)-1)," features, cluster_cutoff < ", num2str(cutoff_para));
% end

all_featuresName_tbl = table(scl_myTbl.Properties.VariableNames');
all_featuresName_tbl(end,:) = [];
descriptor_name_all = scl_myTbl.Properties.VariableNames;
descriptor_name_all(end) = [];

x = scl_myTbl_matrix(:,1:end-1);
y = myTbl_matrix(:,end);

x_ave = mean(x,1);x_std = std(x,1);
y_ave = mean(y,1);y_std = std(y,1);

x_scl = (x - x_ave) ./ x_std;
y_scl = (y - y_ave) ./ y_std;


kfold = ML_para.k_fold;
rng(1)
cv = cvpartition(size(x,1),'KFold',kfold);

drct = "forward"; % sequential add --> "forward" (default), sequential remove --> "backward"

if strcmp(ML_method,"SVR") == 1
    fun = @(xtrain, ytrain, xtest, ytest) svm_error(xtrain, ytrain, xtest, ytest, options,y_ave,y_std);
elseif strcmp(ML_method,"GPR") == 1
    fun = @(xtrain, ytrain, xtest, ytest) gpr_error(xtrain, ytrain, xtest, ytest, options,y_ave,y_std);
elseif strcmp(ML_method,"Tree") == 1
    fun = @(xtrain, ytrain, xtest, ytest) tree_error(xtrain, ytrain, xtest, ytest, options,y_ave,y_std);
elseif strcmp(ML_method, "Ensemble") == 1
    fun = @(xtrain, ytrain, xtest, ytest) ensemble_error(xtrain, ytrain, xtest, ytest, options,y_ave,y_std);
elseif strcmp(ML_method, "simple_stepwise") == 1
    fun = @(xtrain, ytrain, xtest, ytest) simple_stepwise_error(xtrain, ytrain, xtest, ytest, options,y_ave,y_std);
end
% [tokeep,history] = sequentialfs(fun, x_scl, y_scl, "cv", cv,'nfeatures',size(x_scl,2));
[tokeep,history] = sequentialfs(fun, x_scl, y_scl, "cv", cv,"direction",drct);

% revise the criteria values for R2
history.Crit_re = history.Crit*sum(cv.TestSize)/kfold;
history.Crit_r_sq = 1 - history.Crit_re;

figure(7);fig_posi = get(gcf,'Position');close(7);figure(7);set(gcf,'Position',fig_posi) 
plot(history.Crit_r_sq,'-o')
adjfig
xlabel("Number of descriptors [-]")
ylabel("Mean of R^2 test sets [-]")

x_slct = x_scl(:,tokeep);

figure(8);fig_posi = get(gcf,'Position');close(8);figure(8);set(gcf,'Position',fig_posi);hold on;
xlabel("True current [mA/cm^2]");ylabel("Predicted current [mA/cm^2]")
figure(9);fig_posi = get(gcf,'Position');close(9);figure(9);set(gcf,'Position',fig_posi);hold on
xlabel("True current [mA/cm^2]");ylabel("Predicted current [mA/cm^2]")
clear ml_mdl
for k = 1:kfold

    train_idx{k} = training(cv,k);
    x_train_scl{k} = x_slct(train_idx{k},:);
    y_train_scl{k} = y_scl(train_idx{k},:);

    if strcmp(ML_method,"SVR") == 1
        ml_mdl{k} = fitrsvm(x_train_scl{k}, y_train_scl{k}, "KernelFunction", options.SVR.kernel, "Epsilon", options.SVR.epsilon);

    elseif strcmp(ML_method,"GPR") == 1
        ml_mdl{k} = fitrgp(x_train_scl{k}, y_train_scl{k},  ...
            'BasisFunction', options.GPR.Basis, ...
            'KernelFunction', options.GPR.Kernel, ...
            'FitMethod', options.GPR.FitMethod, ...
            'PredictMethod', options.GPR.PredictMethod, ...
            'Sigma', options.GPR.Sigma);

    elseif strcmp(ML_method,"Tree") == 1
        rng(1)
        ml_mdl{k} = fitrtree(x_train_scl{k}, y_train_scl{k}, ...
            'MaxNumSplits' ,options.Tree.MaxNumSplits,...
            'MinLeafSize' ,options.Tree.MinLeafSize,...
            'MinParentSize' ,options.Tree.MinParentSize,...
            'Reproducible',true);

    elseif strcmp(ML_method,"Ensemble") == 1
        rng(2)
        t = templateTree('Reproducible',true, ...
            'MaxNumSplits' ,options.Ensemble.MaxNumSplits,...
            'MinLeafSize' ,options.Ensemble.MinLeafSize,...
            'MinParentSize' ,options.Ensemble.MinParentSize,...
            'QuadraticErrorTolerance',options.Ensemble.Tolerance, ...
            'NumVariablesToSample','all');
        rng(2);
        if strcmp(options.Ensemble.Method, "LSBoost") == 1

            ml_mdl{k} = fitrensemble(x_train_scl{k}, y_train_scl{k}, ...
                'Method',options.Ensemble.Method, ...
                'NumLearningCycles',options.Ensemble.NumCyle, ...
                'Learners',t,...
                'LearnRate',options.Ensemble.LearnRate);
        elseif strcmp(options.Ensemble.Method, "Bag") == 1
            s = RandStream('mlfg6331_64');
            ml_mdl{k} = fitrensemble(x_train_scl{k}, y_train_scl{k}, ...
                'Method',options.Ensemble.Method, ...
                'NumLearningCycles',options.Ensemble.NumCyle, ...
                'Learners',t,...
                'Options',statset('UseParallel',true,'UseSubstreams',true,'Streams',s));
        end

    elseif strcmp(ML_method, "simple_stepwise") == 1
        ml_mdl{k} = fitlm(x_train_scl{k}, y_train_scl{k},options.simple_stepwise.modelspec);

    end

    if strcmp(ML_method, "simple_stepwise") == 1
        YpredTrain_scl{k} = ml_mdl{k}.Fitted;
    else
        YpredTrain_scl{k} = resubPredict(ml_mdl{k});
    end

    YpredTrain{k} = YpredTrain_scl{k}*y_std + y_ave;
    y_train{k} = y_train_scl{k}*y_std + y_ave;
    tmp_ymax_train(k) = max([YpredTrain{k};y_train{k}]);
    tmp_ymin_train(k) = min([YpredTrain{k};y_train{k}]);

    error_trn(k) = error_all(y_train{k}, YpredTrain{k});
    r_sq_train(k) = error_trn(k).R_sq;

    figure(8)
    scatter(y_train{k},YpredTrain{k},100, 'filled')


    test_idx{k} = test(cv,k);
    x_test_scl{k} = x_slct(test_idx{k},:);
    y_test_scl{k} = y_scl(test_idx{k});

    YpredTest_scl{k} = predict(ml_mdl{k},x_test_scl{k});

    YpredTest{k} = YpredTest_scl{k}*y_std + y_ave;
    y_test{k} = y_test_scl{k} * y_std + y_ave;
    tmp_ymax_test(k) = max([YpredTest{k};y_test{k}]);
    tmp_ymin_test(k) = min([YpredTest{k};y_test{k}]);

    error_tst(k) = error_all(y_test{k}, YpredTest{k});
    r_sq_tst(k) = error_tst(k).R_sq;

    figure(9)
    scatter(y_test{k},YpredTest{k},100, 'filled')

end

figure(8);hold off
tmp_max = max(tmp_ymax_train(k)) * 1.2;
tmp_min = min(tmp_ymin_train(k)) / 1.2;

axis equal;adjfig; h = refline(1,0);h.Color ='black';h.LineWidth = 1.5;
r_sq_train_ave = mean(r_sq_train);
r_sq_train_str = strcat("R^2 = ",num2str(round(r_sq_train_ave,3))); 
annot = annotation('textbox',[0.55 0.15 0.2 0.15],'FitBoxToText','on','String',r_sq_train_str,'EdgeColor','none');
set(annot,'FontSize',16,'FontName','Arial')
legend({'1st' '2nd' '3rd' '4th' '5th'}, 'location', 'NW');
legend('boxoff')
xlim([tmp_min, tmp_max]); ylim([tmp_min, tmp_max]);
figure(9);hold off
tmp_max = max(tmp_ymax_test(k)) * 1.2;
tmp_min = min(tmp_ymin_test(k)) / 1.2;

axis equal;adjfig; h =refline(1,0);h.Color ='black';h.LineWidth = 1.5;
r_sq_tst_ave = mean(r_sq_tst);
r_sq_tst_str = strcat("R^2 = ",num2str(round(r_sq_tst_ave,3))); 
annot = annotation('textbox',[0.55 0.15 0.2 0.15],'FitBoxToText','on','String',r_sq_tst_str,'EdgeColor','none');
set(annot,'FontSize',16,'FontName','Arial')
legend({'1st' '2nd' '3rd' '4th' '5th'}, 'location', 'NW');
legend('boxoff')
xlim([tmp_min, tmp_max]); ylim([tmp_min, tmp_max]);

clear idx
for j = 1:size(history.In,1)
    if j == 1
        idx(j,:) = history.In(1,:);
    else
        idx(j,:) = history.In(j,:)-history.In(j-1,:);
    end
end

descriptors_c = [];
for j = 1:size(history.In,1)
    descriptors_c  = [descriptors_c;descriptor_name_all(idx(j,:))];
end
selected_descriptor_tbl = cell2table(descriptors_c);
selected_descriptor_tbl.Properties.VariableNames{1} = 'Descriptor name';

selected_descriptor_tbl
data_type
result_message = append(num2str(size(x_slct,2))," features were selected based on sequential ",ML_method, "(test ",r_sq_tst_str,")")
if strcmp(ML_method,"SVR") == 1
    para_tbl = (options.SVR);

elseif strcmp(ML_method,"GPR") == 1
    para_tbl = (options.GPR);

elseif strcmp(ML_method,"Tree") == 1
    para_tbl = (options.Tree);
elseif strcmp(ML_method, "Ensemble") == 1
    para_tbl = (options.Ensemble);

elseif strcmp(ML_method, "simple_stepwise") == 1
    para_tbl = (options.simple_stepwise);
end
para_array = table2array(para_tbl);
parameters_tbl = array2table(para_array', 'VariableNames', "Value", 'RowNames', para_tbl.Properties.VariableNames);
summary_tbl = table([string(round(r_sq_train_ave,3));string(round(r_sq_tst_ave,3))],'VariableNames', "Values", 'RowNames',["R_sq_train","R_sq_test"]);

% fig_idx.CV{1} = 8;fig_idx.CV{2} = 9;fig_idx.CV{3} = 7;
% savefig_name.CV{1} = strcat('03_',ML_method,'_train_CV.fig');saveas(figure(fig_idx.CV{1}),savefig_name.CV{1})
% savefig_name.CV{2} = strcat('03_',ML_method,'_test_CV.fig');saveas(figure(fig_idx.CV{2}),savefig_name.CV{2})
% savefig_name.CV{3} = strcat('03_',ML_method,'_history_r_sq.fig');saveas(figure(fig_idx.CV{3}),savefig_name.CV{3})
% savevar_name.CV{2} = strcat('03_',ML_method,'_CV.mat');save(savevar_name.CV{2})
% 
% [calc_summary_info,calc_folder_last] = matlabnote_save("CV",note_info,savefig_name,fig_idx,savevar_name);
% matlabnote_ppt("CV",note_info,savefig_name,fig_idx,savevar_name,calc_folder_last);



%% Shapley additive explanations
% 
% save("00_note_info.mat","note_info")
% savevar_name.shapley{1} = '04_inputs.mat';save(savevar_name.shapley{1})
desTbl = scl_myTbl(:,descriptors_c); % creat table of selected descriptors

if strcmp(ML_method,"SVR") == 1
    mdl = fitrsvm(desTbl, scl_myTbl(:,end), "KernelFunction", options.SVR.kernel, "Epsilon", options.SVR.epsilon);
elseif strcmp(ML_method,"GPR") == 1
    mdl = fitrgp(desTbl, scl_myTbl(:,end),'BasisFunction',options.GPR.Basis,'KernelFunction', options.GPR.Kernel, ...
        'FitMethod', options.GPR.FitMethod,'PredictMethod',options.GPR.PredictMethod,'Sigma', options.GPR.Sigma);
elseif strcmp(ML_method,"Tree") == 1
    rng(1)
    mdl = fitrtree(desTbl, scl_myTbl(:,end), ...
        'MaxNumSplits' ,options.Tree.MaxNumSplits,...
        'MinLeafSize' ,options.Tree.MinLeafSize,...
        'MinParentSize' ,options.Tree.MinParentSize,...
        'Reproducible',true);

elseif strcmp(ML_method,"Ensemble") == 1
    rng(2)
    t = templateTree('Reproducible',true, ...
        'MaxNumSplits' ,options.Ensemble.MaxNumSplits,...
        'MinLeafSize' ,options.Ensemble.MinLeafSize,...
        'MinParentSize' ,options.Ensemble.MinParentSize,...
        'QuadraticErrorTolerance',options.Ensemble.Tolerance, ...
        'NumVariablesToSample','all');
    rng(2);
    if strcmp(options.Ensemble.Method, "LSBoost") == 1

        mdl = fitrensemble(desTbl, scl_myTbl(:,end), ...
            'Method',options.Ensemble.Method, ...
            'NumLearningCycles',options.Ensemble.NumCyle, ...
            'Learners',t,...
            'LearnRate',options.Ensemble.LearnRate);
    elseif strcmp(options.Ensemble.Method, "Bag") == 1
        s = RandStream('mlfg6331_64');
        mdl = fitrensemble(desTbl, scl_myTbl(:,end), ...
            'Method',options.Ensemble.Method, ...
            'NumLearningCycles',options.Ensemble.NumCyle, ...
            'Learners',t,...
            'Options',statset('UseParallel',true,'UseSubstreams',true,'Streams',s));
    end

elseif strcmp(ML_method, "simple_stepwise") == 1
    mdl = fitlm(desTbl, scl_myTbl(:,end),options.simple_stepwise.modelspec);

end


valShap = [];
numS = length(desTbl.Variables);
numD = length(descriptors_c);

for i = 1 : numS
    queryPoint = desTbl(i,:);
    explainer = shapley(mdl,'QueryPoint',queryPoint);
    %plot(explainer)
    shapT = explainer.ShapleyValues{:,2};
    valShap = [valShap; shapT'];
end

meanAbShap_tmp = mean(abs(valShap));
[meanAbShap,idxShap] = sort(meanAbShap_tmp); % reorder from lowest to highest
des = strrep((descriptors_c(idxShap)),"_","\_");

figure(10);fig_posi = get(gcf,'Position');close(10);figure(10);set(gcf,'Position',fig_posi);adjfig
barh(meanAbShap)
yticklabels(des)
% title('Shapley summary plot (mean)')
xlabel("mean(|Shapley Value|)" + newline + "(average impact on model output)")
ax = gca;
ax.LineWidth = 1;
ax.FontSize = 10;

figure(11);fig_posi = get(gcf,'Position');close(11);figure(11);set(gcf,'Position',fig_posi);adjfig
hold on
for i = 1 : numD
   scatter(valShap(:,idxShap(i)), ... % x-value of each point is the shapley value
           i*ones(numS,1), ... % y-value of each point is an integer corresponding to a predictor (to be jittered below)
           [], ... % Marker size for each data point, taking the default here
           normalize(table2array(desTbl(:,idxShap(i))),'range',[1 256]), ... % Colors based on feature values
           'filled', ... % Fills the circles representing data points
           'YJitter','density', ... % YJitter according to the density of the points in this row
           'YJitterWidth',0.8)
   if (i==1) hold on; end
end

% title('Shapley summary plot')
xlabel('Shapley Value (impact on model output)')
yticks(1:numD)
yticklabels(des)

colormap (CoolBlueToWarmRedColormap) % set the color
% colormap(parula) % default colormap
cb= colorbar('Ticks', [1 256], 'TickLabels', {'Low', 'High'});
cb.Label.String = "Scaled Feature Value";
cb.Label.FontSize = 15;
cb.Label.Rotation = 270;
set(gca, 'YGrid', 'on')
xline(0, 'LineWidth', 1)
ax = gca;
ax.LineWidth = 1;
ax.FontSize = 10;
box on
hold off

% 
% fig_idx.shapley{1} = 10;
% fig_idx.shapley{2} = 11;
% 
% savefig_name.shapley{1} = strcat('04_',ML_method,'_waterfall_shapley.fig');saveas(figure(fig_idx.shapley{1}),savefig_name.shapley{1})
% savefig_name.shapley{2} = strcat('04_',ML_method,'_summary_shapley.fig');saveas(figure(fig_idx.shapley{2}),savefig_name.shapley{2})
% savevar_name.shapley{2} = strcat('04_',ML_method,'_shapley.mat');save(savevar_name.shapley{2})
% 
% [calc_summary_info,calc_folder_last] = matlabnote_save("shapley",note_info,savefig_name,fig_idx,savevar_name);
% matlabnote_ppt("shapley",note_info,savefig_name,fig_idx,savevar_name,calc_folder_last);


%% cos_s map
% save("00_note_info.mat","note_info")
% savevar_name.cos_map{1} = '05_inputs.mat';save(savevar_name.cos_map{1})
selected_descriptors = scl_myTbl(:,descriptors_c);
dat_tbl_slct = [selected_descriptors,array2table(scl_myTbl.PEC)];
dat_tbl_slct.Properties.VariableNames{end} = 'PEC';
cos_sim_tbl_slct = cos_s_map(dat_tbl_slct,12);
cos_sim_tbl =  cos_s_map(scl_myTbl,13);

% fig_idx.cos_map{1} = 12;
% fig_idx.cos_map{2} = 13;
% 
% savefig_name.cos_map{1} = strcat('05_slct_',ML_method,'_cos_map.fig');saveas(figure(fig_idx.cos_map{1}),savefig_name.cos_map{1})
% savefig_name.cos_map{2} = strcat('05_cos_map.fig');saveas(figure(fig_idx.cos_map{2}),savefig_name.cos_map{2})
% savevar_name.cos_map{2} = strcat('05_',ML_method,'_cos_map.mat');save(savevar_name.cos_map{2})
% cluster_tbl;
% 
% [calc_summary_info,calc_folder_last] = matlabnote_save("cos_map",note_info,savefig_name,fig_idx,savevar_name);
% matlabnote_ppt("cos_map",note_info,savefig_name,fig_idx,savevar_name,calc_folder_last);



% %% embedded method 
% 
% %% LASSO
% save("00_note_info.mat","note_info")
% savevar_name.Lasso{1} = '06_inputs.mat';save(savevar_name.Lasso{1})
% auto_input_organizer;
% 
% % k_fold = ipt_prm.k_fold;
% % lambda = ipt_prm.lasso_lambda;
% range = ML_para.Lasso.range; % example ML_para.Lasso.range = [-5 1 100],  divide data into 10^-5 ~ 10^1 in 100 points
% criteria = ML_para.Lasso.criteria; % "mse" or "1se"
% alpha = ML_para.Lasso.alpha; % 0~1, 1 --> LASSO, 0 --> Ridge 
% 
% % if strcmp(inverse_id,"true") == 1
% %     data_type = append(sample_name," ",num2str(size(scl_myTbl_matrix,1))," samples, ",num2str(size(scl_myTbl_matrix,2)-1)," features with inverse, cluster_cutoff < ", num2str(cutoff_para));
% % else
% %     data_type = append(sample_name," ",num2str(size(scl_myTbl_matrix,1))," samples, ",num2str(size(scl_myTbl_matrix,2)-1)," features, cluster_cutoff < ", num2str(cutoff_para));
% % end
% 
% all_featuresName_tbl = table(scl_myTbl.Properties.VariableNames');
% all_featuresName_tbl(end,:) = [];
% descriptor_name_all = scl_myTbl.Properties.VariableNames;
% descriptor_name_all(end) = [];
% 
% x = scl_myTbl_matrix(:,1:end-1);
% y = myTbl_matrix(:,end);
% 
% x_ave = mean(x,1);x_std = std(x,1);
% y_ave = mean(y,1);y_std = std(y,1);
% 
% x_scl = (x - x_ave) ./ x_std;
% y_scl = (y - y_ave) ./ y_std;
% 
% 
% kfold = ML_para.k_fold;
% rng(1)
% cv = cvpartition(size(x,1),'KFold',kfold);
% 
% [B_lasso, FitInfo_lasso] = lasso(x_scl, y_scl, 'CV', cv, 'Lambda', logspace(range(1),range(2),range(3)),'Alpha',alpha); % CV: cross validation
% figure(14);fig_posi = get(gcf,'Position');close(14);figure(14);set(gcf,'Position',fig_posi)
% lassoPlot(B_lasso, FitInfo_lasso,'PlotType','CV','Parent',gca(14));%6
% legend({'MSE with Error Bars','LambdaMinMSE','Lambda1SE'}) % Show legend
% title([])
% adjfig
% 
% % Fitinfo includes the index of MinMSE and Index1SE
% % MSE: mean square error
% if criteria == "mse"
%     idx = FitInfo_lasso.IndexMinMSE;
% elseif criteria == "1se"
%     idx = FitInfo_lasso.Index1SE;
% end
% 
% B_slct = B_lasso(:, idx);
% % true_data = myTbl_matrix(:, end); 
% % predict_data = X * B_slct * std(myTbl_matrix(:, end)) + mean(myTbl_matrix(:, end));
% 
% % statistical evaluation
% % TSS = sum((true_data - mean(true_data)).^2);
% % RSS_Lasso = sum((true_data - predict_data).^2);
% % rsquaredLasso = 1 - RSS_Lasso/TSS;
% % 
% % error_lasso = error_all(true_data, predict_data);
% 
% % figure %7
% % scatter(true_data, predict_data, 'filled')
% % tmp_max = max(max([true_data, predict_data])) * 1.2;
% % tmp_min = min(min([true_data, predict_data])) * 1.2;
% % axis equal; refline(1,0);
% % xlim([tmp_min, tmp_max]); ylim([tmp_min, tmp_max]);
% 
% large_B_idx = find(B_slct~=0);
% lst_large_B_name = (descriptor_name_all(large_B_idx))';
% lst_large_B_tbl = [array2table(lst_large_B_name), array2table(B_slct(large_B_idx))];
% lst_large_B_tbl.Properties.VariableNames(1) = "Descriptor name";
% lst_large_B_tbl.Properties.VariableNames(2) = "Coefficients";
% lst_large_B_tbl = [lst_large_B_tbl,array2table(abs(B_slct(large_B_idx)))];
% lst_large_B_tbl.Properties.VariableNames(3) = "Abs coefficients";
% 
% lst_large_B_sort = sortrows(lst_large_B_tbl,"Abs coefficients","descend");
% lst_large_B_sort(:,3) = [];
% lst_large_B_sort_ppt = [lst_large_B_sort(:,1),table(num2str(round(table2array(lst_large_B_sort(:,2)),3)))];
% lst_large_B_sort_ppt.Properties.VariableNames(2) = "Coefficients";
% descriptors_c = table2cell(lst_large_B_sort_ppt(:,1));
% 
% 
% figure(15);fig_posi = get(gcf,'Position');close(15);figure(15);set(gcf,'Position',fig_posi);hold on;
% xlabel("True current [mA/cm^2]");ylabel("Predicted current [mA/cm^2]")
% figure(16);fig_posi = get(gcf,'Position');close(16);figure(16);set(gcf,'Position',fig_posi);hold on
% xlabel("True current [mA/cm^2]");ylabel("Predicted current [mA/cm^2]")
% 
% 
% for k = 1:kfold
%     % training 
%     train_idx{k} = training(cv,k);
%     x_train_scl{k} = x_scl(train_idx{k},:);
%     y_train_scl{k} = y_scl(train_idx{k},:);
% 
%     YpredTrain_scl{k} = x_train_scl{k} * B_slct;
%     YpredTrain{k} = YpredTrain_scl{k}*y_std + y_ave;
%     y_train{k} = y_train_scl{k}*y_std + y_ave;
% 
%     tmp_ymax_train(k) = max([YpredTrain{k};y_train{k}]);
%     tmp_ymin_train(k) = min([YpredTrain{k};y_train{k}]);
% 
%     error_trn(k) = error_all(y_train{k}, YpredTrain{k});
%     r_sq_train(k) = error_trn(k).R_sq;
% 
%     figure(15)
%     scatter(y_train{k},YpredTrain{k},100, 'filled')
% 
%     % test 
%     test_idx{k} = test(cv,k);
%     x_test_scl{k} = x_scl(test_idx{k},:);
%     y_test_scl{k} = y_scl(test_idx{k});
% 
%     YpredTest_scl{k} = x_test_scl{k} * B_slct;
%     YpredTest{k} = YpredTest_scl{k}*y_std + y_ave;
%     y_test{k} = y_test_scl{k} * y_std + y_ave;
% 
%     tmp_ymax_test(k) = max([YpredTest{k};y_test{k}]);
%     tmp_ymin_test(k) = min([YpredTest{k};y_test{k}]);
% 
%     error_tst(k) = error_all(y_test{k}, YpredTest{k});
%     r_sq_tst(k) = error_tst(k).R_sq;
% 
%     figure(16)
%     scatter(y_test{k},YpredTest{k},100, 'filled')
% end
% 
% figure(15);hold off
% tmp_max = max(tmp_ymax_train(k)) * 1.2;
% tmp_min = min(tmp_ymin_train(k)) / 1.2;
% axis equal;adjfig; h = refline(1,0);h.Color ='black';h.LineWidth = 1.5;
% r_sq_train_ave = mean(r_sq_train);
% r_sq_train_str = strcat("R^2 = ",num2str(round(r_sq_train_ave,3))); 
% annot = annotation('textbox',[0.55 0.15 0.2 0.15],'FitBoxToText','on','String',r_sq_train_str,'EdgeColor','none');
% set(annot,'FontSize',16,'FontName','Arial')
% legend({'1st' '2nd' '3rd' '4th' '5th'}, 'location', 'NW');
% legend('boxoff')
% xlim([tmp_min, tmp_max]); ylim([tmp_min, tmp_max]);
% 
% figure(16);hold off
% tmp_max = max(tmp_ymax_test(k)) * 1.2;
% tmp_min = min(tmp_ymin_test(k)) / 1.2;
% axis equal;adjfig; h =refline(1,0);h.Color ='black';h.LineWidth = 1.5;
% r_sq_tst_ave = mean(r_sq_tst);
% r_sq_tst_str = strcat("R^2 = ",num2str(round(r_sq_tst_ave,3))); 
% annot = annotation('textbox',[0.55 0.15 0.2 0.15],'FitBoxToText','on','String',r_sq_tst_str,'EdgeColor','none');
% set(annot,'FontSize',16,'FontName','Arial')
% legend({'1st' '2nd' '3rd' '4th' '5th'}, 'location', 'NW');
% legend('boxoff')
% xlim([tmp_min, tmp_max]); ylim([tmp_min, tmp_max]);
% 
% lst_large_B_sort
% data_type
% result_message = append(num2str(size(B_slct,1))," features were selected based on LASSO,", "(test", r_sq_tst_str,")")
% 
% % record
% fig_idx.Lasso{1} = 14;
% fig_idx.Lasso{2} = 15;
% fig_idx.Lasso{3} = 16;
% 
% savefig_name.Lasso{1} = strcat('06_Lasso_plot.fig');saveas(figure(fig_idx.Lasso{1}),savefig_name.Lasso{1})
% savefig_name.Lasso{2} = strcat('06_Lasso_Train.fig');saveas(figure(fig_idx.Lasso{2}),savefig_name.Lasso{2})
% savefig_name.Lasso{3} = strcat('06_Lasso_Test.fig');saveas(figure(fig_idx.Lasso{3}),savefig_name.Lasso{3})
% savevar_name.Lasso{2} = strcat('06_Lasso_result.mat');save(savevar_name.Lasso{2})
% 
% [calc_summary_info,calc_folder_last] = matlabnote_save("Lasso",note_info,savefig_name,fig_idx,savevar_name);
% matlabnote_ppt("Lasso",note_info,savefig_name,fig_idx,savevar_name,calc_folder_last);
% 
% 
% %% PLSR  
% save("00_note_info.mat","note_info")
% savevar_name.PLSR{1} = '07_inputs.mat';save(savevar_name.PLSR{1})
% auto_input_organizer;
% 
% % if strcmp(inverse_id,"true") == 1
% %     data_type = append(sample_name," ",num2str(size(scl_myTbl_matrix,1))," samples, ",num2str(size(scl_myTbl_matrix,2)-1)," features with inverse, cluster_cutoff < ", num2str(cutoff_para));
% % else
% %     data_type = append(sample_name," ",num2str(size(scl_myTbl_matrix,1))," samples, ",num2str(size(scl_myTbl_matrix,2)-1)," features, cluster_cutoff < ", num2str(cutoff_para));
% % end
% PLSR_comp = ML_para.PLSR.PLSR_comp; % Max10
% num_large_corr = ML_para.PLSR.num_large_corr; % Max10
% 
% all_featuresName_tbl = table(scl_myTbl.Properties.VariableNames');
% all_featuresName_tbl(end,:) = [];
% descriptor_name_all = scl_myTbl.Properties.VariableNames;
% descriptor_name_all(end) = [];
% 
% x = scl_myTbl_matrix(:,1:end-1);
% y = myTbl_matrix(:,end);
% 
% x_ave = mean(x,1);x_std = std(x,1);
% y_ave = mean(y,1);y_std = std(y,1);
% 
% x_scl = (x - x_ave) ./ x_std;
% y_scl = (y - y_ave) ./ y_std;
% 
% 
% kfold = ML_para.k_fold;
% rng(1)
% cv = cvpartition(size(x,1),'KFold',kfold);
% [Xl, Yl, Xs, Ys, beta, pctVar, PLSmsep,stats] = plsregress(x_scl, y_scl, PLSR_comp, 'CV',cv);
% [Xl_r, Yl_r, Xs_r, Ys_r, beta_r, pctVar_r, PLSmsep_r,stats_r] = plsregress_rescl(x_scl, y_scl, PLSR_comp, 'CV',cv);
% figure(17);fig_posi = get(gcf,'Position');close(17);figure(17);set(gcf,'Position',fig_posi) %4
% plot(1:PLSR_comp, cumsum(100 * pctVar(2,:)), '-bo');
% xlabel('Number of PLS components');
% ylabel('Percent Variance Explained in Y');
% adjfig
% 
% figure(18);fig_posi = get(gcf,'Position');close(18);figure(18);set(gcf,'Position',fig_posi)
% plot(1:size(x_scl, 2), stats.W, '-');
% xlabel('Variable');
% ylabel('PLS Weight');
% legend({'1st Component' '2nd Component' '3rd Component'}, 'location', 'NW');
% 
% adjfig
% 
% 
% figure(19);fig_posi = get(gcf,'Position');close(19);figure(19);set(gcf,'Position',fig_posi);hold on;
% xlabel("True current [mA/cm^2]");ylabel("Predicted current [mA/cm^2]")
% figure(20);fig_posi = get(gcf,'Position');close(20);figure(20);set(gcf,'Position',fig_posi);hold on
% xlabel("True current [mA/cm^2]");ylabel("Predicted current [mA/cm^2]")
% 
% 
% for k = 1:kfold
%     % training 
%     train_idx{k} = training(cv,k);
%     x_train_scl{k} = x_scl(train_idx{k},:);
%     y_train_scl{k} = y_scl(train_idx{k},:);
%     % [Xl_tmp{k}, Yl_tmp{k}, Xs_tmp{k}, Ys_tmp{k}, .....
%     % beta_tmp{k}, pctVar_tmp{k}, PLSmsep_tmp{k},stats_tmp{k}] = plsregress(x_train_scl{k}, y_train_scl{k}, PLSR_comp, 'CV','resubstitution');
% 
%     YpredTrain_scl{k} = Xs(train_idx{k},:) * Yl' + mean(y_train_scl{k});
%     YpredTrain{k} = YpredTrain_scl{k}*y_std + y_ave;
%     y_train{k} = y_train_scl{k}*y_std + y_ave;
% 
%     tmp_ymax_train(k) = max([YpredTrain{k};y_train{k}]);
%     tmp_ymin_train(k) = min([YpredTrain{k};y_train{k}]);
% 
%     error_trn(k) = error_all(y_train{k}, YpredTrain{k});
%     r_sq_train(k) = error_trn(k).R_sq;
% 
%     figure(19)
%     scatter(y_train{k},YpredTrain{k},100, 'filled')
% 
%     % test 
%     test_idx{k} = test(cv,k);
%     x_test_scl{k} = x_scl(test_idx{k},:);
%     y_test_scl{k} = y_scl(test_idx{k});
% 
%     YpredTest_scl{k} = Xs(test_idx{k},:) * Yl' + mean(y_test_scl{k});
%     YpredTest{k} = YpredTest_scl{k}*y_std + y_ave;
%     y_test{k} = y_test_scl{k} * y_std + y_ave;
% 
%     tmp_ymax_test(k) = max([YpredTest{k};y_test{k}]);
%     tmp_ymin_test(k) = min([YpredTest{k};y_test{k}]);
% 
%     error_tst(k) = error_all(y_test{k}, YpredTest{k});
%     r_sq_tst(k) = error_tst(k).R_sq;
% 
%     figure(20)
%     scatter(y_test{k},YpredTest{k},100, 'filled')
% end
% 
% figure(19);hold off
% tmp_max = max(tmp_ymax_train(k)) * 1.2;
% tmp_min = min(tmp_ymin_train(k)) / 1.2;
% axis equal;adjfig; h = refline(1,0);h.Color ='black';h.LineWidth = 1.5;
% r_sq_train_ave = mean(r_sq_train);
% r_sq_train_str = strcat("R^2 = ",num2str(round(r_sq_train_ave,3))); 
% annot = annotation('textbox',[0.55 0.15 0.2 0.15],'FitBoxToText','on','String',r_sq_train_str,'EdgeColor','none');
% set(annot,'FontSize',16,'FontName','Arial')
% legend({'1st' '2nd' '3rd' '4th' '5th'}, 'location', 'NW');
% legend('boxoff')
% xlim([tmp_min, tmp_max]); ylim([tmp_min, tmp_max]);
% 
% figure(20);hold off
% tmp_max = max(tmp_ymax_test(k)) * 1.2;
% tmp_min = min(tmp_ymin_test(k)) / 1.2;
% axis equal;adjfig; h =refline(1,0);h.Color ='black';h.LineWidth = 1.5;
% r_sq_tst_ave = mean(r_sq_tst);
% r_sq_tst_str = strcat("R^2 = ",num2str(round(r_sq_tst_ave,3))); 
% annot = annotation('textbox',[0.55 0.15 0.2 0.15],'FitBoxToText','on','String',r_sq_tst_str,'EdgeColor','none');
% set(annot,'FontSize',16,'FontName','Arial')
% legend({'1st' '2nd' '3rd' '4th' '5th'}, 'location', 'NW');
% legend('boxoff')
% xlim([tmp_min, tmp_max]); ylim([tmp_min, tmp_max]);
% 
% result_message = append(num2str(size(descriptor_name_all,2))," features were used based on PLSR,", "(test", r_sq_tst_str,")")
% PLS_descriptor_all = cell2table((descriptor_name_all)');
% PLS_descriptor_all.Properties.VariableNames{1} = 'Descriptor_name';
% PLS_coeff_tbl = [PLS_descriptor_all,array2table(beta(2:end),'VariableNames',{'Beta'})];
% PLS_coeff_tbl = [PLS_coeff_tbl, array2table(abs(beta(2:end)),'VariableNames',{'Beta_abs'})];
% PLS_coeff_tbl_sort = sortrows(PLS_coeff_tbl,"Beta_abs",'descend');
% PLS_coeff_tbl_sort(:,3) = [];
% PLS_coeff_tbl_sort.Properties.VariableNames{2} = 'Beta';
% PLS_coeff_tbl_sort_ppt = PLS_coeff_tbl_sort;
% PLS_coeff_tbl_sort_ppt(:,3) = cellstr(string(num2str(round(table2array(PLS_coeff_tbl_sort(:,2)),3))));
% PLS_coeff_tbl_sort_ppt(:,2) = [];
% PLS_coeff_tbl_sort_ppt.Properties.VariableNames{2} = 'Beta';
% 
% %% PLSR covariance feature selection 
% 
% %% sort each component based on weight of each component
% 
% PLSR_tbl_cov = table;
% for i = 1:PLSR_comp
%     w = stats_r.W(:, i);
%     w_abs = abs(w);
%     tmp_tbl = [all_featuresName_tbl, array2table(w),array2table(w_abs)];
%     tmp_tbl.Properties.VariableNames = {strcat('Descriptor_comp_',num2str(i)),strcat('w_comp_',num2str(i)),strcat('w_',num2str(i),'_abs')};
%     tmp_tbl = sortrows(tmp_tbl,tmp_tbl.Properties.VariableNames{3},'descend');
%     tmp_tbl(:,3) = [];
%     tmp_tbl(:,3) = array2table(i*ones(size(tmp_tbl,1),1));
%     tmp_tbl.Properties.VariableNames{3} = strcat('comp',num2str(i));
%     tmp_tbl(:,4) = array2table((1:size(tmp_tbl,1))');
%     tmp_tbl.Properties.VariableNames{4} = strcat('idx',num2str(i));
%     PLSR_tbl_cov = [PLSR_tbl_cov,tmp_tbl];
% end
% 
% 
% %% calculate the value of covariance loop 
% 
% cov_thrshld = ML_para.PLSR.cov_thrshld; 
% maxCmp = ML_para.PLSR.cov_maxCmp; % usually 2, check percent variance plot saturation.
% 
% % confirm the covariance of 1st component with non-sorted matrix
% X0 = scl_myTbl_matrix(:, 1 : end-1);
% Y0 = scl_myTbl_matrix(:, end);
% 
% 
% for m = 1:maxCmp
%     W{m} = stats_r.W(:, m);
%     t{m} = X0*W{m};
%     Covariance{m} = (t{m})'*Y0;
%     strcat("Covar_",num2str(m)," = ", num2str(Covariance{m}))
%     X0_m_t = [ ];
% 
%     for i = 1 : length(W{m})
%         descriptor_name_cov{i,m} = table2cell(PLSR_tbl_cov(i,4*m-3));
% 
%         X0_part_t = scl_myTbl(:, descriptor_name_cov{i,m});
%         X0_m_t = [X0_m_t, X0_part_t];
%         % order = Pri_comp(i, 1);
%         % W_order = PLSR_tbl.w_comp_1;
%         % W1_order = [W1_order; W_order];
%     end
%     W_order_m = table2array(PLSR_tbl_cov(:,4*m-2));
%     X0_m = table2array(X0_m_t);
% 
%     % take the multiplying value between each component and X0 table
%     % criteria = Covar_1;
%     criteria = 0;
%     numbers{m} = 1;
%     while criteria < Covariance{m}*cov_thrshld
%         numbers{m} = numbers{m} + 1;
%         % order = Pri_comp(1 : number_1, 1);
%         W_order = W_order_m(1: numbers{m});
%         Xm_part = X0_m(:, 1 : numbers{m});
%         tm_part = Xm_part*W_order;
%         criteria = tm_part'*Y0;
%     end
%     strcat("number_",num2str(m)," = ",num2str(numbers{m}))
%     criteria_c{m} = criteria;
%     % caliculate the pm and qm
%     % t{m} = X0*W{m};
%     p{m} = (X0'*t{m})/((t{m})'*t{m});
%     q{m} = (Y0'*t{m})/((t{m})'*t{m});
%     % update the X0 and Y0
%     X0 = X0 - t{m}*(p{m})';
%     Y0 = Y0 - t{m}*q{m};
% 
% end
% 
% covar_tbl = cell2table(Covariance);
% covar_tbl(2,:) = cell2table(criteria_c);
% covar_tbl(3,:) = array2table((table2array(covar_tbl(2,:))./table2array((covar_tbl(1,:)))));
% covar_tbl(4,:) = cell2table(numbers);
% covar_tbl.Properties.RowNames ={'Max','Act','ratio','NumSlct'};
% % covar_tbl = round(covar_tbl,3);
% 
% covar_tbl_ppt = splitvars(table(strsplit(string(num2str(round(cell2mat(Covariance),3))))));
% covar_tbl_ppt(2,:) = splitvars(table(strsplit(string(num2str(round(cell2mat(criteria_c),3))))));
% covar_tbl_ppt(3,:) = splitvars(table(strsplit(string(num2str(round(cell2mat(criteria_c)./cell2mat(Covariance),3))))));
% covar_tbl_ppt(4,:) = cell2table(numbers);
% covar_tbl_ppt.Properties.RowNames ={'Cov',strcat('Cov_',num2str(cov_thrshld)),'ratio','NumSlct'};
% 
% for i = 1:size(covar_tbl_ppt,2)
%     covar_tbl_ppt.Properties.VariableNames{i} = strcat('Covar',num2str(i));
% 
% end
% 
% %% Narrowing the number of features and create new myTble
% 
% % descriptor_name_cov_concate = [];
% % descriptor_weight_cov_concate = [];
% % descriptor_weight_cov_concate_re = [];
% % % descriptor_name_tbl = cell2table(descriptor_name_cov);
% % 
% % for m = 1:maxCmp
% % 
% %     descriptor_name_tmp = descriptor_name_cov(1:numbers{m},m);
% %     descriptor_w_tmp = table2array(PLSR_tbl_cov(1:numbers{m},2*m));
% %     descriptor_w_tmp_re = descriptor_w_tmp * pctVar(2,m);
% %     % descriptor_w_tmp = (descriptor_w_tmp * pctVar(2,m));
% %     descriptor_weight_cov_concate_re = [descriptor_weight_cov_concate_re;descriptor_w_tmp_re];
% %     descriptor_weight_cov_concate = [descriptor_weight_cov_concate;descriptor_w_tmp];
% %     descriptor_name_cov_concate = [descriptor_name_cov_concate;descriptor_name_tmp];
% % 
% % end
% %  descriptor_PLSR_cov = [table(descriptor_name_cov_concate),array2table(descriptor_weight_cov_concate),array2table(descriptor_weight_cov_concate_re)];
% % 
% % [selected_descriptors_cov,idx_unique] = unique(string(descriptor_name_cov_concate),'stable');
% % idx_removed = setdiff(([1:length(descriptor_name_cov_concate)])',idx_unique);
% % 
% % % [~,idx_overlap] = unique(string(descriptor_name_cov_concate),'stable');
% % descriptor_c = cellstr(selected_descriptors_cov');
% % descriptor_coeff_PLSR = descriptor_weight_cov_concate(idx_unique);
% 
% 
% %
% % PLSR_tbl_cov_re = PLSR_tbl_cov;
% % for  m = 1:maxCmp
% % PLSR_tbl_cov_re(:,2*m) =  array2table(abs(pctVar(2,m) * table2array(PLSR_tbl_cov_re(:,2*m))));
% % end
% % % PLSR_tbl_cov_re(:,2*maxCmp+1:end) = [];
% % 
% % PLSR_tbl_cov_re0 = PLSR_tbl_cov_re;
% % 
% % [C,ia,ib] = intersect(PLSR_tbl_cov_re.Descriptor_comp_1(1:numbers{1}),PLSR_tbl_cov_re.Descriptor_comp_2(1:numbers{2}),'stable');
% % [CC,iaa,ibb ] = union(PLSR_tbl_cov_re.Descriptor_comp_1(1:numbers{1}),PLSR_tbl_cov_re.Descriptor_comp_2(1:numbers{2}),'stable');
% % [CCC,iaaa,ibbb] = setxor(PLSR_tbl_cov_re.Descriptor_comp_1(1:numbers{1}),PLSR_tbl_cov_re.Descriptor_comp_2(1:numbers{2}),'stable');
% % 
% % C2 = table(C);
% % C2(:,2) = array2table(PLSR_tbl_cov_re.w_comp_1(ia) + PLSR_tbl_cov_re.w_comp_2(ib));
% % C2.Properties.VariableNames = {'Descriptor_Name','ConctatedWeight'};
% % CCC2 = table(CCC);
% % CCC2(:,2) = [array2table(PLSR_tbl_cov_re.w_comp_1(iaaa));array2table(PLSR_tbl_cov_re.w_comp_2(ibbb))];
% % CCC2.Properties.VariableNames = {'Descriptor_Name','ConctatedWeight'};
% % CC2 = table(CC);
% % D = [C2;CCC2];
% % 
% % D2 = sortrows(D,2,'descend');
% % 
% % %%
% PLSR_tbl_cov_re = PLSR_tbl_cov;
% 
% for  m = 1:maxCmp
% PLSR_tbl_cov_re(:,4*m-2) =  array2table(abs(pctVar(2,m) * table2array(PLSR_tbl_cov_re(:,4*m-2))));
% end
% % PLSR_tbl_cov_re(:,2*maxCmp+1:end) = [];
% 
% PLSR_tbl_cov_re0 = PLSR_tbl_cov_re(1:numbers{1},1:4);
% % PLSR_tbl_cov_re0(:,3) = array2table(ones(size(PLSR_tbl_cov_re0,1),1));
% comp_idx = '1';
% 
% 
% for m = 1:maxCmp-1
%     [overlap_dcp,i_ovrlp_a,i_ovrlp_b] = intersect(table2cell(PLSR_tbl_cov_re0(:,1)),table2cell(PLSR_tbl_cov_re((1:numbers{m+1}),4*m+1)),'stable');
%     [sum_dcp] = union(table2cell(PLSR_tbl_cov_re0(:,1)),table2cell(PLSR_tbl_cov_re((1:numbers{m+1}),4*m+1)),'stable');
%     [or_dcp,i_or_a,i_or_b] = setxor(table2cell(PLSR_tbl_cov_re0(:,1)),table2cell(PLSR_tbl_cov_re((1:numbers{m+1}),4*m+1)),'stable');
%     if m == 1
%     comp_idx2 = num2str(table2array(PLSR_tbl_cov_re0(i_ovrlp_a,4)));
%     else
%       comp_idx2 = PLSR_tbl_cov_re0.Comp_idx(i_ovrlp_a);
%     end
% 
%     overlap_dcp_tbl = table(overlap_dcp);
%     overlap_dcp_tbl(:,2) = array2table(table2array(PLSR_tbl_cov_re0(i_ovrlp_a,2)) + table2array(PLSR_tbl_cov_re(i_ovrlp_b,4*m+2)));
%     overlap_dcp_tbl.Properties.VariableNames = {'Descriptor_Name','ConctatedWeight'};
%     comp_idx = strcat(comp_idx,'-',num2str(m+1));
%     overlap_dcp_tbl(:,3) = table(string(comp_idx));
%     comp_idx2 = strcat(comp_idx2,'-',num2str(table2array(PLSR_tbl_cov_re(i_ovrlp_b,4*m+4))));
%     overlap_dcp_tbl(:,4) = table(string(comp_idx2));
%     overlap_dcp_tbl.Properties.VariableNames = {'Descriptor_Name','ConctatedWeight','PC_idx','Comp_idx'};
%     % overlap_dcp_idx = table(string(strcat(comp_idx,'-',num2str(m+1))));
% 
% 
%     or_dcp_tbl = table(or_dcp);
%     or_dcp_tbl(:,2) = array2table([table2array(PLSR_tbl_cov_re0(i_or_a,2));table2array(PLSR_tbl_cov_re(i_or_b,4*m+2))]);
%     or_dcp_tbl(:,3) = array2table([table2array(PLSR_tbl_cov_re0(i_or_a,3));table2array(PLSR_tbl_cov_re(i_or_b,4*m+3))]);
%     or_dcp_tbl(:,4) = array2table([table2array(PLSR_tbl_cov_re0(i_or_a,4));table2array(PLSR_tbl_cov_re(i_or_b,4*m+4))]);
%     % or_dcp_tbl(:,3) = table(string(strcat(comp_idx,'-',num2str(m+1))));
% 
%     or_dcp_tbl.Properties.VariableNames = {'Descriptor_Name','ConctatedWeight','PC_idx','Comp_idx'};
% 
%     sum_dcp_tbl = table(sum_dcp);
%     PLSR_tbl_cov_re0 = [overlap_dcp_tbl;or_dcp_tbl];
%     PLSR_tbl_cov_re0 = sortrows(PLSR_tbl_cov_re0,2,'descend');
% 
% end
% 
% PLSR_tbl_cov_re0_ppt = PLSR_tbl_cov_re0(:,1);
% PLSR_tbl_cov_re0_ppt(:,2) = cellstr(string(num2str(round(table2array(PLSR_tbl_cov_re0(:,2)),3))));
% PLSR_tbl_cov_re0_ppt(:,3) = PLSR_tbl_cov_re0(:,3);
% PLSR_tbl_cov_re0_ppt(:,4) = PLSR_tbl_cov_re0(:,4);
% PLSR_tbl_cov_re0_ppt.Properties.VariableNames = PLSR_tbl_cov_re0.Properties.VariableNames;
% 
% % PLS_coeff_tbl_sort_ppt(:,3) = cellstr(string(num2str(round(table2array(PLS_coeff_tbl_sort(:,2)),3))));
% PLSR_cov_summary_tbl = table;
% result_message = append(num2str(size(PLSR_tbl_cov_re0,1))," features were used based on PLSR,", "(test", r_sq_tst_str,")")
% 
% % record 
% 
% fig_idx.PLSR{1} = 17;
% fig_idx.PLSR{2} = 18;
% fig_idx.PLSR{3} = 19;
% fig_idx.PLSR{4} = 20;
% 
% savefig_name.PLSR{1} = strcat('07_PLSR_Variance_plot.fig');saveas(figure(fig_idx.PLSR{1}),savefig_name.PLSR{1})
% savefig_name.PLSR{2} = strcat('07_PLSR_Weight_plot.fig');saveas(figure(fig_idx.PLSR{2}),savefig_name.PLSR{2})
% savefig_name.PLSR{3} = strcat('07_PLSR_Train.fig');saveas(figure(fig_idx.PLSR{3}),savefig_name.PLSR{3})
% savefig_name.PLSR{4} = strcat('07_PLSR_Test.fig');saveas(figure(fig_idx.PLSR{4}),savefig_name.PLSR{4})
% 
% savevar_name.PLSR{2} = strcat('07_PLSR_result.mat');save(savevar_name.PLSR{2})
% 
% [calc_summary_info,calc_folder_last] = matlabnote_save("PLSR",note_info,savefig_name,fig_idx,savevar_name);
% matlabnote_ppt("PLSR",note_info,savefig_name,fig_idx,savevar_name,calc_folder_last);
% 
% 
% %% Random forest
% x = scl_myTbl_matrix(:,1:end-1);
% y = myTbl_matrix(:,end);
% 
% x_ave = mean(x,1);x_std = std(x,1);
% y_ave = mean(y,1);y_std = std(y,1);
% 
% x_scl = (x - x_ave) ./ x_std;
% y_scl = (y - y_ave) ./ y_std;
% 
% 
% kfold = ML_para.k_fold;
% rng(1)
% cv = cvpartition(size(x,1),'KFold',kfold);
% 
% 
% 
% % Define the number of trees in the Random Forest
% numTrees = 100; % You can adjust this value based on your specific needs
% 
% % Define the number of folds for cross-validation
% numFolds = kfold; % You can adjust this value based on your specific needs
% options.Ensemble = ML_para.Ensemble;
% % Perform cross-validation with Random Forest regression
% % cvmodel = crossval(@(xtr, ytr, xte, yte) RandomForestRegression(numTrees, xtr, ytr), x_scl, y_scl, 'CVPartition', cv);
%   t = templateTree('Reproducible',true, ...
%   'MaxNumSplits' ,options.Ensemble.MaxNumSplits,...
%     'MinLeafSize' ,options.Ensemble.MinLeafSize,...
%     'MinParentSize' ,options.Ensemble.MinParentSize,...
%         'QuadraticErrorTolerance',options.Ensemble.Tolerance, ...
%         'NumVariablesToSample','all');
%   ml_mdl = fitrensemble(x_scl, y_scl, ...
%         'Method',options.Ensemble.Method, ...
%         'NumLearningCycles',options.Ensemble.NumCyle, ...
%         'Learners',t);
%   cv_mdl = crossval(ml_mdl,'CVPartition',cv);
% % % Estimate predictor importance
% % importance = predictorImportance(cv_mdl);
% % 
% % % Plot the predictor importance
% % bar(importance);
% % xlabel('Predictor Index');
% % ylabel('Importance');
% % title('Predictor Importance in Random Forest (Regression)');
% %%
% impOOB = oobPermutedPredictorImportance(cv_mdl);
% % impOOB は予測子の重要度の推定が格納されている 1 行 7 列のベクトルで、Mdl.PredictorNames 内の予測子に対応しています。この推定は、多くのレベルが含まれている予測子に偏ってはいません。
% % 
% % 予測子の重要度の推定を比較します。
% 
% figure
% bar(impOOB)
% title('Unbiased Predictor Importance Estimates')
% xlabel('Predictor variable')
% ylabel('Importance')
% h = gca;
% h.XTickLabel = Mdl.PredictorNames;
% h.XTickLabelRotation = 45;
% h.TickLabelInterpreter = 'none';
% %%
% figure(18);close(18);figure(18);hold on;
% xlabel("True current [mA/cm^2]");ylabel("Predicted current [mA/cm^2]")
% figure(19);close(19);figure(19);hold on
% xlabel("True current [mA/cm^2]");ylabel("Predicted current [mA/cm^2]")
% 
% rng(2);
%     t = templateTree('Reproducible',true, ...
%   'MaxNumSplits' ,options.Ensemble.MaxNumSplits,...
%     'MinLeafSize' ,options.Ensemble.MinLeafSize,...
%     'MinParentSize' ,options.Ensemble.MinParentSize,...
%         'QuadraticErrorTolerance',options.Ensemble.Tolerance, ...
%         'NumVariablesToSample','all');
% 
% for k = 1:kfold
%     % training 
%     train_idx{k} = training(cv,k);
%     x_train_scl{k} = x_scl(train_idx{k},:);
%     y_train_scl{k} = y_scl(train_idx{k},:);
%     % [Xl_tmp{k}, Yl_tmp{k}, Xs_tmp{k}, Ys_tmp{k}, .....
%     % beta_tmp{k}, pctVar_tmp{k}, PLSmsep_tmp{k},stats_tmp{k}] = plsregress(x_train_scl{k}, y_train_scl{k}, PLSR_comp, 'CV','resubstitution');
% 
%     rng(2);
%     ml_mdl{k} = fitrensemble(xtrain_scl{k}, ytrain_scl{k}, ...
%         'Method',options.Ensemble.Method, ...
%         'NumLearningCycles',options.Ensemble.NumCyle, ...
%         'Learners',t);
%     YpredTrain_scl{k} = oobPredict(ml_mdl{k});
%     YpredTrain{k} = YpredTrain_scl{k}*y_std + y_ave;
%     y_train{k} = y_train_scl{k}*y_std + y_ave;
% 
%     tmp_ymax_train(k) = max([YpredTrain{k};y_train{k}]);
%     tmp_ymin_train(k) = min([YpredTrain{k};y_train{k}]);
% 
%     error_trn(k) = error_all(y_train{k}, YpredTrain{k});
%     r_sq_train(k) = error_trn(k).R_sq;
% 
%     figure(16)
%     scatter(y_train{k},YpredTrain{k},100, 'filled')
% 
%     % test 
%     test_idx{k} = test(cv,k);
%     x_test_scl{k} = x_scl(test_idx{k},:);
%     y_test_scl{k} = y_scl(test_idx{k});
% 
%     YpredTest_scl{k} = Xs(test_idx{k},:) * Yl' + mean(y_test_scl{k});
%     YpredTest{k} = YpredTest_scl{k}*y_std + y_ave;
%     y_test{k} = y_test_scl{k} * y_std + y_ave;
% 
%     tmp_ymax_test(k) = max([YpredTest{k};y_test{k}]);
%     tmp_ymin_test(k) = min([YpredTest{k};y_test{k}]);
% 
%     error_tst(k) = error_all(y_test{k}, YpredTest{k});
%     r_sq_tst(k) = error_tst(k).R_sq;
% 
%     figure(17)
%     scatter(y_test{k},YpredTest{k},100, 'filled')
% end
% 
% figure(18);hold off
% tmp_max = max(tmp_ymax_train(k)) * 1.2;
% tmp_min = min(tmp_ymin_train(k)) * 1.2;
% xlim([tmp_min, tmp_max]); ylim([tmp_min, tmp_max]);
% axis equal;adjfig; h = refline(1,0);h.Color ='black';h.LineWidth = 1.5;
% r_sq_train_ave = mean(r_sq_train);
% r_sq_train_str = strcat("R^2 = ",num2str(round(r_sq_train_ave,3))); 
% annot = annotation('textbox',[0.55 0.15 0.2 0.15],'FitBoxToText','on','String',r_sq_train_str,'EdgeColor','none');
% set(annot,'FontSize',16,'FontName','Arial')
% legend({'1st' '2nd' '3rd' '4th' '5th'}, 'location', 'NW');
% legend('boxoff')
% 
% figure(19);hold off
% tmp_max = max(tmp_ymax_test(k)) * 1.2;
% tmp_min = min(tmp_ymin_test(k)) * 1.2;
% xlim([tmp_min, tmp_max]); ylim([tmp_min, tmp_max]);
% axis equal;adjfig; h =refline(1,0);h.Color ='black';h.LineWidth = 1.5;
% r_sq_tst_ave = mean(r_sq_tst);
% r_sq_tst_str = strcat("R^2 = ",num2str(round(r_sq_tst_ave,3))); 
% annot = annotation('textbox',[0.55 0.15 0.2 0.15],'FitBoxToText','on','String',r_sq_tst_str,'EdgeColor','none');
% set(annot,'FontSize',16,'FontName','Arial')
% legend({'1st' '2nd' '3rd' '4th' '5th'}, 'location', 'NW');
% legend('boxoff')
% 
% impOOB = oobPermutedPredictorImportance(Mdl);
% 
% %% Variables to keep 
% % list the variable you want to keep until last section to avoid automatic
% % removal of varialbes that you need by auto_input_organizer.
% % auto_input_organizer remove the variables which are not used just afeter
% % the line that auto_input_organizer is located. 
% myTbl;
% data_type;
% cluster_tbl;
% scl_myTbl_clust_c;
% scl_myTbl_clust_concate;
% scl_myTbl_RSTD ;
% scl_myTbl_matrix_RSTD;
%% sub functions
% This is the start of sub functions
%% Filter sequential
 function criterion = mycorr(X)
    if size(X,2) < 2
        criterion = 0;
    else
        p = size(X,2);
        % R = corr(X,"Rows","pairwise");
        R = corrcoef(X,"Rows","pairwise"); % mostly same because of autoscaling
        R(logical(eye(p))) = NaN;
        criterion = max(abs(R),[],"all");
    end
 end

 % %% for negative correlation inversion
 % function criterion = mycorr2(X)
 % if size(X,2) < 2
 %     criterion = 0;
 % else
 %     p = size(X,2);
 %     % R = corr(X,"Rows","pairwise");
 %     R = corrcoef(X,"Rows","pairwise"); % mostly same because of autoscaling
 %     R(logical(eye(p))) = NaN;
 %     criterion = max(-(R),[],"all");
 % end
 % end

%% Custom distance function by 1-abs(cosine) to consider negative correlation 
function D2 = dist_cos_abs(Zi,Zj)

n = size(Zj,1);
D2 = zeros(1,n);


    for j = 1:n
        cosineSim = abs(dot(Zi,Zj(j,:)) / (norm(Zi) * norm(Zj(j,:))));
        D2(j) = 1 - cosineSim;
    end


end
%% cos-s map 
function cos_sim_tbl = cos_s_map(scl_myTbl,fig_num)
scl_myTbl_matrix = table2array(scl_myTbl);
dat_matrix = scl_myTbl_matrix;
scl_myTbl_size = size(scl_myTbl_matrix);
cos_sim = zeros(scl_myTbl_size(2),scl_myTbl_size(2));
for i = 1:scl_myTbl_size(2)
    for j = 1:scl_myTbl_size(2)
        cos_sim(i,j) = (dat_matrix(:,i)' * dat_matrix(:,j)) / (norm(dat_matrix(:,i)) * norm(dat_matrix(:,j)));     
    end       
end
%  cos_sim = dat_matrix' * dat_matrix;
cos_sim_tbl = array2table(cos_sim);
cos_sim_tbl.Properties.VariableNames = scl_myTbl.Properties.VariableNames;
cos_sim_tbl.Properties.RowNames = scl_myTbl.Properties.VariableNames;
%     cos_sim = dat_matrix' * dat_matrix;
heatmap_ax = cos_sim_tbl.Properties.VariableNames;
for i = 1:length(heatmap_ax)
    heatmap_ax{i} = strrep(heatmap_ax{i},'_','\_');
end

figure(fig_num);
p = get(gcf,'Position');close(fig_num);figure(fig_num);set(gcf,'Position',p)
set(figure(fig_num),'Color', [1 1 1]);
h = heatmap(heatmap_ax,heatmap_ax,cos_sim);
h.Colormap = jet;
h.FontSize  = 10;h.FontName ='Arial';
clim([-1 1])

end


%%  svr
function criteria = svm_error(xtrain, ytrain, xtest, ytest, options,y_ave,y_std)
    svr_mdl = fitrsvm(xtrain, ytrain, "KernelFunction", options.SVR.kernel, "Epsilon", options.SVR.epsilon);
    predY_scl = predict(svr_mdl, xtest);
    predY = predY_scl * y_std + y_ave;
    ytest_re = ytest * y_std + y_ave;
    [Errors] = error_all(ytest_re, predY);
    criteria = (1-Errors.R_sq);
end

%% gpr
function criteria = gpr_error(xtrain, ytrain, xtest, ytest, options,y_ave,y_std)
    gpr_mdl = fitrgp(xtrain, ytrain, ...
            'BasisFunction', options.GPR.Basis, ...
    'KernelFunction', options.GPR.Kernel, ...
    'FitMethod', options.GPR.FitMethod, ...
    'PredictMethod', options.GPR.PredictMethod, ...
    'Sigma', options.GPR.Sigma);
    predY_scl = predict(gpr_mdl, xtest);
    predY = predY_scl * y_std + y_ave;
    ytest_re = ytest * y_std + y_ave;
    [Errors] = error_all(ytest_re, predY);
    criteria = (1-Errors.R_sq);

end

%% Tree
function criteria = tree_error(xtrain, ytrain, xtest, ytest, options, y_ave, y_std)
rng(1)
tree_mdl = fitrtree(xtrain, ytrain, ...
    'MaxNumSplits' ,options.Tree.MaxNumSplits,...
    'MinLeafSize' ,options.Tree.MinLeafSize,...
    'MinParentSize' ,options.Tree.MinParentSize,...
    'Reproducible',true);
predY_scl = predict(tree_mdl,xtest);
predY = predY_scl * y_std + y_ave;
ytest_re = ytest * y_std + y_ave;
[Errors] = error_all(ytest_re, predY);
criteria = (1-Errors.R_sq);

end

%% Ensemble
function criteria = ensemble_error(xtrain, ytrain, xtest, ytest, options, y_ave, y_std)
rng(2);
    t = templateTree('Reproducible',true, ...
  'MaxNumSplits' ,options.Ensemble.MaxNumSplits,...
    'MinLeafSize' ,options.Ensemble.MinLeafSize,...
    'MinParentSize' ,options.Ensemble.MinParentSize,...
        'QuadraticErrorTolerance',options.Ensemble.Tolerance, ...
        'NumVariablesToSample','all');
if strcmp(options.Ensemble.Method, "LSBoost") == 1
%     rng(2);
%     t = templateTree('Reproducible',true, ...
%   'MaxNumSplits' ,options.Ensemble.MaxNumSplits,...
%     'MinLeafSize' ,options.Ensemble.MinLeafSize,...
%     'MinParentSize' ,options.Ensemble.MinParentSize,...
%         'QuadraticErrorTolerance',options.Ensemble.Tolerance, ...
%         'NumVariablesToSample','all');
%     rng(2);
   
    ensemble_mdl = fitrensemble(xtrain, ytrain, ...
        'Method',options.Ensemble.Method, ...
        'NumLearningCycles',options.Ensemble.NumCyle, ...
        'Learners',t,...
        'LearnRate',options.Ensemble.LearnRate);
elseif strcmp(options.Ensemble.Method, "Bag") == 1
    rng(2);
    s = RandStream('mlfg6331_64');
    ensemble_mdl = fitrensemble(xtrain, ytrain, ...
        'Method',options.Ensemble.Method, ...
        'NumLearningCycles',options.Ensemble.NumCyle, ...
        'Learners',t, ...
        'Options',statset('UseParallel',true,'UseSubstreams',true, 'Streams',s));
% elseif strcmp(options.Ensemble.BagMethod, "TreeBagger") == 1
%     rng(2);
%     ensemble_mdl = TreeBagger(options.Ensemble.NumTrees,xtrain, ytrain, ...
%         'Method','regression', ...
%         MinLeafSize=options.Ensemble.MinLeafSize, ...
%         NumPredictorsToSample = 'all', ...
%         OOBPrediction='on', ...
%         OOBPredictorImportance='on');

% else
%     rng(2);
%     ensemble_mdl = fitrensemble(xtrain, ytrain, ...
%         'Method',options.Ensemble.Method, ...
%         'NumLearningCycles',options.Ensemble.NumCyle);
end


predY_scl = predict(ensemble_mdl,xtest);
predY = predY_scl * y_std + y_ave;
ytest_re = ytest * y_std + y_ave;
[Errors] = error_all(ytest_re, predY);
criteria = (1-Errors.R_sq);
end

%% stepwise regression 
function  criteria = simple_stepwise_error(xtrain, ytrain, xtest, ytest, options, y_ave, y_std)
stepwise_mdl = fitlm(xtrain, ytrain ...
    ,options.simple_stepwise.modelspec);
predY_scl = predict(stepwise_mdl,xtest);
predY = predY_scl * y_std + y_ave;
ytest_re = ytest * y_std + y_ave;
[Errors] = error_all(ytest_re, predY);
criteria = (1-Errors.R_sq);

end

%% PLSR rescale
% weights are scaled in plsregress. Revised normti and Weights in simpls.
% not sure this revision is correct or not. 
function [Xloadings,Yloadings,Xscores,Yscores, ...
                    beta,pctVar,mse,stats] = plsregress_rescl(X,Y,ncomp,varargin)
%PLSREGRESS Partial least squares regression.
%   [XLOADINGS,YLOADINGS] = PLSREGRESS(X,Y,NCOMP) computes a partial least
%   squares regression of Y on X, using NCOMP PLS components or latent
%   factors, and returns the predictor and response loadings.  X is an N-by-P
%   matrix of predictor variables, with rows corresponding to observations,
%   columns to variables.  Y is an N-by-M response matrix.  XLOADINGS is a
%   P-by-NCOMP matrix of predictor loadings, where each row of XLOADINGS
%   contains coefficients that define a linear combination of PLS components
%   that approximate the original predictor variables.  YLOADINGS is an
%   M-by-NCOMP matrix of response loadings, where each row of YLOADINGS
%   contains coefficients that define a linear combination of PLS components
%   that approximate the original response variables.
%
%   [XLOADINGS,YLOADINGS,XSCORES] = PLSREGRESS(X,Y,NCOMP) returns the
%   predictor scores, i.e., the PLS components that are linear combinations of
%   the variables in X.  XSCORES is an N-by-NCOMP orthonormal matrix with rows
%   corresponding to observations, columns to components.
%
%   [XLOADINGS,YLOADINGS,XSCORES,YSCORES] = PLSREGRESS(X,Y,NCOMP)
%   returns the response scores, i.e., the linear combinations of the
%   responses with which the PLS components XSCORES have maximum covariance.
%   YSCORES is an N-by-NCOMP matrix with rows corresponding to observations,
%   columns to components.  YSCORES is neither orthogonal nor normalized.
%
%   PLSREGRESS uses the SIMPLS algorithm, and first centers X and Y by
%   subtracting off column means to get centered variables X0 and Y0.
%   However, it does not rescale the columns.  To perform partial least
%   squares regression with standardized variables, use ZSCORE to normalize X
%   and Y.
%
%   If NCOMP is omitted, its default value is MIN(SIZE(X,1)-1, SIZE(X,2)).
%
%   The relationships between the scores, loadings, and centered variables X0
%   and Y0 are
%
%      XLOADINGS = (XSCORES\X0)' = X0'*XSCORES,
%      YLOADINGS = (XSCORES\Y0)' = Y0'*XSCORES,
%
%   i.e., XLOADINGS and YLOADINGS are the coefficients from regressing X0 and
%   Y0 on XSCORES, and XSCORES*XLOADINGS' and XSCORES*YLOADINGS' are the PLS
%   approximations to X0 and Y0.  PLSREGRESS initially computes YSCORES as
%
%      YSCORES = Y0*YLOADINGS = Y0*Y0'*XSCORES,
%
%   however, by convention, PLSREGRESS then orthogonalizes each column of
%   YSCORES with respect to preceding columns of XSCORES, so that
%   XSCORES'*YSCORES is lower triangular.
%
%   [XL,YL,XS,YS,BETA] = PLSREGRESS(X,Y,NCOMP,...) returns the PLS regression
%   coefficients BETA.  BETA is a (P+1)-by-M matrix, containing intercept
%   terms in the first row, i.e., Y = [ONES(N,1) X]*BETA + Yresiduals, and
%   Y0 = X0*BETA(2:END,:) + Yresiduals.
%
%   [XL,YL,XS,YS,BETA,PCTVAR] = PLSREGRESS(X,Y,NCOMP) returns a 2-by-NCOMP
%   matrix PCTVAR containing the percentage of variance explained by the
%   model.  The first row of PCTVAR contains the percentage of variance
%   explained in X by each PLS component and the second row contains the
%   percentage of variance explained in Y.
%
%   [XL,YL,XS,YS,BETA,PCTVAR,MSE] = PLSREGRESS(X,Y,NCOMP) returns a
%   2-by-(NCOMP+1) matrix MSE containing estimated mean squared errors for
%   PLS models with 0:NCOMP components.  The first row of MSE contains mean
%   squared errors for the predictor variables in X and the second row
%   contains mean squared errors for the response variable(s) in Y.
%
%   [XL,YL,XS,YS,BETA,PCTVAR,MSE] = PLSREGRESS(...,'PARAM1',val1,...) allows
%   you to specify optional parameter name/value pairs to control the
%   calculation of MSE.  Parameters are:
%
%      'CV'      The method used to compute MSE.  When 'CV' is a positive
%                integer K, PLSREGRESS uses K-fold cross-validation.  Set
%                'CV' to a cross-validation partition, created using
%                CVPARTITION, to use other forms of cross-validation.  When
%                'CV' is 'resubstitution', PLSREGRESS uses X and Y both to
%                fit the model and to estimate the mean squared errors,
%                without cross-validation.  The default is 'resubstitution'.
%
%      'MCReps'  A positive integer indicating the number of Monte-Carlo
%                repetitions for cross-validation.  The default value is 1.
%                'MCReps' must be 1 if 'CV' is 'resubstitution'.
%      
%      'Options' A structure that specifies options that govern how PLSREGRESS
%                performs cross-validation computations. This argument can be
%                created by a call to STATSET. PLSREGRESS uses the following 
%                fields of the structure:
%                    'UseParallel'
%                    'UseSubstreams'
%                    'Streams'
%                For information on these fields see PARALLELSTATS.
%                NOTE: If supplied, 'Streams' must be of length one.
%
%   
%   [XL,YL,XS,YS,BETA,PCTVAR,MSE,STATS] = PLSREGRESS(X,Y,NCOMP,...) returns a
%   structure that contains the following fields:
%       W            P-by-NCOMP matrix of PLS weights, i.e., XSCORES = X0*W
%       T2           The T^2 statistic for each point in XSCORES
%       Xresiduals   The predictor residuals, i.e. X0 - XSCORES*XLOADINGS'
%       Yresiduals   The response residuals, i.e. Y0 - XSCORES*YLOADINGS'
%
%   Example: Fit a 10 component PLS regression and plot the cross-validation
%   estimate of MSE of prediction for models with up to 10 components.  Plot
%   the observed vs. the fitted response for the 10-component model.
%
%      load spectra
%      [xl,yl,xs,ys,beta,pctvar,mse] = plsregress(NIR,octane,10,'CV',10);
%      plot(0:10,mse(2,:),'-o');
%      octaneFitted = [ones(size(NIR,1),1) NIR]*beta;
%      plot(octane,octaneFitted,'o');
%
%   See also PCA, BIPLOT, CANONCORR, FACTORAN, CVPARTITION, STATSET,
%            PARALLELSTATS, RANDSTREAM.

% References:
%    [1] de Jong, S. (1993) "SIMPLS: an alternative approach to partial least squares
%        regression", Chemometrics and Intelligent Laboratory Systems, 18:251-263.
%    [2] Rosipal, R. and N. Kramer (2006) "Overview and Recent Advances in Partial
%        Least Squares", in Subspace, Latent Structure and Feature Selection:
%        Statistical and Optimization Perspectives Workshop (SLSFS 2005),
%        Revised Selected Papers (Lecture Notes in Computer Science 3940), C.
%        Saunders et al. (Eds.) pp. 34-51, Springer.

%   Copyright 2007-2020 The MathWorks, Inc.


if nargin > 3
    [varargin{:}] = convertStringsToChars(varargin{:});
end

if nargin < 2
    error(message('stats:plsregress:TooFewInputs'));
end

[n,dx] = size(X);
ny = size(Y,1);
if ny ~= n
    error(message('stats:plsregress:SizeMismatch'));
end

% Return at most maxncomp PLS components
maxncomp = min(n-1,dx);
if nargin < 3
    ncomp = maxncomp;
elseif ~isscalar(ncomp) || ~isnumeric(ncomp) || (ncomp~=round(ncomp)) || (ncomp<=0)
    error(message('stats:plsregress:BadNcomp'));
elseif ncomp > maxncomp
    error(message('stats:plsregress:MaxComponents', maxncomp));
end

names = {'cv'                  'mcreps'            'options'};
dflts = {'resubstitution'        1                      []   };
[cvp,mcreps,ParOptions] = internal.stats.parseArgs(names, dflts, varargin{:});

if isnumeric(cvp) && isscalar(cvp) && (cvp==round(cvp)) && (0<cvp) && (cvp<=n)
    % ok, cvp is a kfold value. It will be passed as such to crossval.
elseif isequal(cvp,'resubstitution')
    % ok
elseif isa(cvp,'cvpartition')
    if strcmp(cvp.Type,'resubstitution')
        cvp = 'resubstitution';
    else
        % ok
    end
else
    error(message('stats:plsregress:InvalidCV'));
end

if ~(isnumeric(mcreps) && isscalar(mcreps) && (mcreps==round(mcreps)) && (0<mcreps))
    error(message('stats:plsregress:InvalidMCReps'));
elseif mcreps > 1 && isequal(cvp,'resubstitution')
    error(message('stats:plsregress:InvalidResubMCReps'));
end

% Center both predictors and response, and do PLS
meanX = mean(X,1);
meanY = mean(Y,1);
X0 = X - meanX;
Y0 = Y - meanY;

if nargout <= 2
    [Xloadings,Yloadings] = simpls(X0,Y0,ncomp);
    
elseif nargout <= 4
    [Xloadings,Yloadings,Xscores,Yscores] = simpls(X0,Y0,ncomp);
    
else
    % Compute the regression coefs, including intercept(s)
    [Xloadings,Yloadings,Xscores,Yscores,Weights] = simpls(X0,Y0,ncomp);
    beta = Weights*Yloadings';
    beta = [meanY - meanX*beta; beta];
    
    % Compute the percent of variance explained for X and Y
    if nargout > 5
        pctVar = [sum(abs(Xloadings).^2,1) ./ sum(sum(abs(X0).^2,1));
                  sum(abs(Yloadings).^2,1) ./ sum(sum(abs(Y0).^2,1))];
    end
    
    if nargout > 6
        if isequal(cvp,'resubstitution')
            % Compute MSE for models with 0:ncomp PLS components, by
            % resubstitution.  CROSSVAL can handle this, but don't waste time
            % fitting the whole model again.
            mse = zeros(2,ncomp+1,class(pctVar));
            mse(1,1) = sum(sum(abs(X0).^2, 2));
            mse(2,1) = sum(sum(abs(Y0).^2, 2));
            for i = 1:ncomp
                X0reconstructed = Xscores(:,1:i) * Xloadings(:,1:i)';
                Y0reconstructed = Xscores(:,1:i) * Yloadings(:,1:i)';
                mse(1,i+1) = sum(sum(abs(X0 - X0reconstructed).^2, 2));
                mse(2,i+1) = sum(sum(abs(Y0 - Y0reconstructed).^2, 2));
            end
            mse = mse / n;
            % We now have the reconstructed values for the full model to use in
            % the residual calculation below
        else
            % Compute MSE for models with 0:ncomp PLS components, by cross-validation
            mse = plscv(X,Y,ncomp,cvp,mcreps,ParOptions);
            if nargout > 7
                % Need these for the residual calculation below
                X0reconstructed = Xscores*Xloadings';
                Y0reconstructed = Xscores*Yloadings';
            end
        end
    end
    
    if nargout > 7
        % Save the PLS weights and compute the T^2 values.
        stats.W = Weights;
        stats.T2 = sum( (abs(Xscores).^2)./var(Xscores,[],1) , 2);
        
        % Compute X and Y residuals
        stats.Xresiduals = X0 - X0reconstructed;
        stats.Yresiduals = Y0 - Y0reconstructed;
    end
end

end
%------------------------------------------------------------------------------
%SIMPLS Basic SIMPLS.  Performs no error checking.
function [Xloadings,Yloadings,Xscores,Yscores,Weights] = simpls(X0,Y0,ncomp)

[n,dx] = size(X0);
dy = size(Y0,2);

% Preallocate outputs
outClass = superiorfloat(X0,Y0);
Xloadings = zeros(dx,ncomp,outClass);
Yloadings = zeros(dy,ncomp,outClass);
if nargout > 2
    Xscores = zeros(n,ncomp,outClass);
    Yscores = zeros(n,ncomp,outClass);
    if nargout > 4
        Weights = zeros(dx,ncomp,outClass);
    end
end

% An orthonormal basis for the span of the X loadings, to make the successive
% deflation X0'*Y0 simple - each new basis vector can be removed from Cov
% separately.
V = zeros(dx,ncomp);

Cov = X0'*Y0;
for i = 1:ncomp
    % Find unit length ti=X0*ri and ui=Y0*ci whose covariance, ri'*X0'*Y0*ci, is
    % jointly maximized, subject to ti'*tj=0 for j=1:(i-1).
    [ri,si,ci] = svd(Cov,'econ'); ri = ri(:,1); ci = ci(:,1); si = si(1);
    ti = X0*ri;
    % normti = norm(ti); ti = ti ./ normti; % ti'*ti == 1
    normti = norm(ti); ti = ti; % ti'*ti == 1
    Xloadings(:,i) = X0'*ti;
    
    qi = si*ci/normti; % = Y0'*ti
    Yloadings(:,i) = qi;
    
    if nargout > 2
        Xscores(:,i) = ti;
        Yscores(:,i) = Y0*qi; % = Y0*(Y0'*ti), and proportional to Y0*ci
        if nargout > 4
            % Weights(:,i) = ri ./ normti; % rescaled to make ri'*X0'*X0*ri == ti'*ti == 1
            Weights(:,i) = ri; % rescaled to make ri'*X0'*X0*ri == ti'*ti == 1
        end
    end

    % Update the orthonormal basis with modified Gram Schmidt (more stable),
    % repeated twice (ditto).
    vi = Xloadings(:,i);
    for repeat = 1:2
        for j = 1:i-1
            vj = V(:,j);
            vi = vi - (vj'*vi)*vj;
        end
    end
    vi = vi ./ norm(vi);
    V(:,i) = vi;

    % Deflate Cov, i.e. project onto the ortho-complement of the X loadings.
    % First remove projections along the current basis vector, then remove any
    % component along previous basis vectors that's crept in as noise from
    % previous deflations.
    Cov = Cov - vi*(vi'*Cov);
    Vi = V(:,1:i);
    Cov = Cov - Vi*(Vi'*Cov);
end

if nargout > 2
    % By convention, orthogonalize the Y scores w.r.t. the preceding Xscores,
    % i.e. XSCORES'*YSCORES will be lower triangular.  This gives, in effect, only
    % the "new" contribution to the Y scores for each PLS component.  It is also
    % consistent with the PLS-1/PLS-2 algorithms, where the Y scores are computed
    % as linear combinations of a successively-deflated Y0.  Use modified
    % Gram-Schmidt, repeated twice.
    for i = 1:ncomp
        ui = Yscores(:,i);
        for repeat = 1:2
            for j = 1:i-1
                tj = Xscores(:,j);
                ui = ui - (tj'*ui)*tj;
            end
        end
        Yscores(:,i) = ui;
    end
end

end

%------------------------------------------------------------------------------
%PLSCV Efficient cross-validation for X and Y mean squared error in PLS.
function mse = plscv(X,Y,ncomp,cvp,mcreps,ParOptions)

[n,dx] = size(X);

% Return error for as many components as asked for; some columns may be NaN
% if ncomp is too large for CV.
mse = NaN(2,ncomp+1);

% The CV training sets are smaller than the full data; may not be able to fit as
% many PLS components.  Do the best we can.
if isa(cvp,'cvpartition')
    cvpType = 'partition';
    maxncomp = min(min(cvp.TrainSize)-1,dx);
    nTest = sum(cvp.TestSize);
else
    cvpType = 'Kfold';
%    maxncomp = min(min( floor((n*(cvp-1)/cvp)-1), dx));
    maxncomp = min( floor((n*(cvp-1)/cvp)-1), dx);
    nTest = n;
end
if ncomp > maxncomp
    warning(message('stats:plsregress:MaxComponentsCV', maxncomp));
    ncomp = maxncomp;
end

% Cross-validate sum of squared errors for models with 1:ncomp components,
% simultaneously.  Sum the SSEs over CV sets, and compute the mean squared
% error
CVfun = @(Xtr,Ytr,Xtst,Ytst) sseCV(Xtr,Ytr,Xtst,Ytst,ncomp);
sumsqerr = crossval(CVfun,X,Y,cvpType,cvp,'mcreps',mcreps,'options',ParOptions);
mse(:,1:ncomp+1) = reshape(sum(sumsqerr,1)/(nTest*mcreps), [2,ncomp+1]);

end
%------------------------------------------------------------------------------
%SSECV Sum of squared errors for cross-validation
function sumsqerr = sseCV(Xtrain,Ytrain,Xtest,Ytest,ncomp)

XmeanTrain = mean(Xtrain);
YmeanTrain = mean(Ytrain);
X0train = Xtrain - XmeanTrain;
Y0train = Ytrain - YmeanTrain;

% Get and center the test data
X0test = Xtest - XmeanTrain;
Y0test = Ytest - YmeanTrain;

% Fit the full model, models with 1:(ncomp-1) components are nested within
[Xloadings,Yloadings,~,~,Weights] = simpls(X0train,Y0train,ncomp);
XscoresTest = X0test * Weights;

% Return error for as many components as the asked for.
outClass = superiorfloat(Xtrain,Ytrain);
sumsqerr = zeros(2,ncomp+1,outClass); % this will get reshaped to a row by CROSSVAL

% Sum of squared errors for the null model
sumsqerr(1,1) = sum(sum(abs(X0test).^2, 2));
sumsqerr(2,1) = sum(sum(abs(Y0test).^2, 2));

% Compute sum of squared errors for models with 1:ncomp components
for i = 1:ncomp
    X0reconstructed = XscoresTest(:,1:i) * Xloadings(:,1:i)';
    sumsqerr(1,i+1) = sum(sum(abs(X0test - X0reconstructed).^2, 2));

    Y0reconstructed = XscoresTest(:,1:i) * Yloadings(:,1:i)';
    sumsqerr(2,i+1) = sum(sum(abs(Y0test - Y0reconstructed).^2, 2));
end

end

%% Random forest
function model = RandomForestRegression(numTrees, x, y)
    % Create a Random Forest regression model
    model = TreeBagger(numTrees, x, y, 'Method', 'regression');
end
%% error 
function [Errors] = error_all(a, b)
    % a = actual data
    % b = predict data
    [m, n] = size(a);

    Errors.mse = sum((a-b).^2) / (m * n);
    Errors.rmse = sqrt(Errors.mse);

    Errors.rmspe = sqrt(sum(((a - b)./a).^2) * (1/ (m * n)));
    Errors.mae = sum(abs(a - b)) * (1/(m * n));
    Errors.mape = 100 * Errors.mae;

    Errors.R_sq = 1 - (sum((a-b).^2)/sum((a-mean(a)).^2));
    
end
%% CoolBlueToWarmRedColormap
function colormap = CoolBlueToWarmRedColormap()
% Define start point, middle luminance, and end point in L*ch colorspace
% https://www.mathworks.com/help/images/device-independent-color-spaces.html
% The three components of L*ch are Luminance, chroma, and hue.
blue_lch = [54 70 4.6588]; % Starting blue point
l_mid = 40; % luminance of the midpoint
red_lch = [54 90 6.6378909]; % Ending red point
nsteps = 256;
% Build matrix of L*ch colors that is nsteps x 3 in size
% Luminance changes linearly from start to middle, and middle to end.
% Chroma and hue change linearly from start to end.
lch=[[linspace(blue_lch(1), l_mid, nsteps/2), linspace(l_mid, red_lch(1), nsteps/2)]', ... luminance column
    [linspace(blue_lch(2), red_lch(2), nsteps)]', ... chroma column
    [linspace(blue_lch(3), red_lch(3), nsteps)]']; ... hue column
% Convert L*ch to L*a*b, where a = c * cos(h) and b = c * sin(h)
lab=[lch(:,1) lch(:,2).*cos(lch(:,3)) lch(:,2).*sin(lch(:,3))];
% Convert L*a*b to RGB
colormap=lab2rgb(lab,'OutputType','uint8');
end
%% work space organization 
function auto_input_organizer
% [program_str,CurrentLine,program_part,var_clear_list] = auto_input_organizer(var_list)
var_list = evalin('base','whos');
program_info = matlab.desktop.editor.getActive;
program_text = program_info.Text;
program_str = splitlines(string(program_text));

try
    figure(-1)
catch ME

end

CurrentLine = ME.stack(2).line;
main_end_idx = contains(program_str,"% This is the start of sub functions");
L = 1:length(program_str);
main_end_line = min(L(main_end_idx))-1;
program_part = program_str(CurrentLine:main_end_line,:);

program_token = tokenizedDocument(program_part);
program_token_cat = [];

for i = 1:length(program_token)

    program_token_cat = [program_token_cat,program_token(i,1).Vocabulary];

end

var_apr = zeros(length(var_list),1);
for j = 1:length(var_list)
    var_apr(j) = sum(strcmp(program_token_cat,var_list(j).name));

end
var_clear_idx= var_apr == 0;
var_clear_list = var_list(var_clear_idx,:);

for k = 1:length(var_clear_list)
    var_clear_name = string(var_clear_list(k).name);
    message = "clearvars" + " "+ var_clear_name;
    evalin('base',message)
end

end
%%  Auto data save
%ver 1.03
function [calc_summary_info,calc_save_newfolder_name] = matlabnote_save(Target_calc,note_info,savefig_name,savefig_idx,savevar_name)

try
    savefig_name_target = getfield(savefig_name,Target_calc);
    nofig_idx1 = "off";
catch
    nofig_idx1 = "on";
end

try
    savefig_idx_target = getfield(savefig_idx,Target_calc);
    nofig_idx2 = "off";
catch
    nofig_idx2 = "on";
end

try
    savevar_name_target = getfield(savevar_name,Target_calc);
catch

end


slide_lists = note_info.Properties.RowNames;
for i = 1:size(note_info,1)
    target_slide_idx(i,1) = contains(slide_lists{i},Target_calc);
end
target_slide_info = note_info(target_slide_idx,:);
ppt_files = target_slide_info.ppt_file;
Notepath_all = fileparts(ppt_files);
Notepath_all = unique(Notepath_all);
if strcmp(nofig_idx1,"on") && strcmp(nofig_idx2,"on")
    save_files = string(savevar_name_target);
else
    save_files = [string(savefig_name_target),string(savevar_name_target)];
end
% saved_files_fullpath = cell(i);
for j = 1:length(save_files)
    saved_files_tmp = dir(save_files{j});

    if isempty(saved_files_tmp)
        saved_files_fullpath{j} = 'ismissing';
        saved_files_name{j} = 'ismissing';

    else
        saved_files_fullpath{j} = fullfile(saved_files_tmp.folder,saved_files_tmp.name);
        saved_files_name{j} = saved_files_tmp.name;
    end
end

%%

for i = 1:length(Notepath_all)
    i = 1;

    % find and move to notepath (Notepath\PPT)
    Notepath = Notepath_all{i};

    oldFolder = cd(Notepath);
    cd('..//')
    Notepath_base = cd;
    % find and move to calc folder (Notepath\Calc)
    Calc_folder = dir('Calc*');

    if isempty(Calc_folder)
        mkdir('Calc')
        Calc_folder = dir('Calc*');
    end

    Calc_backupFolder = fullfile(Calc_folder.folder,Calc_folder.name);
    cd(Calc_backupFolder);

    % create/find and move to calc save folder (Notepath\Calc\Run_XXXXX)

    calc_save_folder = dir('Run_*');
    if isempty(calc_save_folder)
        mkdir('Run_00001')
        calc_save_folder = dir('Run_*');
    end
    cd(calc_save_folder(end).name)

    % Find overlap files
    all_in_folder_ori = dir('*');
    all_in_folder_tbl = struct2table(all_in_folder_ori);
    all_in_folder = all_in_folder_tbl.name;

    ovrlp_file_idx = matches(save_files,all_in_folder);
    ovrlp_file_num = sum(ovrlp_file_idx);

    % Check the program is refreshed or the section is going ahead

    % get runnning program info
    program_info = matlab.desktop.editor.getActive;
    program_text = program_info.Text;
    programdir_source = program_info.Filename;
    [~,programname_ori,prgrm_ext] = fileparts(programdir_source);
    programname = [programname_ori,prgrm_ext];

    % get running current section info
    section_current = strcat("matlabnote_save(",'"',Target_calc,'"',",note_info,");
    section_current_idx_all = strfind(program_text,section_current);
    section_current_all = extract(program_text,section_current);

    for j = 1:length(section_current_idx_all)
        section_current_head = program_text(section_current_idx_all(j)-1);

        if ~strcmp(section_current_head,'%')
            section_current_cell{j} = section_current_all{j};
            section_current_idx(j) = section_current_idx_all(j);
        end

    end

    if length(section_current_cell) > 1
        disp([section_current, '(current calc section) is used more than once'])
    end

    % get old section info in previous running
    try
        calc_summary_info = evalin('caller','calc_summary_info');

    catch
        calc_summary_info = [];
    end

    idx_program_refresh = nan;

    if ~isempty(calc_summary_info)
        programname_old = calc_summary_info.ProgramName;
        programname_current = programname;

        idx_program_refresh =~ strcmp(programname_old,programname_current);
    end


    %         calc_summary_info_old.ProgramName = programname;
    %
    %         calc_summary_info_old.section = strcat("matlabnote_save(note_info,",'"',Target_calc,'"');
    %        if strcmp(programname_old,programname_current) || strcmp(section_old,section_current)
    %            disp('copy')
    %        end
    if ~isempty(calc_summary_info) && idx_program_refresh == 0
        section_old = calc_summary_info.section;
        section_old_idx_all = strfind(program_text,section_old);
        section_old_all = extract(program_text,section_old);




        for j = 1:length(section_old_idx_all)
            %         section_current_tmp = program_info.Text(section_idx(j));
            section_old_head = program_text(section_old_idx_all(j)-1);

            if ~strcmp(section_old_head,'%')
                section_old_cell{j} = section_old_all{j};
                section_old_idx(j) = section_old_idx_all(j);
            end

        end






        if length(section_old_cell) > 1
            disp([calc_summary_info.section, '(old calc section) is used more than once'])

        end

        section_old = string(section_old_cell);
        section_current = string(section_current_cell);
        %         section_old_loc = strfind(program_text,section_old);
        %         section_current_loc = strfind(program_text,section_current);

        %     section_current = program_info.Text(section_idx-1)



        section_proceed = section_current_idx - section_old_idx;
    else
        %         idx_program_refresh = 1;
        section_proceed = -1;

    end

    % updates old calc information about section and program name
    calc_summary_info.section = section_current;
    calc_summary_info.ProgramName = programname;

    % create and move new folder if the files are same with previous calculation or program or section is going back

    if ovrlp_file_num ~= 0 || idx_program_refresh == 1 || section_proceed < 0

        idx = length(calc_save_folder) + 1;
        folder_count = sprintf('%05i',idx);
        calc_save_newfolder_name = ['Run_',folder_count];
        cd('..//')
        mkdir(calc_save_newfolder_name)
        cd(calc_save_newfolder_name);

    else
        calc_save_newfolder_name = calc_save_folder(end).name;

    end

    % copy or save the figs and variables to the latest Run_XXXXX

    copy_file_idx = matches(save_files,saved_files_name);
    %         num_copy = sum(copy_file_idx);
    %         num_save = length(save_files) - num_copy;

    for j = 1:length(save_files)

        if copy_file_idx(j) == 1
            copyfile(saved_files_fullpath{j},save_files{j})
        else
            save_files_ext{j} = save_files{j}(end-3:end);
            if strcmp(save_files_ext{j},'.mat') == 1
                save(savevar_name_target{j})
            elseif strcmp(save_files_ext{j},'.fig') == 1
                saveas(figure(savefig_idx_target{j}),save_flies{j})
            end

        end

    end


    %% program backup


    % find and move to the program folder (Notepath\Programs)
    cd('..//..//')
    program_folder = dir('Programs*');

    if isempty(program_folder)
        mkdir('Programs')
    end


    program_file_source = dir(programdir_source);
    program_file_backup = dir(['Programs\',programname]);

    % Copy the program if there are some updates
    if ~isequal(program_file_source,program_file_backup)
        copyfile(programdir_source,['Programs\',programname])
    end


    % Copy the
    cd(Calc_backupFolder)
    cd(calc_save_newfolder_name)

    Backup_program_Folder = dir('Backup_program');
    if isempty(Backup_program_Folder)
        mkdir('Backup_program')
        Backup_program_Folder = dir('Backup_program');
    end


    backup_program_name = strcat(calc_save_newfolder_name,'_',programname(1:end-2),'_',char(Target_calc),'.txt');
    cd(Backup_program_Folder(1).folder)
    copyfile(programdir_source,backup_program_name)






    %%




    cd(oldFolder)

end

% update calc_summaryinfo
%
%
% calc_summary_info_old.ProgramName = programname;
%
% calc_summary_info_old.section = strcat("matlabnote_save(note_info,",'"',Target_calc,'"');
end



%% Auto powerpoint generation
% ver 1.15 
function  matlabnote_ppt(Target_calc,note_info,savefig_name,fig_idx,savevar_name,calc_folder_last)

try

    savefig_name_target = getfield(savefig_name,Target_calc);
    savefig_name_target_str = string(savefig_name_target);
    nofig_idx1 = "off";
catch
    nofig_idx1 = "on";
end

savevar_name_target = getfield(savevar_name,Target_calc);

slide_lists = note_info.Properties.RowNames;

% Match names below with powerpoint template file
TitleBox = "SlideTitle";
DescriptionBox = "Description";
ObjectBox = "Object";
ObjectCapBox = "ObjCaption";
ObjectLinkBox = "ObjLink";
SummaryBox = "Summary";
OriginalDataLinkBox = "OriginalDataLink";
InputsLinkBox = "InputsLink";
ProgramLinkBox = "ProgramLink";
OutputsLinkBox = "OutputsLink";

for i = 1:size(note_info,1)
    target_slide_idx(i,1) = contains(slide_lists{i},Target_calc);
end

target_slide_info = note_info(target_slide_idx,:);
ppt_files = target_slide_info.ppt_file;
ppt_template = target_slide_info.ppt_template;
Notepath = unique(fileparts(ppt_files));

% preparation for figure number in target_slide_info
tmp_obj_tbl = [];
tmp_obj_idx_tbl = [];
for ii = 1:size(target_slide_info,1)
    tmp_obj_tbl = [tmp_obj_tbl;target_slide_info.Objects{ii,1}];
    tmp_obj_idx_tbl = [tmp_obj_idx_tbl;array2table(repmat(ii,size(target_slide_info.Objects{ii,1},1),1))];
    % tmp_obj_tbl = [tmp_obj_tbl,array2table(repmat(ii,size(target_slide_info.Objects{ii,1},1),1))];
end
tmp_obj_tbl = [tmp_obj_tbl,tmp_obj_idx_tbl];
fig_count = 0;
for jj = 1:size(tmp_obj_tbl,1)
    if strcmpi(string(tmp_obj_tbl.Obj_Type(jj)),"figure") == 1
        fig_count = fig_count+1;
        tmp_obj_tbl.Obj_Var(jj) = cellstr(string(num2str(fig_count)));

    end
end

% initial_obj_num = 1;
for kk = 1:size(target_slide_info,1)
    idx = (tmp_obj_tbl.Var1 == kk);
    target_slide_info.Objects{kk,1} = tmp_obj_tbl(idx,1:end-1);
    % tmp_obj_num = size(target_slide_info.Objects{kk,1},1);
    % target_slide_info.Objects{kk,1} = tmp_obj_tbl(initial_obj_num:tmp_obj_num+initial_obj_num-1,:);
    % initial_obj_num = tmp_obj_num+1;
end

oldFolder = cd(Notepath);
% cd('..//')
% Notepath_base = cd;
% cd(oldFolder)

import mlreportgen.ppt.*
slide_format = strings(1,size(ppt_files,1));
for j = 1: size(target_slide_info,1)
    % initialization (ppt, master, layout)
    ppts = Presentation(ppt_files(j),ppt_template(j));
    open(ppts);
    masters{j,1} = getMasterNames(ppts);
    masters_str{j,1} = string(masters{j,1});
    masters_target = target_slide_info.Master;

    for k = 1:size(masters_str{j,1},2)
        masters_idx(j,k) = matches(masters_str{j,1}(k),masters_target);
    end

    layoutnames{j,1} = getLayoutNames(ppts,masters_str{j,1}(masters_idx(j,:)));
    layoutnames_str = split(string(layoutnames{j,1}));
    layoutnames_target = target_slide_info.Layout;

    for k = 1:size(layoutnames_str,2)
        layoutnames_idx(j,k) = matches(layoutnames_str(k),layoutnames_target(j));
    end

    slide_format(j) = layoutnames_str(layoutnames_idx(j,:));


    % add the slide

    contents_slides{j,1} = add(ppts,slide_format(j));




    % add title
    SlideTitleObj{j,1} = find(contents_slides{j,1},TitleBox);
    SlideTitleRaw(j) = target_slide_info.SlideTitle(j);
    SlideTitle(j) = SlideTitleRaw(j);
    clear SlideTitle_tmp
    if contains(SlideTitleRaw(j),'eval(') == 1
        Delimiter_title_ori = extractBetween(SlideTitleRaw(j),'eval(',')');
        Delimiter_title = strcat('eval(',Delimiter_title_ori,')');
        SlideTitles_part = strsplit(SlideTitleRaw(j),Delimiter_title) ;

        for i = 1:length(Delimiter_title_ori)
            clear tmp_var
            tmp_var = evalin('caller',Delimiter_title_ori(i));
            SlideTitle_tmp(i) = strcat(SlideTitles_part(i),string(tmp_var)) ;
        end

        Delimiter_title_ori(end+1) = "";
        SlideTitle_tmp(end+1) = strcat(SlideTitles_part(end),Delimiter_title_ori(end));
        SlideTitle(j) = join(SlideTitle_tmp,"");

    end


    replace(SlideTitleObj{j,1}(1),SlideTitle(j));


    % add description
    DescriptionObj{j,1} = find(contents_slides{j,1},DescriptionBox);
    DescriptionRaw(j) = target_slide_info.Description(j);
    Description(j) = DescriptionRaw(j);
    clear Description_tmp
    if contains(DescriptionRaw(j),'eval(') == 1
        Delimiter_Dcp_ori = extractBetween(DescriptionRaw(j),'eval(',')');
        Delimiter_Dcp = strcat('eval(',Delimiter_Dcp_ori,')');
        Descriptions_part = strsplit(DescriptionRaw(j),Delimiter_Dcp) ;

        for i = 1:length(Delimiter_Dcp_ori)
            clear tmp_var
            tmp_var = evalin('caller',Delimiter_Dcp_ori(i));
            Description_tmp(i) = strcat(Descriptions_part(i),string(tmp_var)) ;


        end
        Delimiter_Dcp_ori(end+1) = "";
        Description_tmp(end+1) = strcat(Descriptions_part(end),Delimiter_Dcp_ori(end));
        Description(j) = join(Description_tmp,"");

    end

    replace(DescriptionObj{j,1}(1),Description(j));



    % add contents 
    if strcmp(nofig_idx1,"on") == 0
    fig_idx_target_all = cell2mat(getfield(fig_idx,Target_calc));
    savefig_name_target_all = savefig_name_target_str;
    end
%     if j == 1
%     fig_idx_target_all = cell2mat(getfield(fig_idx,Target_calc));
%     savefig_name_target_all = savefig_name_target_str;
%     else
%         fig_idx_target_all(1:length(FigNum)) = [];
%         savefig_name_target_all(1:length(FigNum)) = [];
%     end
%     idx_fig = 0; 
%     FigContents_idx = strcmpi(string(target_slide_info.Objects{j,1}.Obj_Type),"Figure");
%    
%     target_slide_info_fig = target_slide_info.Objects{j,1}(FigContents_idx,:);
%     FigNum = size(target_slide_info_fig,1);
%     fig_idx_target = fig_idx_target_all(1:FigNum);
%     savefig_name_target = savefig_name_target_all(1:FigNum);
%     if j == 1
%         ObjNum = size(target_slide_info_fig,1);
%         
%         fig_idx_target = fig_idx_target_all(1:ObjNum);
% 
%     else
% %         ObjNum1 = ObjNum;
%         ObjNum = size(target_slide_info_fig,1);
%         fig_idx_target = fig_idx_target_all(ObjNum:ObjNum);
%     end

%    
%     ContentsNum_all = ([1:size(FigContents_idx,1)])';
%     FigContentsNum = ContentsNum_all(FigContents_idx);
%     FigContentsMat = [FigContentsNum, fig_idx_target'];   

%     if j == 1
%         ObjNum = size(target_slide_info_fig,1);
%         fig_idx_obj = fig_idx_target(1:ObjNum);
%     else
%         ObjNum1 = ObjNum;
%         ObjNum = size(target_slide_info_fig,1);
%         fig_idx_obj = fig_idx_target(ObjNum1+1:ObjNum);
%     end

    for k = 1:size(target_slide_info.Objects{j,1},1)
        clear ContentsObj ContentsCaptionObj ContentsLinkObj
        clear ObjectBoxName ObjectCapBoxName ObjectLinkBoxName
        ObjectBoxName = strcat(ObjectBox,num2str(k));
        ObjectCapBoxName = strcat(ObjectCapBox,num2str(k));
        ObjectLinkBoxName = strcat(ObjectLinkBox,num2str(k));
        ContentsObj = find(contents_slides{j,1},ObjectBoxName);
        ContentsCaptionObj = find(contents_slides{j,1},ObjectCapBoxName);
        ContentsLinkObj = find(contents_slides{j,1},ObjectLinkBoxName);

        if contains(target_slide_info.Objects{j,1}.Obj_Type{k},"Figure",'IgnoreCase',true) == 1
            clear picture
%             fig_idx_target = getfield(fig_idx,Target_calc);
%             idx_fig = idx_fig + 1;
%             fig_num = FigContentsMat(idx_fig,2); 
%             fig_num = fig_idx_target{n_fig};
%             num_fig_target = length(fig_idx_target);
            %             fig_num = double(target_slide_info.Objects{j,1}.Obj_Var{k});
%             fig_num = evalin('caller',target_slide_info.Objects{j,1}.Obj_Var{k});
    
            fig_num_idx = str2double(target_slide_info.Objects{j,1}.Obj_Var{k});
            fig_num = fig_idx_target_all(fig_num_idx);

            figure(fig_num);gcf;
            if ispc == 1
                fig_name = ['AutoGenFig',num2str(fig_num),'.emf'];
            else
                fig_name = ['AutoGenFig',num2str(fig_num),'.tif'];
            end
            exportgraphics(gcf,fig_name,'BackgroundColor','none')
            picture = Picture(fig_name);
            AsRatio_ppt = str2double(extractBefore(ContentsObj.Width,'emu'))/str2double(extractBefore(ContentsObj.Height,'emu'));
            AsRatio_fig = str2double(extractBefore(picture.Width,'px'))/str2double(extractBefore(picture.Height,'px'));
            if AsRatio_fig  <= AsRatio_ppt
                picture.Height = ContentsObj.Height;
                picture_Width = str2double(extractBefore(ContentsObj.Height,'emu'))*AsRatio_fig;
                picture.Width = char(strcat(num2str(picture_Width),"emu"));

            else

              picture.Width = ContentsObj.Width;
              picture_Height = str2double(extractBefore(ContentsObj.Width,'emu'))/AsRatio_fig;
              picture.Height = char(strcat(num2str(picture_Height),"emu"));

            end


            picture_caption_raw = string(target_slide_info.Objects{j,1}.Properties.RowNames{k,1});
            picture_caption = string(picture_caption_raw);



            % conversion of variables in caption
            clear picture_caption_tmp
            if contains(picture_caption_raw,'eval(') == 1
                Delimiter_pic_cap_ori = extractBetween(picture_caption_raw,'eval(',')');
                Delimiter_pic_cap = strcat('eval(',Delimiter_pic_cap_ori,')');
                picture_caption_part = strsplit(picture_caption_raw,Delimiter_pic_cap) ;

                for i = 1:length(Delimiter_pic_cap_ori)
                    clear tmp_var tmp_var_ori
                    %                     tmp_var_ori = cell2mat(Delimiter_pic_cap_ori(i));
                    tmp_var = evalin('caller',Delimiter_pic_cap_ori(i));
                    picture_caption_tmp(i) = strcat(picture_caption_part(i),string(tmp_var)) ;


                end
                Delimiter_pic_cap_ori(end+1) = "";
                picture_caption_tmp(end+1) = strcat(picture_caption_part(end),Delimiter_pic_cap_ori(end));
                picture_caption = join(picture_caption_tmp,"");

            end
            picture.Name = picture_caption;
            replace(ContentsObj,picture);
            replace(ContentsCaptionObj,picture.Name);

            % Link of figure

%             fig_file_count = double(extractBetween(target_slide_info.Objects{j,1}.Obj_Var{k},"{","}"));
%             fig_file_count = fig_num; 
%             fig_file_count = idx_fig; 
            fig_file_count = fig_num;
            tmp_savefig_name = char(savefig_name_target_all(fig_num_idx));
            fig_file_loc = join(['..//Calc\',calc_folder_last,'\',tmp_savefig_name]);
            obj_count = k;
            link_disp_tmp = join(['FigLink',string(obj_count)]);

            link_disp = join(link_disp_tmp);
            link_tmp = ExternalLink(fig_file_loc,link_disp);
            link_paragraph = Paragraph('');
            append(link_paragraph,link_tmp);
            replace(contents_slides{j,1},ObjectLinkBoxName,link_paragraph);
%             replace(ppts.Children(end),ContentsLinkObj.Name,link_paragraph);
%             ContentsLinkObj.X = '1000000emu';


        elseif contains(target_slide_info.Objects{j,1}.Obj_Type{k},"Table",'IgnoreCase',true) == 1
            clear tbl_ppt tmp_var
            tmp_var = evalin('caller',target_slide_info.Objects{j,1}.Obj_Var{k});
            tbl_ppt = Table(tmp_var);
            tbl_ppt.StyleName = string(target_slide_info.Objects{j,1}.TableFormat(k,:).TableStyle);
            HAlignMethod = string(target_slide_info.Objects{j,1}.TableFormat(k,:).HAlign);
            VAlignMethod = string(target_slide_info.Objects{j,1}.TableFormat(k,:).VAlign);
            tbl_Bold_id = string(target_slide_info.Objects{j,1}.TableFormat(k,:).TableBold);

            %             tbl_caption = target_slide_info.Objects{j,1}.Properties.RowNames{k,1};



            tbl_caption_raw = string(target_slide_info.Objects{j,1}.Properties.RowNames{k,1});
            tbl_caption= string(tbl_caption_raw);

            clear tbl_caption_tmp
            if contains(tbl_caption_raw,'eval(') == 1
                Delimiter_tbl_cap_ori = extractBetween(tbl_caption_raw,'eval(',')');
                Delimiter_tbl_cap = strcat('eval(',Delimiter_tbl_cap_ori,')');
                tbl_caption_part = strsplit(tbl_caption_raw,Delimiter_tbl_cap) ;

                for i = 1:length(Delimiter_tbl_cap_ori)
                    clear tmp_var tmp_var_ori
                    %                     tmp_var_ori = cell2mat(Delimiter_tbl_cap_ori(i));
                    tmp_var = evalin('caller',Delimiter_tbl_cap_ori(i));
                    tbl_caption_tmp(i) = strcat(tbl_caption_part(i),string(tmp_var)) ;


                end
                Delimiter_tbl_cap_ori(end+1) = "";
                tbl_caption_tmp(end+1) = strcat(tbl_caption_part(end),Delimiter_tbl_cap_ori(end));
                tbl_caption = join(tbl_caption_tmp,"");

            end






            if matches(tbl_Bold_id,"false") == 1 || matches(tbl_Bold_id,"true") == 1

                tbl_ppt.Style =[ {HAlign(HAlignMethod)} {VAlign(VAlignMethod)} {Bold(eval(tbl_Bold_id))}];
            else
                tbl_ppt.Style =[ {HAlign(HAlignMethod)} {VAlign(VAlignMethod)}];
            end

            replace(ContentsObj,tbl_ppt);
            replace(ContentsCaptionObj,tbl_caption);
        end

        


    end


    % add summary
    SummaryObj{j,1} = find(contents_slides{j,1},SummaryBox);
    SummaryRaw(j) = target_slide_info.Summary(j);
    Summary(j) = SummaryRaw(j);
    clear Summary_tmp
    if contains(SummaryRaw(j),'eval(') == 1
        Delimiter_smry_ori = extractBetween(SummaryRaw(j),'eval(',')');
        Delimiter_smry = strcat('eval(',Delimiter_smry_ori,')');
        Summary_part = strsplit(SummaryRaw(j),Delimiter_smry) ;

        for i = 1:length(Delimiter_smry_ori)
            clear tmp_var
            tmp_var = evalin('caller',Delimiter_smry_ori(i));
            Summary_tmp(i) = strcat(Summary_part(i),string(tmp_var)) ;
        end
        Delimiter_smry_ori(end+1) = "";
        Summary_tmp(end+1) = strcat(Summary_part(end),Delimiter_smry_ori(end));
        Summary(j) = join(Summary_tmp,"");

    end

    replace(SummaryObj{j,1}(1),Summary(j));



    % add link
    % Original data link
    ori_data_link = target_slide_info.OriginalDataDir(j);

    if strlength(ori_data_link) == 0

    else

        OriDataLinkObj = find(contents_slides{j,1},OriginalDataLinkBox);

        cd('..//')
        OriData_folder = dir('OriginalData');
        if isempty(OriData_folder)
            mkdir('OriginalData')
        end
        cd(Notepath)
        [~,ori_data_link_disp,~] = fileparts(ori_data_link);
        link_ori_data = ExternalLink(ori_data_link,ori_data_link_disp);
        ori_data_link = Paragraph('Original Data:    ');
        append(ori_data_link,link_ori_data);
        replace(contents_slides{j,1},OriginalDataLinkBox,ori_data_link);
%         replace(contents_slides{j,1},OriDataLinkObj.Name,ori_data_link);
    end


    % Inputs link
    InputsLinkObj = find(contents_slides{j,1},InputsLinkBox);
    if~isempty(InputsLinkObj) || ~isempty(savevar_name_target{1})
        savevar_name_target_in = savevar_name_target{1};

        inputs_file_loc_in = join(['..//Calc\',calc_folder_last,'\',savevar_name_target_in]);
        link_inputs = ExternalLink(inputs_file_loc_in,savevar_name_target_in);
        i_link = Paragraph('Inputs:    ');
        append(i_link,link_inputs);
        replace(contents_slides{j,1},InputsLinkBox,i_link);
%         replace(contents_slides{j,1},InputsLinkObj.Name,i_link);
        %         add(OutputsLinkObj,'Folder');

        calc_folder_link_ori = ExternalLink(join(['..//Calc\',calc_folder_last]),calc_folder_last);
        calc_folder_link = Paragraph('Folder:   ');
        append(calc_folder_link,calc_folder_link_ori);
        add(InputsLinkObj,calc_folder_link);

    end
    % program link
    ProgramLinkObj = find(contents_slides{j,1},ProgramLinkBox);
    %     programname = calc_summary_info.ProgramName;
    %
    program_info = matlab.desktop.editor.getActive;
    programdir_source = program_info.Filename;
    [~,programname_ori,prgrm_ext] = fileparts(programdir_source);
    programname = [programname_ori,prgrm_ext];
    programdir = join(['..//Programs\',programname]);
    %     programdir = program_info.Filename;
    %     programname = programdir;
    %     slash_idx = (strfind(programname,'\'));
    %     programname(1:slash_idx(end)) = [];
    %     programname(end-1:end) = [];
    link_program = ExternalLink(programdir,programname);
    p_link = Paragraph('Program:   ');
    append(p_link,link_program);
    replace(contents_slides{j,1},ProgramLinkBox,p_link);
%     replace(contents_slides{j,1},ProgramLinkObj.Name,p_link);


    % Result Link
    OutputsLinkObj = find(contents_slides{j,1},OutputsLinkBox);
    if ~isempty(OutputsLinkObj) || ~isempty(savevar_name_target{2})
        
        savevar_name_target_out = savevar_name_target{2};

        Outputs_file_loc = join(['..//Calc\',calc_folder_last,'\',savevar_name_target_out]);
        link_Outputs = ExternalLink(Outputs_file_loc,savevar_name_target_out);
        r_link = Paragraph('Outputs:    ');
        append(r_link,link_Outputs);
        replace(contents_slides{j,1},OutputsLinkBox,r_link);
%         replace(contents_slides{j,1},OutputsLinkObj.Name,r_link);
        %         add(OutputsLinkObj,'Folder');
        clear calc_folder_link calc_folder_link_ori
        calc_folder_link_ori = ExternalLink(join(['..//Calc\',calc_folder_last]),calc_folder_last);
        calc_folder_link = Paragraph('Folder:   ');
        append(calc_folder_link,calc_folder_link_ori);
        add(OutputsLinkObj,calc_folder_link);




    end
%     replace(ppts.Children(end),contents_slides{j,1});
%     ppts.Children(end) = contents_slides{j,1};
    close(ppts);


end

% ppts1 = Presentation(ppt_files(j),ppt_template(j));
% pause(1)

% remove tmp fig file
if ispc == 1
    tmp_fig_file = dir('AutoGenFig*.emf');
else
    tmp_fig_file = dir('AutoGenFig*.tif');
end
if isempty(tmp_fig_file)

else
    delete(tmp_fig_file.name)
end
cd(oldFolder)
% rptview(ppts)
end






%% Figure line width, color, font, font size ect... adjust
function adjfig
% Before run this program, add and decide the xlabel, ylabel, colormap and legend
% this function adjust the figure and copy the fig to your clipboard automatically
%%
current_figure = gcf;
current_figure.Color = [1 1 1];
figure_axis = gca;
box(figure_axis,'on');
set(figure_axis,'FontSize',18,'LineWidth',1.5,'FontName','Arial');
% colormap(jet);

%%
fig_prop = get(figure_axis,'children');

for j = 1:length(fig_prop)

    fig_obj = findobj(fig_prop(j));

    fig_type = fig_obj.Type;


    if strcmp(fig_type,'line') == 1  % if the fig is line plot. eg spectra, signal, response
        set(fig_prop,'LineWidth',2);

    elseif strcmp(fig_type,'scatter') == 1 % if the fig is scatter. eg ML prediction

        for i = 1:length(fig_obj)
            fig_obj(i).SizeData = 100;
        end

    elseif strcmp(fig_type,'image') == 1 % if the fig is image(2D color) eg similarity map,  microscopic image
        if size(fig_obj.CData,1) == size(fig_obj.CData,2) % if the aspect ratio is same
            axis square
            %     else    % aspect ratio is not equal and adjust the ratio by x and y ratio (pixel ratio)
            %         axis image

        end

    elseif strcmp(fig_type,'bar') == 1 % if the fig is bar
        for i = 1:length(fig_obj)
            fig_obj(i).BarLayout = 'stacked';
        end

    elseif strcmp(fig_type,'histogram') == 1 % if the fig is histogram

        fig_obj.LineWidth = 1.0;

    elseif strcmp(fig_type,'stem') == 1     % if the fig is stem(sparse sharp fig) eg database XRD
        for i = 1:length(fig_obj)
            fig_obj(i).LineWidth = 1.5;
        end

    elseif strcmp(fig_type,'surface') == 1  % if the fig is 3D (mesh or surf)

        %   view(figure_axis,[31.5 21.0140350877193])   % input the angle data for visialization

    elseif strcmp(fig_type,'patch') == 1    % if the fig is 3D spectra (waterfall)

        set(fig_prop,'LineWidth',2,'FaceColor',[1 1 1],'EdgeColor','flat')

    end


    % if isprop(fig_prop,'CData')
    % %     set(ax,'square')
    %     axis('square')
    % else
    %     set(fig_prop,'LineWidth',2);
    % end
end
%% Color setting and copy
color_config=[        0         0    1.0000
    1.0000         0         0
    0    0.5000         0
    1.0000         0    1.0000
    0.5000         0    1.0000
    0    0.7500    1.0000
    1.0000    0.5000         0
    0.5000    0.7500         0
    1.0000    0.7500    0.7500
    0.5000    0.5000    1.0000
    0.7500    0.7500         0
    0.5000    0.5000    0.5000
    0.7500         0         0
    0         0         0];

colororder(color_config)

% set(current_figure,'InvertHardcopy','off')
% set(current_figure,'Color','none');
% set(figure_axis,'Color','none');
%
% print -dmeta -painters
%
% set(current_figure,'InvertHardcopy','on')
% set(current_figure,'Color',[1,1,1]);
% set(figure_axis,'Color','[1,1,1]');


end