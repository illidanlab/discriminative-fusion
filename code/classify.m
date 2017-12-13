function info = classify(result_dir, task_name, X, group, ...
    pos_class_arr, neg_class_arr, repeat_exp, method_str, classifier_param,...
    overwrite)
% input: 
%        result_dir: directory to save results
%        task_name: name of the task, e.g. 'classification AD with MCI'
%        X:input data. It is a n*d*m tensor. n is sample size, d is feature
%         dimension, m is modalities number.
%        group: labels of all samples, i.e., 1, 2, 3
%        pos_class_arr: postive class labels array. For example, if      
%                        group == 1 and group == 2 all denote postive class, 
%                        pos_class_arr = [1, 2] 
%        neg_class_arr: negative class lables array.
%        repeat_exp: number of validaton folds. 
%        method_str: 'fusion'
%        classifier_param: parameter to control model complexity to prevent
%                          overfitting
%        overwrite: if True, the current setting will overwrite existing results
% The results are all saved in the results folder
    talk = true;

    %result_dir = './Result'; 
    if ~exist(result_dir, 'dir')
        mkdir(result_dir);
    end

    if nargin<10, overwrite = false; end
    if overwrite
        warning('EXP:Classify','The current setting will overwrite existing results.');
    end

    
    if(talk), fprintf('Classify:task: %s\n', task_name); end
    if(talk), fprintf('Classify:repeat: %u\n', repeat_exp); end
    if(talk), fprintf('Classify:classifier: %s\n', method_str); end
    if(talk), fprintf('Classify:classifier parameter:'); end
    if(talk), disp(classifier_param); end
    % result folder. 
    
    TR_RATIO = 0.85; % the training ratio of the smaller class. 
    
    pos_idx = group == pos_class_arr(1);
    if length(pos_class_arr)> 1
        for ii = 2: length(pos_class_arr)
            pos_idx = pos_idx | group == pos_class_arr(ii);
        end
    end
    neg_idx = group == neg_class_arr(1);
    if length(neg_class_arr)> 1
        for ii = 2: length(neg_class_arr)
            neg_idx = neg_idx | group == neg_class_arr(ii);
        end
    end
    
    y = zeros(size(group));
    y(pos_idx) = 1;
    y(neg_idx) = -1;
    sel_idx = y~=0;
    y = y(sel_idx);
    X = zscore(X(sel_idx, :)); % normalization. 
    
    sample_num_pos = nnz(y == 1);
    sample_num_neg = nnz(y == -1);
    
    tr_num = round(min(sample_num_pos, sample_num_neg) * TR_RATIO);
    
    if(talk), 
        fprintf('Classify:Positive Class: %u [tr: %u, te %u]\n', ...
            sample_num_pos, tr_num, sample_num_pos - tr_num);
        fprintf('Classify:Negative Class: %u [tr: %u, te %u]\n', ...
            sample_num_neg, tr_num, sample_num_neg - tr_num);
    end
    
    all_idx = 1: (sample_num_pos + sample_num_neg);
    pos_idx = all_idx(y == 1);
    neg_idx = all_idx(y == -1);
        
    % TODO: write summary file. 
    % skipped for saving space. 
    
    info = cell(repeat_exp, 1);
    for iter = 1: repeat_exp
        % result file.
        if(talk), fprintf('Iteration: %u\n', iter);end
        iter_file_name = sprintf('%s/%s_%s_iter%u.mat', result_dir, task_name, method_str, iter);
        if exist(iter_file_name, 'file')
            if overwrite
                if(talk), fprintf('Result file [%s] found, overwrite.\n', iter_file_name);end
            else
                if(talk), fprintf('Result file [%s] found, skip.\n', iter_file_name);end
                load_data = load(iter_file_name);
                info{iter} = load_data.info_struct;
                continue;
            end
        end
        
        
        rng(iter); % reset random generator to obtain the 
                   % same splittings for different methods. 
        
        % random permute tr/te index. 
        pos_idx = pos_idx(randperm(length(pos_idx)));
        neg_idx = neg_idx(randperm(length(neg_idx)));
        
        % training/testing.  balance classes. 
        tr_idx = [pos_idx(1:tr_num)     neg_idx(1:tr_num)];
        te_idx = [pos_idx(tr_num+1:end) neg_idx(tr_num+1:end)];
        
        info_struct.tr_idx = tr_idx;
        info_struct.te_idx = te_idx;
        
        % use classifier. 
        model = netfusionProj(X(tr_idx, :), y(tr_idx), classifier_param);
        w = model.w;
        c = model.c;
        tau = model.tau;
        [~, perf_info_tr] = apply_classify_netfusion(w, c,tau,  X(tr_idx, :), y(tr_idx));
        [~, perf_info_te] = apply_classify_netfusion(w, c,tau,  X(te_idx, :), y(te_idx));
        info_struct.perf_info_tr = perf_info_tr;
        info_struct.perf_info_te = perf_info_te;
        te_auc = perf_info_te.auc; %#ok
        te_sen = perf_info_te.sensitivity; %#ok
        te_spe = perf_info_te.specificity; %#ok
        
        info{iter} = info_struct;
        % save iteration information
        save(iter_file_name, 'info_struct', 'te_auc', 'te_sen', 'te_spe');
    end
end