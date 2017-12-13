function auc = computeAUC(Actual, Predicted)
% Actual -- n by 1
% Predicted -- n by k
% output -- k by 1;

if size(Actual, 2) == 1 && size(Actual, 1) ~= 1
    % correct.
elseif size(Actual, 1) == 1 && size(Actual, 2) ~= 1
    % transpose input.
    disp('Warning: Transpose vectors deteced in computeAUC input.');
    Actual = Acutal';
    Predicted = Predicted';
end

input_num = size(Predicted, 2); % number of vectors to compare with Actual vector.

if input_num == 1
    [~, ~, ~, auc] = perfcurve(Actual, Predicted, 1);
else
    auc = zeros(input_num , 1);
    for i = 1: input_num
        pred = Predicted(:, i);
        [~, ~, ~, auc(i)]= perfcurve(Actual, pred, 1);
    end
end

end




% NOTE: Alternative way of computing auc.
% nTarget     = sum(double(Actual == 1));
% nBackground = sum(double(Actual == -1));
% % Rank data
% R = tiedrank(Predicted);  % 'tiedrank' from Statistics Toolbox
% % Calculate AUC
% auc = (sum(R(Actual == 1)) - (nTarget^2 + nTarget)/2) / (nTarget * nBackground);
