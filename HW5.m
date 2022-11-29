
(randn('seed', 0))
m=[-5 5 5 -5; 5 -5 5 -5]; % mean vectors m1 m2
S = 2;
N =1000;
[X1,y1] = data_generator(m,S,N);

(randn('seed', 10))
[X2,y2] = data_generator(m,S,N);

C = 100; % 100, 1000
sigma =0.5; % 1, 2, 4
tol = 0.001;
SVMModel = fitcsvm(X1',y1','DeltaGradientTolerance',tol,'KernelFunction','RBF','KernelScale',...
    sigma,'BoxConstraint',C,'Solver','SMO','CacheSize',10000,'IterationLimit',20000,'Verbose',1);


CVSVMModel = crossval(SVMModel);
classLoss = kfoldLoss(CVSVMModel)


tc = fitctree(X1',y1','MaxNumSplits',7,'CrossVal','on')
error = kfoldLoss(tc)
view(tc.Trained{1},'Mode','graph')

tree = fitctree(X1',y1');
[~,~,~,bestlevel] = cvLoss(tree,'SubTrees','All')

view(tree,'Mode','Graph','Prune',1)
pruned = prune(tree,'Level',1); 
error = cvLoss(pruned);


function [x,y]=data_generator(m,s,N)
    S = s*eye(2);
    [l,c] = size(m);
    x = []; % Creating the training set
    for i = 1:c
        x = [x mvnrnd(m(:,i)',S,N)'];
    end
    y=[ones(1,N) ones(1,N) -ones(1,N) -ones(1,N)];
end
