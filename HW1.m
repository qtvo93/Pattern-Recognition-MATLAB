%% Problem 2.7
close('all');
clear

rng(0);
N = 1000;
m = [1 4 8; 1 4 1];
Si = [];
for i = 1:3
    Si(:,:,i) =  6*eye(2);
end
P1 = [1/3, 1/3, 1/3];
P2 = [0.8, 0.1, 0.1];

%(A)
% X5 data generate and plot
[X5, y1] = generate_gauss_classes(m, Si, P1, N);
plot_data(X5, y1, m, 'Data plot of X5');

% X5_prime data generate and plot
[X5_prime, y2] = generate_gauss_classes(m, Si, P2, N);
plot_data(X5_prime, y2, m, 'Data plot of X5 prime');

%(B)
% X5 bayes classifier and euclidean classifier
bayes_X5 = bayes_classifier(m, Si, P1, X5);
plot_data(X5, bayes_X5, m, 'X5 Bayesian classifier');
euclid_X5 = euclidean_classifier(m, X5);
plot_data(X5, euclid_X5, m, 'X5 Euclid classifier');

% X5_prime bayes classifier and euclidean classifier
bayes_X5_prime = bayes_classifier(m, Si, P2, X5_prime);
plot_data(X5_prime, bayes_X5_prime, m,'X5 prime Bayesian classifier');
euclid_X5_prime = euclidean_classifier(m, X5_prime);
plot_data(X5_prime, euclid_X5_prime, m, 'X5 prime Euclid classifier');

% X5 error computing
X5_bayes_error = compute_error(bayes_X5, y1); 
X5_euclid_error = compute_error(euclid_X5, y1);

% X5_prime error computing
X5_prime_bayes_error = compute_error(bayes_X5_prime, y2); 
X5_prime_euclid_error = compute_error(euclid_X5_prime, y2); 

%% Problem 2.8
close('all');
clear

N = 1000;
m = [1 8 13; 1 6 1];
Si = [];
for i = 1:3
    Si(:,:,i) = 6*eye(2);
end
P = [1/3, 1/3, 1/3];
[X3, y3] = generate_gauss_classes(m, Si, P, N);
[Z, yz] = generate_gauss_classes(m, Si, P, N);

k1_X3 = k_nn_classifier(Z, yz, 1, X3);
plot_data(X3, k1_X3, m, 'X3 with KNN classifier k=1');
k11_X3 = k_nn_classifier(Z, yz, 11, X3);
plot_data(X3, k11_X3, m, 'X3 with KNN classifier k=11');


% These imported functions are from the text book "Pattern Recognition"
% Gauss generate
function [X, y]=generate_gauss_classes(m,S,P,N)
    [l, c]=size(m);
    X=[];
    y=[];
    for j=1:c
    % Generating the [p(j)*N)] vectors from each distribution
        t=mvnrnd(m(:,j),S(:,:,j),fix(P(j)*N))';
        % The total number of points may be slightly less than N
        % due to the fix operator
        X=[X t];
        y=[y ones(1,fix(P(j)*N))*j];
    end
end

% plot_data
function plot_data(X,y,m, til)
    [l,N]=size(X); % N=no. of data vectors, l=dimensionality
    [l,c]=size(m); % c=no. of classes
    if(l~=2)
        fprintf('NO PLOT CAN BE GENERATED\n')
        return
    else
        pale=['r.'; 'g.'; 'b.'; 'y.'; 'm.'; 'c.'];
        figure()
        title(til)
        % Plot of the data vectors
        hold on
        for i=1:N
            plot(X(1,i),X(2,i),pale(y(i),:))
        end
        % Plot of the class means
        for j=1:c
            plot(m(1,j),m(2,j),'k+')
        end
    end
end

% compute gauss
function val = comp_gauss_dens_val(m, S, x)
    [l, q] = size(m);
    val = (1/((2*pi)^(l/2)*det(S)^0.5))*exp(-0.5*(x-m)'*inv(S)*(x-m));
end

% bayes classifier
function z=bayes_classifier(m,S,P,X)
    [l,c]=size(m); % l=dimensionality, c=no. of classes
    [l,N]=size(X); % N=no. of vectors
    for i=1:N
        for j=1:c
            t(j)=P(j)*comp_gauss_dens_val(m(:,j),S(:,:,j),X(:,i));
        end
        % Determining the maximum quantity Pi*p(x|wi)
        [l,z(i)]=max(t);
    end
end

% euclidean classifier
function z=euclidean_classifier(m,X)
    [l,c]=size(m); % l=dimensionality, c=no. of classes
    [l,N]=size(X); % N=no. of vectors
    for i=1:N
        for j=1:c
            t(j)=sqrt((X(:,i)-m(:,j))'*(X(:,i)-m(:,j)));
        end
        % Determining the maximum quantity Pi*p(x|wi)
        [l,z(i)]=min(t);
    end
end

% compute error
function clas_error=compute_error(y,y_est)
    [q,N]=size(y); % N= no. of vectors
    c=max(y); % Determining the number of classes
    clas_error=0; % Counting the misclassified vectors
    for i=1:N
        if(y(i)~=y_est(i))
            clas_error=clas_error+1;
        end
    end
    % Computing the classification error
    clas_error=clas_error/N;
end

% knn 
function z=k_nn_classifier(Z,v,k,X)
    [l,N1]=size(Z);
    [l,N]=size(X);
    c=max(v); % The number of classes
    % Computation of the (squared) Euclidean distance
    % of a point from each reference vector
    for i=1:N
        dist=sum((X(:,i)*ones(1,N1)-Z).^ 2);
        %Sorting the above distances in ascending order
        [sorted,nearest]=sort(dist);
        % Counting the class occurrences among the k-closest
        % reference vectors Z(:,i)
        refe=zeros(1,c); %Counting the reference vectors per class
        for q=1:k
            class=v(nearest(q));
            refe(class)=refe(class)+1;
        end
        [l,z(i)]=max(refe);
    end
end