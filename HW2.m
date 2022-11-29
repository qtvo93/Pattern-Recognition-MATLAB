
%% Problem 3.1
close('all');
clear

randn('seed',0);
N = 200;
m = [-5 5 ;0 0];
Si = [];
P = [1/2, 1/2];
for i = 1:2
	Si(:,:,i) = [1 0; 0 1];
end

[X1, y1] = genGaussClasses(m, Si, P, N);
X1 = [X1;ones(1, N)];
%plotData(X1, y1, m, 'Ra');
[X1p, y2] = genGaussClasses(m, Si, P, N);
X1p = [X1p;ones(1, N)];

y1(1, 101:N) = -1;
y2(1, 101:N) = -1;

w1 = [1;1;-0.5];
w2 = [1;-1;-0.5];
w3 = [-1;1;-0.5];

% Classifiers for X1
pw = perce(X1, y1, w1);
verifyVector(X1, y1, pw)
plotLinearClass(X1, y1, m, pw, 'Perce method of w1 on X1');
pw = perce(X1, y1, w2);
verifyVector(X1, y1, pw)
plotLinearClass(X1, y1, m, pw, 'Perce method of w2 on X1');
pw = perce(X1, y1, w3);
verifyVector(X1, y1, pw)
plotLinearClass(X1, y1, m, pw, 'Perce method of w3 on X1');

pw = LMSalg(X1, y1, w1);
verifyVector(X1, y1, pw)
plotLinearClass(X1, y1, m, pw, 'LMS method of w1 on X1');
pw = LMSalg(X1, y1, w2);
verifyVector(X1, y1, pw)
plotLinearClass(X1, y1, m, pw, 'LMS method of w2 on X1');
pw = LMSalg(X1, y1, w3);
verifyVector(X1, y1, pw)
plotLinearClass(X1, y1, m, pw, 'LMS method of w3 on X1');

pw = SSErr(X1, y1);
verifyVector(X1, y1, pw)
plotLinearClass(X1, y1, m, pw, 'SSErr method on X1');

% Classifier for X1p
pw = perce(X1p, y2, w1);
verifyVector(X1p, y2, pw)
plotLinearClass(X1p, y2, m, pw, 'Perce method of w1 on X1prime');
pw = perce(X1p, y2, w2);
verifyVector(X1p, y2, pw)
plotLinearClass(X1p, y2, m, pw, 'Perce method of w2 on X1prime');
pw = perce(X1p, y2, w3);
verifyVector(X1p, y2, pw)
plotLinearClass(X1p, y2, m, pw, 'Perce method of w3 on X1prime');

pw = LMSalg(X1p, y2, w1);
verifyVector(X1p, y2, pw)
plotLinearClass(X1p, y2, m, pw, 'LMS method of w1 on X1prime');
pw = LMSalg(X1p, y2, w2);
verifyVector(X1p, y2, pw)
plotLinearClass(X1p, y2, m, pw, 'LMS method of w2 on X1prime');
pw = LMSalg(X1p, y2, w3);
verifyVector(X1p, y2, pw)
plotLinearClass(X1p, y2, m, pw, 'LMS method of w3 on X1prime');

pw = SSErr(X1p, y2);
verifyVector(X1p, y2, pw)
plotLinearClass(X1p, y2, m, pw, 'SSErr method on X1prime');


% Sourced from text "Pattern Recognition"

function [Dv, classes] = genGaussClasses(m, S, P, N)
    [temp, c] = size(m);
    Dv = [];
    classes = [];
    for i = 1:c
        t = mvnrnd(m(:,i), S(:,:,i), fix(P(i)*N))';
        Dv = [Dv t];
        classes = [classes ones(1, fix(P(i)*N))*i];
    end
end

% Sourced from "Pattern Recognition"

function w=perce(X,y,w_ini)
	[l,N]=size(X);
	max_iter=10000;
	% Maximum allowable number of iterations
	rho=0.05;
	% Learning rate
	w=w_ini;
	% Initialization of the parameter vector
	iter=0;
	% Iteration counter
	mis_clas=N;
	% Number of misclassified vectors
	while (mis_clas>0) && (iter<max_iter)
		iter=iter+1;
		mis_clas=0;
		gradi=zeros(l,1);% Computation of the "gradient"
		% term
		for i=1:N
			if((X(:,i)'*w)*y(i)<0)
				mis_clas=mis_clas+1;
				gradi=gradi+rho*(-y(i)*X(:,i));
			end
		end
		w=w-rho*gradi; % Updating the parameter vector
	end
end


% Rerturns ratio of incorrect classification in a linear classifier

function r=verifyVector(X, y, w)
	n = 0;
	[l, N] = size(y);
	
	for i=1:N
		if w'*X(:,i)*y(i) < 0
			n = n+1;
		end
	end

	r = n/N;
end

% Sourced from text "Pattern Recognition"

function plotLinearClass(X,y,m,w,name)
	plotData(X,y,m,name);
	x = [-25, 25];
    y = (w(3) - w(1)*x)/w(2);
    plot(x,y);
	hold off
end

% Sourced from text "Pattern Recognition"

function plotData(X,y,m,ti)
    [l,N]=size(X); % N=no. of data vectors, l=dimensionality
    [l,c]=size(m); % c=no. of classes
    if(l ~= 2)
        fprintf('NO PLOT CAN BE GENERATED\n')
        return
    else
        pale=['r.'; 'g.'; 'b.'; 'y.'; 'm.'; 'c.'];
    end
    figure()
    title(ti)
    % Plot of the data vectors
    hold on
    for i=1:N
        plot(X(1,i),X(2,i),pale(y(i)+2,:))
    end
    % Plot of the class means
    for j=1:c
        plot(m(1,j),m(2,j),'k+')
    end
end

% Sourced from "Pattern Recognition"

function w=LMSalg(X,y,w_ini)
	[l,N]=size(X);
	rho=0.1;
	% Learning rate initialization
	w=w_ini;
	% Initialization of the parameter vector
	for i=1:N
		w=w+(rho/i)*(y(i)-X(:,i)'*w)*X(:,i);
	end
end

% Sourced from "Pattern Recognition"

function w=SSErr(X,y)
	w=inv(X*X')*(X*y');
end