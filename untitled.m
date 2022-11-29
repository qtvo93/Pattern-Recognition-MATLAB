%%x1 = mvnrnd([0,0],eye(2),100);
%%x2 = mvnrnd([5,0],eye(2),100);
%%clf; hold on
%%plot(x1(:,1),x1(:,2),'.r')
%%plot(x2(:,1),x2(:,2),'.g')

a = 10;
e = 1 ;
sed = 0;
w = [1 1]';
w0 = 0;
N = 1000;
X = generate_hyper(w,w0,a,e,N,sed);
[pc,variances]=pcacov(cov(X'));
d = sqrt(diag(variances));
h = [1 -1]
meanX = mean(X')
hold on;

%%quiver(meanX(1),meanX(2),w(1),w(2),1,'b'); 
%%quiver(meanX(1),meanX(2),h(1),h(2),1,'r'); 
%%quiver(meanX(1),meanX(2),pc(1,1),pc(2,1),d(1),'c'); 
%%quiver(meanX(1),meanX(2),pc(1,2),pc(2,2),d(2),'k'); 

function X=generate_hyper(w,w0,a,e,N,sed)
    l=length(w);
    t=(rand(l-1,N)-.5)*2*a;
    t_last=-(w(1:l-1)/w(l))'*t + 2*e*(rand(1,N)-.5)-(w0/w(l));
    X=[t; t_last];
    %Plots for the 2d and 3d case
    if(l==2)
        figure(1), plot(X(1,:),X(2,:),'.b') 
    elseif(l==3)
        figure(2), plot3(X(1,:),X(2,:),X(3,:),'.r') 
    end
    figure(1), axis equal 
end