clear all;
close all;
clc;

%% setting
global N;
N = 4;

global n;
n = 4;

L = [ 3,  0, -1, -2;
      0,  2, -2,  0;
     -1, -2,  4, -1;
     -2,  0, -1,  3];

F1 = [ 0.4, 0.9;
      -0.9, 0.4];
F2 = [ 0.5, 0.8;
      -0.8, 0.5];

global F;
F = blkdiag(F1, F2);

global Q;
Q = 0.1*eye(n);

H1 = [1 0 0 0];
H2 = [0 1 0 0];
H3 = [0 0 1 0];
H4 = [0 0 0 1];

global H;
H = [H1;H2;H3;H4];

R1 = 0.01;
R2 = 0.01;
R3 = 0.01;
R4 = 0.01;

global R;
R = blkdiag(R1,R2,R3,R4);

global eps_;
eps_ = 1;

k_end = 200;
l_star = 1;


P_CKF = dare(F, H', Q, R);


%% init
est_scale = 10;
x0 = est_scale*randn(n,1);
x = x0;


hat_x1 = est_scale*randn(n, 1);
hat_x2 = est_scale*randn(n, 1);
hat_x3 = est_scale*randn(n, 1);
hat_x4 = est_scale*randn(n, 1);
hat_x = [hat_x1, hat_x2, hat_x3, hat_x4];

P1 = rand*Q;
P2 = rand*Q;
P3 = rand*Q;
P4 = rand*Q;

bar_P = blkdiag(P1,P2,P3,P4);

nu1 = zeros(n*(n+1)/2,1);
nu2 = zeros(n*(n+1)/2,1);
nu3 = zeros(n*(n+1)/2,1);
nu4 = zeros(n*(n+1)/2,1);

global nu;
nu = [nu1;nu2;nu3;nu4];

Theta1 = H(1,:)'*inv(R(1,1))*H(1,:);
Theta2 = H(2,:)'*inv(R(2,2))*H(2,:);
Theta3 = H(3,:)'*inv(R(3,3))*H(3,:);
Theta4 = H(4,:)'*inv(R(4,4))*H(4,:);


bar_Theta = blkdiag(Theta1, Theta2, Theta3, Theta4);
theta = [vech(Theta1); vech(Theta2); vech(Theta3); vech(Theta4)];

global nu;
nu = [nu1;nu2;nu3;nu4];


global alpha_lambda;
alpha_lambda = 0.05;

global alpha_nu;
alpha_nu = 0.05;

e_norm_est = [norm(x-hat_x(:,1)), norm(x-hat_x(:,2)), norm(x-hat_x(:,3)), norm(x-hat_x(:,4))];
e_norm_cov = [];


%% sim
for k=1:1:k_end

    x = F*x + mvnrnd(zeros(n,1), Q, 1)';

    y = zeros(N,1);
    for i=1:1:N
        y(i,1) = H(i,:)*x + mvnrnd(0, R(i,i), 1);
    end

    % local prediction
    hat_x_p = zeros(n,N);       % predicted estimates
    bar_P_p = zeros(n,N);    % predicted covariances
    for i=1:1:N
        hat_x_p(:, i) = predictionEst(hat_x(:, i));
        bar_P_p(idx(i), idx(i)) = predictionP(bar_P(idx(i), idx(i)));
    end

    xi = reshape(hat_x_p,[],1);
    lambda = zeros(N*n,1);

    % distributed correction
    for l=1:1:l_star

        % Dual ascent
        % estimate
        z_xi = kron(L, eye(n))*xi;
        for i=1:1:N
            lambda(idx(i),:) = dualUpdateEstDA(lambda(idx(i),:), z_xi(idx(i),:), bar_P_p(idx(i), idx(i)), i);
        end

        z_lambda = kron(L, eye(n))*lambda;
        for i=1:1:N
            xi(idx(i),:) = primalUpdateEstDA(z_lambda(idx(i),:), hat_x_p(:,i), bar_P_p(idx(i), idx(i)), i, y(i));
        end

        % covariance
        z_theta = kron(L, eye(n*(n+1)/2))*theta;
        for i=1:1:N
            nu(idx_vech(i),:) = dualUpdateCovDA(nu(idx_vech(i),:), z_theta(idx_vech(i),:));
        end

        z_nu = kron(L, eye(n*(n+1)/2))*nu;
        for i=1:1:N
            theta(idx_vech(i),:) = primalUpdateCovDA(z_nu(idx_vech(i)), i);
        end


    end % end distributed correction

    hat_x = reshape(xi,[],4);

    for i=1:1:N
        Theta_i = ivech(theta(idx_vech(i),:));
        bar_P(idx(i), idx(i)) = inv( inv(bar_P_p(idx(i), idx(i))) + Theta_i );
    end

    e_norm_est_k = [norm(x-hat_x(:,1)), norm(x-hat_x(:,2)), norm(x-hat_x(:,3)), norm(x-hat_x(:,4))];
    e_norm_cov_k = [norm(P_CKF - bar_P_p(idx(1), idx(1))), ...
                    norm(P_CKF - bar_P_p(idx(2), idx(2))), ...
                    norm(P_CKF - bar_P_p(idx(3), idx(3))), ...
                    norm(P_CKF - bar_P_p(idx(4), idx(4)))];

    e_norm_est = [e_norm_est; e_norm_est_k];
    e_norm_cov = [e_norm_cov; e_norm_cov_k];
end


%% plotting
figure;
subplot(2,1,1);
hold on;
grid on;
plot(0:1:k_end, e_norm_est(:,1));
plot(0:1:k_end, e_norm_est(:,2));
plot(0:1:k_end, e_norm_est(:,3));
plot(0:1:k_end, e_norm_est(:,4));
legend('Estimator 1', 'Estimator 2', 'Estimator 3', 'Estimator 4');
xlabel('k');
ylabel('norm of estimation error');

subplot(2,1,2);
hold on;
grid on;
plot(1:1:k_end, e_norm_cov(:,1));
plot(1:1:k_end, e_norm_cov(:,2));
plot(1:1:k_end, e_norm_cov(:,3));
plot(1:1:k_end, e_norm_cov(:,4));
legend('Estimator 1', 'Estimator 2', 'Estimator 3', 'Estimator 4');
xlabel('k');
ylabel('norm of covariance error');

%% helper functions
%%
function Y = predictionP(P)
global F;
global Q;
    Y = F*P*F' + Q;
end

function y = predictionEst(x)
global F;
    y = F*x;
end

function xi_c = primalUpdateEstDA(z_lambda, hat_x_p, bar_P_p,i, y)
global N;
global H;
global R;

    D = H(i,:)'*inv(R(i,i))*H(i,:);
    K_cons = inv(D + (1/N)*inv(bar_P_p));
    K_inno = K_cons*H(i,:)'*inv(R(i,i));
    xi_c = hat_x_p + K_inno*(y - H(i,:)*hat_x_p) - K_cons*z_lambda;
end

function lambda_c = dualUpdateEstDA(lambda, z_xi, bar_P_p, i)
global alpha_lambda;
global N;
global n;
global eps_;

    K_dual =  1/norm(N*bar_P_p + 2*N*eps_*eye(n))*eye(n);
    lambda_c = lambda + alpha_lambda*K_dual*z_xi;
end

function theta_c = primalUpdateCovDA(z_nu,i)
global N;
global H;
global R;

    D = H(i,:)'*inv(R(i,i))*H(i,:);
    theta_c = N*vech(D) - z_nu;
end

function nu_c = dualUpdateCovDA(nu, z_eta)
global alpha_nu;
    nu_c = nu + alpha_nu*z_eta;
end

function y = vech(X)
global n;
    ret = zeros(n,1);
    i = 1;
    j = 1;
    for q=1:1:n
        for nn = j:1:n
            ret(i) = X(q, nn);
            i=i+1;
        end
        j=j+1;
    end
    y = ret;
end


function Y= ivech(x)
global n;
    Ret = zeros(4,4);
    i = 1;
    j = 1;

    for q=1:1:n
        for nn = j:1:n
            Ret(q, nn) = x(i);
            Ret(nn, q) = x(i);
            i=i+1;
        end
        j=j+1;
    end

    Y = Ret;
end

function ret = idx(i)
global n;
    ret = (i-1)*n+1:i*n;
end

function ret = idx_vech(i)
global n;
    m = n*(n+1)/2;
    ret = (i-1)*m+1:i*m;
end
