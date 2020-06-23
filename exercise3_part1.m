clear all
close all
clc

dt = 0.01; t_end = 8; t = 0:dt:t_end;
choose_arch = 2;
beta = 8/3; sigma = 10;

if choose_arch == 1
    rho = [10,28,40];
elseif choose_arch == 2
    rho = [10,28,40,16,22,32,36];
end

n_train = 200;
n_test = 20;
        
n_t = length(t);
n_rho = length(rho);
n_tot = n_rho*n_train;
n_tot_test = n_rho*n_test; 

%%  BUILD TRAINING DATASET

rng(444)
X0 = -15 + 30.*rand(n_tot, 3);
rho_nn = repelem(rho, n_train);

dataset = zeros(n_tot*(n_t-1),4);
label = zeros(n_tot*(n_t-1),4);

for i=1:n_tot
    
    Lorenz = @(t,x)([ sigma * (x(2) - x(1))       ; ...
                      rho_nn(i) * x(1)-x(1) * x(3) - x(2) ; ...
                      x(1) * x(2) - beta*x(3)         ]);              
    ode_options = odeset('RelTol',1e-10, 'AbsTol',1e-11);

    X0_nn = X0(i,:);
    [t1, X_nn] = ode45(Lorenz, t, X0_nn);
    rho_temp = rho_nn(i)*ones(n_t-1,1);
    dataset( 1 + (i-1)*(n_t-1): i*(n_t-1),:) = [X_nn(1:end-1,:) rho_temp];
    label(1 + (i-1)*(n_t-1):i*(n_t-1),:) = [X_nn(2:end,:) rho_temp];

end

train_dataset = dataset;
train_label = label;

%%  BUILD TEST DATASET

rng(444)
X0 = -15 + 30.*rand(n_tot_test, 3);
rho_nn = repelem(rho, n_test);

dataset = zeros(n_tot_test*(n_t-1),4);
label = zeros(n_tot_test*(n_t-1),4);

for i=1:n_tot_test
    
    Lorenz = @(t,x)([ sigma * (x(2) - x(1))       ; ...
                      rho_nn(i) * x(1)-x(1) * x(3) - x(2) ; ...
                      x(1) * x(2) - beta*x(3)         ]);              
    ode_options = odeset('RelTol',1e-10, 'AbsTol',1e-11);

    X0_nn = X0(i,:);
    [t1, X_nn] = ode45(Lorenz, t, X0_nn);
    rho_temp = rho_nn(i)*ones(n_t-1,1);
    dataset( 1 + (i-1)*(n_t-1): i*(n_t-1),:) = [X_nn(1:end-1,:) rho_temp];
    label(1 + (i-1)*(n_t-1):i*(n_t-1),:) = [X_nn(2:end,:) rho_temp];

end

test_dataset = dataset;
test_label = label;

%%  TRAINING

mkdir 'es3_partI'
mkdir 'es3_partI/es3_partI_NN_models'
path = './es3_partI/es3_partI_NN_models/NNLorenz_model.mat';
path_more_rho = './es3_partI/es3_partI_NN_models/NNLorenz_model_more_rho.mat';

net = feedforwardnet([10 20 20 10]);
net.layers{1}.transferFcn = 'logsig';
net.layers{2}.transferFcn = 'radbas';
net.layers{3}.transferFcn = 'tansig';
net.layers{4}.transferFcn = 'purelin';

do_first_train = 0;
if do_first_train == 0
    if choose_arch==1
        load(path);
        net = train(net,dataset.',label.');
        save(path, 'net')
    elseif choose_arch == 2
        load(path_more_rho)
        net = train(net,dataset.',label.');
        save(path_more_rho, 'net')
    end
elseif do_first_train == 1
    if choose_arch==1
        net = train(net,dataset.',label.');
        save(path, 'net')
    elseif choose_arch == 2
        net = train(net,dataset.',label.');
        save(path_more_rho, 'net')
    end
end

%%  PERFORMANCE CHECK

check_test_label = net(test_dataset.');
perf = perform(net,check_test_label,test_label.');

%%  PREDICTION 1
if choose_arch==1
    rng(444)
elseif choose_arch==2
    rng(12345)
end
X0_check = -15 + 30.*rand(n_rho, 3);
X_nn_check = zeros(n_t,3,n_rho);
X_nn_pred = zeros(n_t,3,n_rho);
for j = 1:n_rho
    
    Lorenz = @(t,x)([ sigma * (x(2) - x(1))       ; ...
                      rho(j) * x(1)-x(1) * x(3) - x(2) ; ...
                      x(1) * x(2) - beta*x(3)         ]);              
    ode_options = odeset('RelTol',1e-10, 'AbsTol',1e-11);

    X0_nn = X0_check(j,:);
    [t1, X_nn] = ode45(Lorenz, t, X0_nn);
    X_nn_check(:,:,j) = X_nn;
    
    X_nn_pred(1,:,j) = X0_nn;
    
    for jj=2:n_t
        y0 = net([X0_nn.';rho(j)]);
        X_nn_pred(jj,:,j) = y0(1:3).';
        X0_temp = y0(1:3);
        X0_nn = X0_temp.';
    end
    figure(j+5)
    plot3(X_nn(:,1),X_nn(:,2),X_nn(:,3),'b-'), hold on
    plot3(X0_check(j,1),X0_check(j,2),X0_check(j,3),'ro','Linewidth',[2])
    plot3(X_nn_pred(:,1,j),X_nn_pred(:,2,j),X_nn_pred(:,3,j),'r--')
    grid on
end

figure(3)

for k = 1:n_rho
    subplot(n_rho,3,3*(k-1) +1), plot(t,X_nn_check(:,1,k),t,X_nn_pred(:,1,k),'Linewidth',[2])
    subplot(n_rho,3,3*(k-1) +2), plot(t,X_nn_check(:,2,k),t,X_nn_pred(:,2,k),'Linewidth',[2])
    subplot(n_rho,3,3*(k-1) +3), plot(t,X_nn_check(:,3,k),t,X_nn_pred(:,3,k),'Linewidth',[2])

end

%%  SAVING VARIABLE FOR PLOTS

if choose_arch == 1
    filename = './data/es3_partI_prediction1.mat';
elseif choose_arch==2
    filename = './data/es3_partI_prediction1_more_rho.mat';
end

save(filename, 'X0_check', 'X_nn_check', 'X_nn_pred')


%%  PREDICTION 2

rho_new = [17,35];
n_rho = length(rho_new);
X_nn_check = zeros(n_t,3,n_rho);
X_nn_pred = zeros(n_t,3,n_rho);

for j = 1:n_rho
    
    Lorenz = @(t,x)([ sigma * (x(2) - x(1))       ; ...
                      rho_new(j) * x(1)-x(1) * x(3) - x(2) ; ...
                      x(1) * x(2) - beta*x(3)         ]);              
    ode_options = odeset('RelTol',1e-10, 'AbsTol',1e-11);

    X0_nn = X0_check(j,:);
    [t1, X_nn] = ode45(Lorenz, t, X0_nn);
    X_nn_check(:,:,j) = X_nn;
    
    X_nn_pred(1,:,j) = X0_nn;
    
    for jj=2:n_t
        y0 = net([X0_nn.';rho_new(j)]);
        X_nn_pred(jj,:,j) = y0(1:3).';
        X0_temp = y0(1:3);
        X0_nn = X0_temp.';
    end
    figure(j+10)
    plot3(X_nn(:,1),X_nn(:,2),X_nn(:,3),'b-'), hold on
    plot3(X0_check(j,1),X0_check(j,2),X0_check(j,3),'ro','Linewidth',[2])
    plot3(X_nn_pred(:,1,j),X_nn_pred(:,2,j),X_nn_pred(:,3,j),'r--')
    grid on
end


figure(4)

for k = 1:2
    subplot(2,3,3*(k-1) +1), plot(t,X_nn_check(:,1,k),t,X_nn_pred(:,1,k),'Linewidth',[2])
    subplot(2,3,3*(k-1) +2), plot(t,X_nn_check(:,2,k),t,X_nn_pred(:,2,k),'Linewidth',[2])
    subplot(2,3,3*(k-1) +3), plot(t,X_nn_check(:,3,k),t,X_nn_pred(:,3,k),'Linewidth',[2])

end

%%  SAVING VARIABLE FOR PLOTS

if choose_arch == 1
    filename = './data/es3_partI_prediction2.mat';
elseif choose_arch==2
    filename = './data/es3_partI_prediction2_more_rho.mat';
end

save(filename, 'X0_check', 'X_nn_check', 'X_nn_pred')
