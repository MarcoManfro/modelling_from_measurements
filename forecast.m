function [u_dmd, Omega] = forecast(Phi, Lambda, b, time)

dt = time(2)-time(1);
lambda = diag(Lambda);
omega = log(lambda)/dt;
Omega = diag(omega);

r = length(Lambda);
u_modes = zeros(r,length(time)); 
for iter = 1:length(time)
    u_modes(:,iter) =(b.*exp(omega*(time(iter))));
end
u_dmd = Phi*u_modes;