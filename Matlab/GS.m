function [u,v] = GS(k,F)
% ______________________________Description______________________________ %
% Solve the Gray Scott model given the parameters k, and F. Other modeling
% parameters, initial conditions, etc. can be adjusted below.
% Inputs:  (k, F)
% Outputs:  (u,v) at the final time
% 
% _______________________________________________________________________ %

rng(32); %fix random seed to make experiment replicable

% ________Modeling parameters and setup for the PDEs ______________ %
Du	= 2e-1;  %Diffusion coefficient in the u equation
Dv	= 1e-1;  %Diffusion coefficient in the v equation
function [U,V] = update_fields(u,v,Du,Dv,dt,dx,dy,k,F)
    U = u + dt*(Du * laplacian(u,dx,dy) - u.*v.^2 + (1-u)*F );
    V = v + dt*(Dv * laplacian(v,dx,dy) + u.*v.^2 - (k+F)*v );
end
function L = laplacian(u,dx,dy)
    [ny,nx] = size(u);
    xL = [nx 1:nx-1];   %sample u from left, using periodic BCs
    yD = [ny 1:ny-1];   %sample u from below, using periodic BCs
    xR = [2:nx 1];      %sample u from right, using periodic BCs
    yU = [2:ny 1];      %sample u from above, using periodic BCs
    L = (u(:,xL)-2*u+u(:,xR))/dx^2 + (u(yD,:)-2*u + u(yU,:))/dy^2;
end
function MakePlot(X,Y,u)
%     pcolor(X,Y,u);
%     shading interp;
    caxis([0 1]);
    colorbar;
    imshow(u);
end
% ________Discretization parameters and domain setup______________ %
dt	= 0.5;    %time step
Nx	= 256;      %horizontal resolution
Ny	= 256;      %vertical resolution
N	= 4e5;      %number of time steps
Lx	= Nx;      %horizontal length
Ly	= Ny;      %vertical length
dx  = Lx/Nx;
dy  = Ly/Ny;
x	= (1:Nx)*dx;  %set x as a vector
y	= (1:Ny)*dy;  %set y as a vector
[X,Y] = meshgrid(x,y);   %set X, Y as tensors
% __________________________Initial conditions___________________________ %
dR	= ( (2*X/Lx-0.5).^2+(2*Y/Ly-1).^2 < 0.04 )+( (2*X/Lx-0.3).^2+(2*Y/Ly-1).^2 < 0.04 );
% dR	= ((X+Y-.5).^2 + (X-Y-.5).^2/40 < 0.01 );
% dR	= ( abs(X-Lx/2) < 0.1 ).*( abs(Y-Ly/2) < 0.1 );
% dR = 1;
% u	= 1+(0   + 0.1*(2*rand(size(X))-1)).*dR;
% v	= 0+(0.4 + 0.1*(2*rand(size(X))-1)).*dR;
u	= 1+(0   + 0.1).*dR;
v	= 0+(0.4 + 0.1).*dR;

% ____________________________Initial plotting___________________________ %
MakePlot(X,Y,u)
title('Start in: 3');
pause(1);
title('Start in: 2');
pause(1);
title('Start in: 1');
pause(1);
% _____________________Time marching___________________ %
jiggle = [400, 800]
%     , 800, 1100];
for n =1:N
    [u,v] = update_fields(u,v,Du,Dv,dt,dx,dy,k,F);
    if ismember(n, jiggle), v = v+0.1*rand(size(X)).*(1-dR); end
	%if n==400,  v = v+0.1*rand(size(X)).*(1-dR); end  %jiggle the system
	%if n==800,  v = v+0.1*rand(size(X)).*(1-dR); end  %jiggle the system
	%if n==1100,  v = v+0.1*rand(size(X)).*(1-dR); end  %jiggle the system
	if mod(n,100)==0
		MakePlot(X,Y,u);
        title(num2str(n*dt));
        drawnow;
%         saveas(gcf,['Frame_dif_' sprintf('%03.0f', n/100) '.png']);
	end
end

end 