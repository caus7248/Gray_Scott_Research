% This is a cool stamp, other than our circular-square shape

% _________________inline functions for the construction_________________ %
rr   = @(theta, a, k)  (1+a*cos(k*theta))/(1+a);
UU   = @(R,b,c) exp(-36*(R/b).^c);

% ___Some test parameters. See more configurations commented out below___ %
a   = 0.2;  % a = 0 for a circle/square,  0 < a < 1 for petals/gears
k   = 5;    % the number of petals/gear teeth. when a = 0, set k = 1
b   = 1;    % the radius of the stamp, as a percentage of the domain
c   = 100;  % the sharpness parameter
d   = 2;    % use only when a = 0. an even integer; d = 2 is circular, d = 40 for squares

% _________________Set up Cartesian and polar coordinates________________ %
x   = -1:0.01:1;
[X,Y] = meshgrid(x,x);
R   = sqrt(X.^2+Y.^2);
T   = acos(X./R).*sign(Y+eps);  %this "+eps" fixes the angle when x<0, y=0

% _________________Demo, varying the reference angle phi_________________ %
for n=1:100
    phi = (n/100)*2*pi;   % spin the shape one full revolution
    if a == 0
        Rd   = R.*(cos(T-phi).^d+sin(T-phi).^d).^(1/d);     % phi rotates!
    else
        r = rr(T-phi,a,k);      % sets the shape of the stamp
        Rd   = R./r;
    end
    U = UU(Rd,b,c);  %and that's it. Now, let's have a look


    subplot(1,2,1);
    surf(X,Y,U);
    shading interp;
    axis square;
    view(-10,45);

    subplot(1,2,2);
    pcolor(X,Y,U);
    shading interp;
    axis square;

    drawnow;

end

% _________________________default configurations________________________ %
% a   = 0.2;  k   = 5;  b   = 1;      c   = 100;  d   = 2;  % thick star
% a   = 0.5;  k   = 5;  b   = 1;      c   = 100;  d   = 2;  % thin star
% a   = 0.2;  k   = 20; b   = 1;      c   = 100;  d   = 2;  % thick gear
% a   = 0.5;  k   = 20; b   = 1;      c   = 100;  d   = 2;  % thin gear
% a   = 0;    k   = 1;  b   = 0.2;    c   = 100;  d   = 2;  % small circle
% a   = 0;    k   = 1;  b   = 0.7;    c   = 100;  d   = 2;  % large circle
% a   = 0;    k   = 1;  b   = 0.2;    c   = 100;  d   = 40; % small square
% a   = 0;    k   = 1;  b   = 0.7;    c   = 100;  d   = 40; % large square
% a   = 0;    k   = 1;  b   = 0.7;    c   = 10;  d   = 40;  % square pyramid
