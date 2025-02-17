clear
format short e
format compact

scrpt = 'heat_setup';
mu = 1;
tend = 0.25;

E = [];

nx = 160;
ny = nx;

eval(scrpt)

CFL = [];

CPU_TIME = [];
nt = 40;
    dt_max = tend/nt;
    tstart = tic;
    [U,S,V,RRR_M,TTT_M] = LRIAT_OFT(tend,nx,ny,dt_max,scrpt); 