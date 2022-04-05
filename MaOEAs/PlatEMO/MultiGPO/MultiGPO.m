function MultiGPO(Global)
% <algorithm> <A>
% A new many-objective evolutionary algorithm
% gpophi --- 20 --- The enpanding angle of GPO

    %% Parameter setting
    gpophi= Global.ParameterSet(20); % gpophi can be set as {M,2*M,or 3*M}
    
    Population = Global.Initialization();
    zmin       = min(Population.objs,[],1);
    zmax       = max(Population.objs,[],1);
    delta =0.5;
    
    %% Optimization
     while Global.NotTermination(Population)  
         MatingPool=MatingSelection(Population.objs,Global.N,zmin,zmax);
         Offspring  = GA(Population(MatingPool));
         [Population,zmin,zmax] = EnvironmentalSelection([Population,Offspring],zmin,zmax,Global.N,gpophi,delta);         
     end
end

