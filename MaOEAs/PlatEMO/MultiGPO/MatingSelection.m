function MatingPool = MatingSelection(PopObj,N,zmin,zmax)
   % Normalization
   PopObj = (PopObj-repmat(zmin,size(PopObj,1),1))./repmat(zmax-zmin ,size(PopObj,1),1); 
   
   C = sum(PopObj,2); % Calculate Convergence
   Dist= pdist2(PopObj,PopObj,'cosine');
   D  = sort(Dist,2);
   Div = 1./(sum(D(:,2:ceil(end*0.1)),2)+1); % Calculate Diversity
    
   for i = 1 : N
       Index = randperm(N,2);
       if C(Index(1))<C(Index(2)) & Div(Index(1))<Div(Index(2))
           MatingPool(i) = Index(1);
       elseif C(Index(1))>C(Index(2)) & Div(Index(1))>Div(Index(2))
           MatingPool(i) = Index(2);
       else
           if rand()<0.5
               MatingPool(i) = Index(1);
           else
               MatingPool(i) = Index(2);
           end
       end
   end
end

