function dist = mvgd1(IMAGE,MEAN,SIGMA,IMAGE_NUM,SIZE)

first_term=((SIZE/2)*log((sum(sqrt(sum(SIGMA.^2))))));
SIGMA_INVERSE=(inv(SIGMA));

 for iFile = 1:IMAGE_NUM-1;

     summation_term=(((transpose(transpose(IMAGE(:,iFile))-transpose(MEAN)))*(transpose(IMAGE(:,iFile))-transpose(MEAN))))*SIGMA_INVERSE;
     dist(iFile,:)=-(first_term)-((0.5)*(sum(summation_term(:))));
     disp(iFile);

 end

