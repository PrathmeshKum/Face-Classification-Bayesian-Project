function dist = mvgd(IMAGE,MEAN,SIGMA)

summation_term=(((transpose(transpose(IMAGE)-transpose(MEAN)))*(transpose(IMAGE)-transpose(MEAN)))/SIGMA);
first_term=((7200/2)*log10((sum(sqrt(sum(SIGMA.^2))))));
dist=-(first_term)-((0.5)*(sum(summation_term(:))));

