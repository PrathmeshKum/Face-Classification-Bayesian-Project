function outIm = normalizeIm(inIm)
inIm = double(inIm);
minval = min(inIm(:));
maxval = max(inIm(:));
outIm = (inIm-minval)/(maxval-minval);



