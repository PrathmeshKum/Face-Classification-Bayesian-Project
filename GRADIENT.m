clc;
clear all;
close all;
disp(' ## Program for mean & variance of face & background images ## ');

%file directory
    
directory=char(pwd);
TrainingfcDirectory = 'face_train_resized\';
TrainingbgDirectory = 'background_train_resized\';
TestingfcDirectory = 'face_test_resized\';
TestingbgDirectory = 'background_test_resized\';

TrainingfcFiles = dir(TrainingfcDirectory);
TrainingbgFiles = dir(TrainingbgDirectory);
TestingfcFiles = dir(TestingfcDirectory);
TestingbgFiles = dir(TestingbgDirectory);


% Training process

tt= cputime;

% For face image training

[GRAD_MATRIX,IMAGE_NUM]=img_grad(TrainingfcDirectory,TrainingfcFiles);
GRAD_MATRIX=normalizeIm1(GRAD_MATRIX,IMAGE_NUM);
mean_face=zeros(2400,1);

for iFile = 1:IMAGE_NUM-1;
    
    mean_face=mean_face+GRAD_MATRIX(:,iFile);

end

mean_face=mean_face/IMAGE_NUM;
%mean_face=(sum(fc_img_sum))/fc_image_num; % U1

covar_face=zeros(2400,2400);

for iFile = 1:IMAGE_NUM-1;
    
     a=GRAD_MATRIX(:,iFile)- mean_face;
     b=(transpose(a));
     face_covar_numerator=a*b;
     covar_face=covar_face+face_covar_numerator;
end    

covar_face=covar_face/IMAGE_NUM;
%variance_face=(sum(face_var_numerator))/fc_image_num; % E1

%mean_face1 = uint8(round(mean_face*255));
%mean_face1=reshape(mean_face1,[40 30 2]);
%figure;
%subplot(2,2,1);
%showIm=mean_face1;
%imshow(showIm);
%title(' Face Mean ');

covar_face1=diag(covar_face);
covar_face3=sqrt(covar_face1);
covar_face3=diag(covar_face3);
covar_face2=diag(transpose(covar_face1));
covar_face1=sqrt(covar_face);
covar_face1=diag(covar_face1);
%covar_face1 = uint8(round(covar_face1*255));
%covar_face1=reshape(covar_face1,[40 30 2]);
%subplot(2,2,3);
%showIm=covar_face1;
%imshow(showIm);
%title(' Face Covariance ');

disp(['traning face images: ' num2str(cputime-tt)]);




% For background image training

tt= cputime;
bg_image_num=1;

[GRAD_MATRIX,IMAGE_NUM]=img_grad(TrainingbgDirectory,TrainingbgFiles);
GRAD_MATRIX=normalizeIm1(GRAD_MATRIX,IMAGE_NUM);
mean_bg=zeros(2400,1);

for iFile = 1:IMAGE_NUM-1;
    
    mean_bg=mean_bg+GRAD_MATRIX(:,iFile);

end

mean_bg=mean_bg/IMAGE_NUM;
%mean_bg=(sum(bg_img_sum))/bg_image_num; % U0

covar_bg=zeros(2400,2400);

for iFile = 1:IMAGE_NUM-1;
    
   a=GRAD_MATRIX(:,iFile)- mean_bg;
   b=(transpose(a));
   bg_covar_numerator=a*b;
   covar_bg=covar_bg+bg_covar_numerator;  
     
end    

covar_bg=covar_bg/IMAGE_NUM;
%variance_bg=(sum(bg_var_numerator))/bg_image_num; % E0

%mean_bg1 = uint8(round(mean_bg*255));
%mean_bg1=reshape(mean_bg1,[40 30 2]);
%subplot(2,2,2);
%showIm=mean_bg1;
%imshow(showIm);
%title(' Background Mean ');

covar_bg1=diag(covar_bg);
covar_bg3=sqrt(covar_bg1);
covar_bg3=diag(covar_bg3);
covar_bg2=diag(transpose(covar_bg1));
covar_bg1=sqrt(covar_bg);
covar_bg1=diag(covar_bg1);
%covar_bg1 = uint8(round(covar_bg1*255));
%covar_bg1=reshape(covar_bg1,[40 30 2]);
%subplot(2,2,4);
%showIm=covar_bg1;
%imshow(showIm);
%title(' Background Covariance ');

disp(['training background images: ' num2str(cputime-tt)]);


   

% INFERENCE ALGORITHM:


True_fc_num=0;
False_fc_num=0;


[test_face_matrix,fc_test_image_num]=img_grad(TestingfcDirectory,TestingfcFiles);
test_face_matrix=normalizeIm1(test_face_matrix,fc_test_image_num);

disp('computing inference for face: ');


%fc_img_test=mvgd1(test_face_matrix,mean_face,covar_face2,fc_test_image_num,2400);
%fc_img_test_1=mvgd1(test_face_matrix,mean_bg,covar_bg2,fc_test_image_num,2400);

fc_img_test=mvnpdf(transpose(test_face_matrix),transpose(mean_face),covar_face3);
fc_img_test_1=mvnpdf(transpose(test_face_matrix),transpose(mean_bg),covar_bg3);


for iFile = 1:fc_test_image_num-1;
    
    if (fc_img_test(iFile,1)) > (fc_img_test_1(iFile,1));
        
        True_fc_num=True_fc_num+1;
        True_face_images(True_fc_num,1)=iFile; 
        
    else
        
        False_fc_num=False_fc_num+1;
        False_face_images(False_fc_num,1)=iFile;
    
    end
end


True_bg_num=0;
False_bg_num=0;

[test_background_matrix,bg_test_image_num]=img_grad(TestingbgDirectory,TestingbgFiles);
test_background_matrix=normalizeIm1(test_background_matrix,bg_test_image_num);

disp('computing inference for bg: ');

 %bg_img_test=mvgd1(test_background_matrix,mean_bg,covar_bg2,bg_test_image_num,2400);
 %bg_img_test_1=mvgd1(test_background_matrix,mean_face,covar_face2,bg_test_image_num,2400);

bg_img_test=mvnpdf(transpose(test_background_matrix),transpose(mean_bg),covar_bg3);
bg_img_test_1=mvnpdf(transpose(test_background_matrix),transpose(mean_face),covar_face3);
 
for iFile = 1:bg_test_image_num-1;
    
    if (bg_img_test(iFile,1)) > (bg_img_test_1(iFile,1));
        
        True_bg_num=True_bg_num+1;
        True_bg_images(True_bg_num,1)=iFile;
        
    else
        
        False_bg_num=False_bg_num+1;
        False_bg_images(False_bg_num,1)=iFile;
    
    end
end


% Calculation of accuracy:

disp('computing accuracy: ');

face_accuracy=(True_fc_num*100)/(fc_test_image_num-1);
background_accuracy=(True_bg_num*100)/(bg_test_image_num-1);
total_accuracy=((True_fc_num+True_bg_num)*100)/((fc_test_image_num-1)+(bg_test_image_num-1));

path=[directory '\gradient_model_data.mat'];
save(path);