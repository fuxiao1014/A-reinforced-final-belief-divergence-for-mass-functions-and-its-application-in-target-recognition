clc,clear,close all
data=xlsread('data_huitu_1_excel.xlsx');  %Your file directory
[n, m] = size(data); 
r=cell(1,m); 
for i=1:m
    r{i}=data(:,i);
end
for i=1:m
    eval(['r',num2str(i),'=r{i};']);
end

xmax=max(r1);
xmin=min(r1);

ymax=max(r2);
ymin=min(r2);

[X,Y]=meshgrid(xmin:0.01:xmax,ymin:1:ymax); %Set the x axis and y axis

Z=griddata(r1,r2,r3,X,Y); %Picture

%Fig5a
%{
%subplot(2,2,1);
mesh(X,Y,Z,Z);
zlim([0,1.5]);
%surf(X,Y,Z,Z)                  

xlabel('¦Ã');
ylabel('t');
zlabel('divergence');
%shading interp;
%title('(a)Result of proposed divergence');
colorbar
%}

%Fig5b
%{
%subplot(2,2,2);
mesh(X,Y,Z,Z);
zlim([0,1.5]);
%surf(X,Y,Z,Z)  
view(-90,90);
xlabel('¦Ã');
ylabel('t');
zlabel('divergence');
%shading interp;
%title('(b)Variable ¦Ã and t');
colorbar
%}

%Fig5c
%{
%subplot(2,2,3);
mesh(X,Y,Z,Z);
zlim([0,1.5]);
%surf(X,Y,Z,Z)   
view(180,0);
xlabel('¦Ã');
ylabel('t');
zlabel('divergence');
%shading interp;
%title('(c)Divergence measure with variable ¦Ã');
colorbar
%}

%Fig5d
%{
%subplot(2,2,4);
mesh(X,Y,Z,Z);
zlim([0,1.5]);
%surf(X,Y,Z,Z)              
view(-90,0);
xlabel('¦Ã');
ylabel('t');
zlabel('divergence');
%shading interp;
%title('(d)Divergence measure with variable t');
colorbar
%}

set(gcf,'position',[300 50 900 700]);
