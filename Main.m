% load('C:\Users\ALIENWARE\Desktop\decision-making\工作内容\NewDataset_C_DS_DY.mat')
% CPX= [CPX1  CPX2  CPX3 CPX4 CPX5 CPX6 CPX7  CPXc1 CPXc2 CPXw1 CPXw2];
% CPY= [CPY1  CPY2  CPY3 CPY4 CPY5 CPY6 CPY7  CPYc1 CPYc2 CPYw1 CPYw2];
load('C:\Users\ALIENWARE\Desktop\decision-making\工作内容\NewDataset_D_DS_DY.mat')
CPX= [DPX1  DPX2  DPX3 DPX4 DPX5 DPX6 DPX7  DPXc1 DPXc2 DPXw1 DPXw2];
CPY= [DPY1  DPY2  DPY3 DPY4 DPY5 DPY6 DPY7  DPYc1 DPYc2 DPYw1 DPYw2];
angle_bias=50;
distance_bias=25;
% length=size(CPX1,1);
length=size(DPX1,1);
initial_position=zeros(1,length);
block_moving=zeros(7,length);
hand_moving=zeros(2,length);
is_picking=zeros(length,1);
p1_is_placing=zeros(length,7);
p2_is_placing=zeros(length,7);
Gamma=0.9;
reloading=zeros(length,1);
global TS;
global Obs;
global Reward;
% the following part is to turn the trajectory to observations 
for i= 1:length
    initial_position(1,i)=IFONLINE(i,CPX,CPY);
    block_moving(:,i)=IF_BLOCKMOVING(i,CPX,CPY);
    hand_moving(:,i)=IF_HANDMOVING(i,CPX,CPY);
    if sum(block_moving(:,i))>0 && sum(block_moving(:,i))<3 && sum(hand_moving(:,i))~=0
        is_picking(i)=1;
        vxb=zeros(1,7);
        vyb=zeros(1,7);
        vxw=zeros(1,2);
        vyw=zeros(1,2);
        for j=1:7
            vxb(j)=CPX(i,j)-CPX(i-1,j);
            vyb(j)=CPY(i,j)-CPY(i-1,j);
        end
        for j=1:2
            vxw(1,j)=CPX(i,j+9)-CPX(i-1,j+9);
            vyw(1,j)=CPY(i,j+9)-CPY(i-1,j+9);
        end
        wrist1_distance=sqrt(vxw(1)^2+vyw(1)^2);
        wrist2_distance=sqrt(vxw(2)^2+vyw(2)^2);
        for j=1:7
            angle1=vecangle(vxb(j),vyb(j),vxw(1),vyw(1));
            angle2=vecangle(vxb(j),vyb(j),vxw(2),vyw(2));
            block_distance=sqrt(vxb(j)^2+vyb(j)^2);
            if angle1<angle_bias && block_distance>18 && wrist1_distance>10  
                p1_is_placing(i,j)=1;
            end
             if angle2<angle_bias && block_distance>18 && wrist2_distance>10
                p2_is_placing(i,j)=1;
            end
        end
    
    elseif sum(block_moving(:,i))>=3
        reloading(i)=1;
    end
    
    % POMDP
    
end
%Observation
observation=zeros(length,16);
action=zeros(length,8);
statement=zeros(length,16);
for i=1:length
    observation(i,1:7)=p1_is_placing(i,1:7);
    observation(i,8:14)=p2_is_placing(i,1:7);
    observation(i,15)=initial_position(i);
    observation(i,16)=reloading(i);
end
observation(all(observation==0,2),:)=[];
nlength=size(observation,1);
% However, the model cannot give a right result using the dataset with
% noise. So I made a "perfect"dataset.
% Each row represent time steps. Volume 1-7 means p1(human) has placed blocks
% 1-7. Volume 8-14 means p2(robot)has placed bolcks 1-7. Volume 14 means blocks are at
% the initial places(in a row and ready to move). Volume 15 means blocks
% are being put back.
Observation=Ob();
alpha=zeros(16,8,length);
value=zeros(length,i);
for X=1:1 %Number of iterations
belief=zeros(length,16);
belief(1,15)=1;
%[0.01 0.01 0.01 0.01 0.01 0.01 0.01 0.01 0.01 0.01 0.01 0.01 0.01 0.01 0.85 0.01];%inition
for i=2:101 % if you wanna test all the data just change the number 9 to 'length'
         
    Val=zeros(8,1);
    for a=1:8
        SUM3=0;
     for s=1:16
        
       SUM1=0;
       for z=1:16        
           SUM2=0;
          for s1=1:16
              SUM2=SUM2+Obs(a,z,s1)*TS(a,s1,s)*alpha(s1,a,i-1);
              
          end
          
        SUM1=SUM1+SUM2;
       end
       
       alpha(s,a,i)=Reward(a,s)+Gamma*SUM1;
       
       SUM3=SUM3+belief(i-1,s)*alpha(s,a,i);
     end
     
     Val(a)=SUM3;
    end
    if i==34
        Val(:)
    end
    [m,index]=max(Val(:));
   
    value(i)=m;
    action(i,index)=1;
    Act=index;% Act is the best action 
    % action 1-7 represent p2 placing blocks 1-7 and action 8 means p2
    % waiting. So the ideal result is p2 execute action 8 when is p1's turn 
    % to place blocks and execute action 1-7 after p1 placed some block.
    
    [n,Z]=max(Observation(i,:));% if you wanna test the dataset with noise, change the 
    %Observation(i,:) to observation(i,:)
    if Act ~= 8
        Z=Act+7;
    end
    %belief updating
    
    if Observation(i,15)==1
            belief(i,:)=belief(1,:);
            
    else
       for j=1:16   
        sum1=0;
        sum2=0;
        for k=1:16
          sum1=sum1+TS(Act,j,k)*belief(i-1,k);
          sum2=sum2+Obs(Act,Z,j)*TS(Act,j,k)*belief(i-1,k);
        end
        if sum2~=0
                    
            belief(i,j)=Obs(Act,Z,j)*sum1/sum2;
        else
            
            belief(i,j)=0;
        end
        
       end
 
    end
end
end
% plot observation and action
figure();
plot_point_ob(Observation,100);
plot_point_action(action,100);
hold off;
p2_T=0;
% for i=1:nlength
%     if sum(observation(i,8:14))~=0 && sum(action(i,1:7))~=0
%         p2_T=p2_T+1;
%     elseif ((sum(observation(i,1:7))~=0)||(sum(observation(i,15:16))~=0))&& action(i,8)==1
%         p2_T=p2_T+1;
%     end
% end
for i=1:100
    if Observation(i,8:14)==action(i,1:7)
        p2_T=p2_T+1;
    elseif ((sum(Observation(i,1:7))~=0)||(sum(Observation(i,15:16))~=0))&& action(i,8)==1
        p2_T=p2_T+1;
    end
end
p2_T
T_rate=p2_T/100
function  ONLINE=IFONLINE(x,CPX,CPY)
 sumX=0;
 sumY=0;
 ONLINE=0;
 onlineX=0;
 onlineY=0;
for i=1:7
    sumX=sumX+CPX(x,i); 
    sumY=sumY+CPY(x,i);
    
end
 meanX=sumX/7;
 meanY=sumY/7;
 ifonlineX=zeros(1,7);
 ifonlineY=zeros(1,7);
 for i=1:7
     if CPX(x,i)>(0.97*meanX) && CPX(x,i)<(1.03*meanX)
         ifonlineX(:,i)=1;
     end
     
     
     if CPY(x,i)>(0.95*meanY) && CPY(x,i)<(1.05*meanY)
         ifonlineY(:,i)=1;
     end
 end
if sum(ifonlineX)==7
onlineX=1;
end
if sum(ifonlineY)==7
onlineY=1;
end
if onlineX==1||onlineY==1
    ONLINE=1;
end
end
     
function MOVING=IF_BLOCKMOVING(x,CPX,CPY)
ifmoving=zeros(7,1);
if x==1
    MOVING=ifmoving;
else
    for i=1:7
        eudistance=sqrt((CPX(x,i)-CPX(x-1,i))^2+(CPY(x,i)-CPY(x-1,i))^2);
        if eudistance>5
        ifmoving(i,1)=1;
        end
    end
    MOVING=ifmoving;
end

end 

function MOVING=IF_HANDMOVING(x,CPX,CPY)
ifmoving=zeros(2,1);
if x==1
    MOVING=ifmoving;
else
    for i=1:2
        j=i+9;
        eudistance=sqrt((CPX(x,j)-CPX(x-1,j))^2+(CPY(x,j)-CPY(x-1,j))^2);
        if eudistance>6
        ifmoving(i,1)=1;
        end
    end
    MOVING=ifmoving;
end

end 

function angle=vecangle(v1x,v1y,v2x,v2y)
a=v1x*v2x+v1y*v2y;
b=sqrt(v1x^2+v1y^2)*sqrt(v2x^2+v2y^2);
angle=acos(a/b)*180/pi;
end

function Observation=Ob()
Observation=[0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0;
             0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0;
             0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0;
             0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0;
             0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0;
             0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0;
             0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0;
             0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0;
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1;
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0;
             0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0;
             0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0;
             0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0;
             0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0;
             0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0;
             0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0;
             0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0;
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1;
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0;
             0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0;
             0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0;
             0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0;
             0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0;
             0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0;
             0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0;
             0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0;
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1;
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0;
             0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0;
             0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0;
             0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0;
             0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0;
             0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0;
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1;
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0;
             0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0;
             0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0;
             0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0;
             0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0;
             0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0;
             0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0;
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1;
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0;
             0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0;
             0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0;
             0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0;
             0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0;
             0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0;
             0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0;
             0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0;
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1;
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0;
             0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0;
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1;
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0;
             0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0;
             0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0;
             0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0;
             0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0;
             0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0;
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1;
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0;
             0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0;
             0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0;
             0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0;
             0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0;
             0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0;
             0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0;
             0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0;
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1;
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0;
             0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0;
             0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0;
             0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0;
             0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0;
             0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0;
             0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0;
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1;
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1;
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0;
             0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0;
             0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0;
             0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0;
             0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0;
             0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0;
             0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0;
             0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0;
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1;
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0;
             0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0;
             0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0;
             0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0;
             0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0;
             0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0;
             0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0;
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1;
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0;
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0;
             0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0;
             0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0;
             0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0;
             0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0;
             0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0;
             0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0;
             0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0;
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1;
 ];
end

function plot_point_ob(Matrix, L)
  size=10; 
  ylim([0,20]);
  set(gca,'YTick',[0:1:20]);
  J1=0;
  J2=0;
   for i=1:L

     for j=1:16
       if Matrix(i,j)~=0
         J2=j;
         if j<=7
           plot(i,j,'.','Color','r','MarkerSize',size);
           hold on;
         elseif j>7 && j<15
           plot(i,j,'.','Color','b','MarkerSize',size);
           hold on;
         elseif j==15
           plot(i,j,'.','Color','g','MarkerSize',size);
           hold on;
         elseif j==16
           plot(i,j,'.','Color','k','MarkerSize',size);
           hold on;
         end         
       end
     end
     
     plot([i-1,i],[J1,J2],'linestyle','-','color','m');
     hold on;
     J1=J2;
        
   end
   xlabel('time step');
   ylabel('observation');
end

function plot_point_action(Matrix, L)
  size=5; 
  ylim([0,20]);
  set(gca,'YTick',[0:1:20]); 
  J1=0;
  J2=0;
   for i=1:L

     for j=8:15
       if Matrix(i,j-7)~=0
         J2=j;
         plot(i,j,'*','Color','b','MarkerSize',size);
         hold on;                
       end
     end
     
     plot([i-1,i],[J1,J2],'linestyle','--','color','k');
     hold on;
     J1=J2;
        
   end
   xlabel('time step');
   ylabel('observation');
end