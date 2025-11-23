%subjects, in order
subjects={'4';'5';'6';'7';'8';'9';'10';'11';'12';'13';'14';'15';'16';'17';'18';'19';'20'};

%you will need to change the filepath to your own onedrive folders
filepath1 = ['C:\Users\jmartino\OneDrive - University of Edinburgh\Thierry_Data\'] %this is where we put our data

for ns = 1:size(subjects,1)
    
       %load file with behavioral data
        behdata = load ( [filepath1, subjects{ns}, '.result'] );

  %find all oddballs      
  ks = find(behdata(:,3)==2 | behdata(:,3)==3 | behdata(:,3)==4); 
 
  %set a vector of shapes used for selection and iteratively overwritten
  sel=behdata(:,3);
  
  %precede element after diff col non target with 2 and element after diff
  %col target with 4
  for n=1:(length(ks)-1) %length -1 because last element is oddball so there is no +1 for it
  behdata(ks(n)+1,4)=str2double(['9',num2str(behdata(ks(n),3)),num2str(behdata(ks(n)+1,4))]);
    sel(ks(n)+1)=99; %reset
  end   
  
  %now find each 5th standard
  s5=[];value5=[];
  for n=1:size(behdata,1)
      if (sel(n)==1) && (sel(n+1)==1) && (sel(n+2)==1) && (sel(n+3)==1)
          s5=[s5, n+3];
          value5=[value5, str2double([num2str(behdata(n,2)),num2str(behdata(n,2)),num2str(behdata(n,2)),num2str(behdata(n,2)),'1'])];
      end
  end
   
  %now replace them and set selection vector to 99 there
  for s5n=1:length(s5)
      behdata(s5(s5n),4)=value5(s5n);
      sel(s5(s5n))=99;
  end
  
  %now find each 4th standard
  s4=[]; value4=[];
  for n=1:size(behdata,1)
      if (sel(n)==1) && (sel(n+1)==1) && (sel(n+2)==1) 
          s4=[s4, n+2];
          value4=[value4, str2double([num2str(behdata(n,2)),num2str(behdata(n,2)),num2str(behdata(n,2)),'1'])];
      end
  end
  
  %now replace them and set selection vector to 99 there
  for s4n=1:length(s4)
      behdata(s4(s4n),4)=value4(s4n);
      sel(s4(s4n))=99;
  end
  
  %now find each 3rd standard
  s3=[]; value3=[];
  for n=1:size(behdata,1)
      if (sel(n)==1) && (sel(n+1)==1)        
          %assign value based on which condition it is
          s3=[s3, n+1];
          value3=[value3, str2double([num2str(behdata(n,2)),num2str(behdata(n,2)),'1'])];
      end
  end
   
   for s3n=1:length(s3)
      behdata(s3(s3n),4)=value3(s3n);
      sel(s3(s3n))=99;
   end
 
  %to do:
  
  %SAve the edited file with triggers so that they can be imported and
  %different types of standards compared
savename=[filepath1,subjects{ns}, '_edited.result'];
  save(savename,'behdata','-ascii');
  
end

%this bit of code can be used to check the counts of events in the file
 uns=unique(behdata(:,4));
 
 for n=1:length(uns)
     uns(n)
     count=find(behdata(:,4)==uns(n));
     length(count)
 end