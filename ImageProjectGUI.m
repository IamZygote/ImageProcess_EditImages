
function varargout = ImageProjectGUI(varargin)
% IMAGEPROJECTGUI MATLAB code for ImageProjectGUI.fig
%      IMAGEPROJECTGUI, by itself, creates a new IMAGEPROJECTGUI or raises the existing
%      singleton*.
%
%      H = IMAGEPROJECTGUI returns the handle to a new IMAGEPROJECTGUI or the handle to
%      the existing singleton*.
%
%      IMAGEPROJECTGUI('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in IMAGEPROJECTGUI.M with the given input arguments.
%
%      IMAGEPROJECTGUI('Property','Value',...) creates a new IMAGEPROJECTGUI or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before ImageProjectGUI_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to ImageProjectGUI_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help ImageProjectGUI

% Last Modified by GUIDE v2.5 01-Jan-2022 15:06:32

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @ImageProjectGUI_OpeningFcn, ...
                   'gui_OutputFcn',  @ImageProjectGUI_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT


% --- Executes just before ImageProjectGUI is made visible.
function ImageProjectGUI_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to ImageProjectGUI (see VARARGIN)

% Choose default command line output for ImageProjectGUI
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes ImageProjectGUI wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = ImageProjectGUI_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;
global im;
im=imread('coins.png');
axes(handles.axes1);
imshow(im);



function median(hObject, eventdata, handles,MeanFilter_Row,MeanFilter_Column,A)

axes(handles.axes1);
imshow(A);
title('Original Pic');
 
%pad the input img with zeros
%create a padded empty img from the original img
paddedImg=zeros(size(A)+2);
 
%create an empty image with the size of original img(output img)
output=zeros(size(A));
 
%copy the original image to the padded image
        for x=1:size(A,1)
            for y=1:size(A,2)
                paddedImg(x+1,y+1)=A(x,y);
            end
        end
      %LET THE WINDOW BE AN ARRAY
      %STORE 3x3 VALUES IN THE ARRAY
      %SORT AND FIND THE MIDDLE ELEMENT (5TH ELEMENT)
 
for i= 1:size(paddedImg,1)-2
    for j=1:size(paddedImg,2)-2
        window=zeros(9,1);
        inc=1;
        for x=1:3
            for y=1:3
                window(inc)=paddedImg(i+x-1,j+y-1);
                inc=inc+1;
            end
        end
 
        med=sort(window);
        %PLACE THE MEDIAN ELEMENT IN THE OUTPUT MATRIX
        output(i,j)=med(MeanFilter_Row);
 
    end
end
%CONVERT THE OUTPUT MATRIX TO 0-255 RANGE IMAGE TYPE
output=uint8(output);
axes(handles.axes2), imshow(output);
title('Median');


% --- Executes on button press in pushbutton1.
function pushbutton1_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
im=imread('coins.png');
NA=addnoise(hObject, eventdata, handles,im,10);
median(hObject, eventdata, handles,3,3,NA);


function GaussianFilter(hObject, eventdata, handles,sz)

A=addnoise(hObject, eventdata, handles,Img,10);
axes(handles.axes1);
imshow(A);
title('Original Pic');
%Image with noise
%figure,imshow(A);
I = double(A);
%Design the Gaussian Kernel
%Standard Deviation

%Window size
%sz = 3;
A = imread('coins.png');
sigma = 1.76;
[x,y]=meshgrid(-sz:sz,-sz:sz);

M = size(x,1)-1;
N = size(y,1)-1;
Exp_comp = -(x.^2+y.^2)/(2*sigma*sigma);
Kernel= exp(Exp_comp)/(2*pi*sigma*sigma);
%Initialize
Output=zeros(size(I));
%Pad the vector with zeros
I = padarray(I,[sz sz]);

%Convolution
for i = 1:size(I,1)-M
    for j =1:size(I,2)-N
        Temp = I(i:i+M,j:j+M).*Kernel;
        Output(i,j)=sum(Temp(:));
    end
end
%Image without Noise after Gaussian blur
Output = uint8(Output);
axes(handles.axes2);
imshow(Output);
title('Gaussian Filter');

% --- Executes on button press in pushbutton2.
function pushbutton2_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton9 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
%Read an Image
GaussianFilter(hObject, eventdata, handles,3);



function Histogram(hObject, eventdata, handles,Slider)
GIm=imread('tire.tif');
axes(handles.axes1);
imshow(GIm);

numofpixels=size(GIm,1)*size(GIm,2);


%figure,imshow(GIm);

title('Original Image');

HIm=uint8(zeros(size(GIm,1),size(GIm,2)));

freq=zeros(256,1);

probf=zeros(256,1);

probc=zeros(256,1);

cum=zeros(256,1);

output=zeros(256,1);


%freq counts the occurrence of each pixel value.

%The probability of each occurrence is calculated by probf.


for i=1:size(GIm,1)

    for j=1:size(GIm,2)

        value=GIm(i,j);

        freq(value+1)=freq(value+1)+1;

        probf(value+1)=freq(value+1)/numofpixels;

    end

end


sum=0;

no_bins=255;


%The cumulative distribution probability is calculated. 

for i=1:size(probf)

   sum=sum+freq(i);

   cum(i)=sum;

   probc(i)=cum(i)/numofpixels;

   output(i)=round(probc(i)*no_bins);

end

for i=1:size(GIm,1)

    for j=1:size(GIm,2)

            HIm(i,j)=output(GIm(i,j)+Slider);

    end

end

%figure,imshow(HIm);
axes(handles.axes2);
imshow(HIm);
figure('Position',get(0,'screensize'));
title('Histogram equalization');
subplot(2,2,2); bar(GIm);
title('Before Histogram equalization');
subplot(2,2,4); bar(HIm);
title('After Histogram equalization');

function Interpolation(hObject, eventdata, handles,scale_factor)
A=imread('cameraman.tif');
axes(handles.axes1);
imshow(A);axis([0,512,0,512]);axis on;
% Initalise matrix dimensions
[col, row] = size(A);

% Initalising dimensions of scaled matrix
row_scale = row*scale_factor;
col_scale = col*scale_factor;

% Initalise matrix to hold scaled image
scaled_image = uint8(zeros([col_scale row_scale]));

% Assigns scaled matrix values from orignal image
for r = 1:row_scale
   for c = 1:col_scale
       
       a = ceil(r/scale_factor); %Dividing row value by scale factor
       b = ceil(c/scale_factor); %Dividing column value by scale factor
       
       scaled_image(c,r) = A(b,a);
       
   end
end

axes(handles.axes2); figure,imshow(scaled_image);title('AFTER INTERPOLATION');  axis([0,512,0,512]);axis on;  axes(handles.axes2); imshow(scaled_image);title('AFTER INTERPOLATION');  axis([0,512,0,512]);axis on;

function NA=addnoise(hObject, eventdata, handles,A,sz)
% add uniform noise to an image with sz% of the image
% convert to two dimensions
axes(handles.axes1);
imshow(A);
title('Original Pic');
[h w d]=size(A);
s=h*w*d;
nsz=sz*s/100;

%choose pixels random pixels
r=1:s;
%linear random unique integers
for i=1:s
    rn=round((s-1)*rand(1,1))+1;
    tmp=r(i);
    r(i)=r(rn);
    r(rn)=tmp;
end

%fill with random unifrom noise
%map 1D to 2D row wise 
for i=1:nsz
    xy=r(i);
    y=mod(xy,w);
    if y==0
       y=w; 
       x=floor(xy/w);
    else
       x=floor(xy/w)+1;
    end
   A(x,y)=round(255*rand(1,1));
end
NA=A;
axes(handles.axes2);
imshow(A);
title('Uniform Noise');
clear A r

% --- Executes on button press in pushbutton3.
function pushbutton3_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
Histogram(hObject, eventdata, handles,1);


% --- Executes on button press in pushbutton4.
function pushbutton4_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
im=imread('coins.png');
addnoise(hObject, eventdata, handles,im,10);


% --- Executes on button press in pushbutton5.
function pushbutton5_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton5 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
Interpolation(hObject, eventdata, handles,2);


function laplacian(hObject, eventdata, handles,Slider)
k=imread("coins.png");
axes(handles.axes1);
imshow(k);
title('Original Pic');
% Convert into double format.
k1=double(k);

% Define the Laplacian filter.
Laplacian=[0 1 0; 1 -4 1; 0 1 0];

% Convolve the image using Laplacian Filter
k2=convolve(k1,Laplacian);
axes(handles.axes2);
imshow(k2);
title('Laplacian Filter');

function B = convolve(A, k)
[r c] = size(A);
[m n] = size(k);
h = rot90(k, 2);
center = floor((size(h)+1)/2);
left = center(2) - 1;
right = n - center(2);
top = center(1) - 1;
bottom = m - center(1);
Rep = zeros(r + top + bottom, c + left + right);
for x = 1 + top : r + top
    for y = 1 + left : c + left
        Rep(x,y) = A(x - top, y - left);
    end
end
B = zeros(r , c);
for x = 1 : r
    for y = 1 : c
        for i = 1 : m
            for j = 1 : n
                q = x - 1;
                w = y -1;
                B(x, y) = B(x, y) + (Rep(i + q, j + w) * h(i, j));
            end
        end
    end
end
% --- Executes on button press in pushbutton6.
function pushbutton6_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton6 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
laplacian(hObject, eventdata, handles,-4);




% --- Executes on slider movement.
function slider4_Callback(hObject, eventdata, handles)
% hObject    handle to slider4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'Value') returns position of slider
%        get(hObject,'Min') and get(hObject,'Max') to determine range of slider
sliderVal=get(hObject,'Value');
assignin('base',"sliderVal",sliderVal);   
sliderVal = round(sliderVal);
sliderVal= sliderVal+1;
im=imread('coins.png');
NA=addnoise(hObject, eventdata, handles,im,10);
median(hObject, eventdata, handles,sliderVal,sliderVal,NA);


% --- Executes during object creation, after setting all properties.
function slider4_CreateFcn(hObject, eventdata, handles)
% hObject    handle to slider4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: slider controls usually have a light gray background.
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end


% --- Executes on slider movement.
function slider5_Callback(hObject, eventdata, handles)
% hObject    handle to slider5 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'Value') returns position of slider
%        get(hObject,'Min') and get(hObject,'Max') to determine range of slider
sliderVal=get(hObject,'Value');
assignin('base',"sliderVal",sliderVal);   
sliderVal = round(sliderVal);
GaussianFilter(hObject, eventdata, handles,sliderVal+1);


% --- Executes during object creation, after setting all properties.
function slider5_CreateFcn(hObject, eventdata, handles)
% hObject    handle to slider5 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: slider controls usually have a light gray background.
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end


% --- Executes on slider movement.
function slider6_Callback(hObject, eventdata, handles)
% hObject    handle to slider6 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'Value') returns position of slider
%        get(hObject,'Min') and get(hObject,'Max') to determine range of slider
sliderVal=get(hObject,'Value');
assignin('base',"sliderVal",sliderVal);   
sliderVal = round(sliderVal);
Histogram(hObject, eventdata, handles,sliderVal);


% --- Executes during object creation, after setting all properties.
function slider6_CreateFcn(hObject, eventdata, handles)
% hObject    handle to slider6 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: slider controls usually have a light gray background.
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end


% --- Executes on slider movement.
function slider8_Callback(hObject, eventdata, handles)
% hObject    handle to slider8 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'Value') returns position of slider
%        get(hObject,'Min') and get(hObject,'Max') to determine range of slider
im=imread('coins.png');
sliderVal=get(hObject,'Value');
assignin('base',"sliderVal",sliderVal);   
sliderVal = round(sliderVal);
addnoise(hObject, eventdata, handles,im,sliderVal+10);


% --- Executes during object creation, after setting all properties.
function slider8_CreateFcn(hObject, eventdata, handles)
% hObject    handle to slider8 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: slider controls usually have a light gray background.
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end


% --- Executes on slider movement.
function slider9_Callback(hObject, eventdata, handles)
% hObject    handle to slider9 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'Value') returns position of slider
%        get(hObject,'Min') and get(hObject,'Max') to determine range of slider
im=imread('cameraman.tif');
sliderVal=get(hObject,'Value');
assignin('base',"sliderVal",sliderVal);   
sliderVal = round(sliderVal);
Interpolation(hObject, eventdata, handles,sliderVal+1);


% --- Executes during object creation, after setting all properties.
function slider9_CreateFcn(hObject, eventdata, handles)
% hObject    handle to slider9 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: slider controls usually have a light gray background.
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end

function sobelEdge(hObject, eventdata, handles)
A=imread('coins.png');
axes(handles.axes1);
imshow(A);
title('Original Pic');
B=im2gray(A);

C=double(B);


for i=1:size(C,1)-2
    for j=1:size(C,2)-2
        %Sobel mask for x-direction:
        Gx=((2*C(i+2,j+1)+C(i+2,j)+C(i+2,j+2))-(2*C(i,j+1)+C(i,j)+C(i,j+2)));
        %Sobel mask for y-direction:
        Gy=((2*C(i+1,j+2)+C(i,j+2)+C(i+2,j+2))-(2*C(i+1,j)+C(i,j)+C(i+2,j)));
     
        %The gradient of the image
        %B(i,j)=abs(Gx)+abs(Gy);
        B(i,j)=sqrt(Gx.^2+Gy.^2);
     
    end
end
axes(handles.axes2);
imshow(B); title('Sobel gradient');
% --- Executes on button press in pushbutton7.
function pushbutton7_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton7 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
sobelEdge(hObject, eventdata, handles);
