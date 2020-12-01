import cv2
import numpy as np
import pandas as pd
import re



def Header_Boundary(img,scaling_factor):
    crop_img=img[:1200,:6800,:].copy()
    blur_cr_img=cv2.blur(crop_img,(7,7))
    crop_img_resize=cv2.resize(blur_cr_img, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
    crop_img_resize_n=cv2.fastNlMeansDenoisingColored(crop_img_resize,None,10,10,7,21)
    crop_img_resize_n_gray=cv2.cvtColor(crop_img_resize_n,cv2.COLOR_BGR2GRAY)
    th3 = cv2.adaptiveThreshold(crop_img_resize_n_gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,37,1)
    max_start=int(np.argmax(np.sum(th3,axis=1))/scaling_factor)
    return max_start

def rotate_check_column_border(img,th3,angle,scaling_factor,imgshape_f):
    th4=rotate(th3,angle,value_replace=0)
    th_sum00=np.sum(th4,axis=0)
    empty_spc_clm=np.where(th_sum00<(np.min(th_sum00)+100))[0]
    empty_spc_clm_dif=np.ediff1d(empty_spc_clm)
    Column_boundries=(empty_spc_clm[np.where(empty_spc_clm_dif>np.mean(empty_spc_clm_dif))[0]+1]/(scaling_factor/2)).astype(int)
    
    Column_boundries=np.delete(Column_boundries,np.where(Column_boundries<(img.shape[1]/5))[0])
    
    Column_boundries=np.append(Column_boundries,[0,img.shape[1]])
    Column_boundries=np.unique(Column_boundries)
    for i in range(len(Column_boundries)):
        closer=np.where(np.ediff1d(Column_boundries)<(img.shape[1])/5)[0]
        if len(closer)>0:
            Column_boundries=np.delete(Column_boundries,closer[-1])
        else:
            break
    
    #[2968 7864 8016]
    
    return Column_boundries[1:]

def rotate(image, angle, center = None, scale = 1.0,value_replace=0):
    (h, w) = image.shape[:2]
    if center is None:
        center = (w / 2, h / 2)
    # Perform the rotation
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h),borderValue=value_replace)
    return rotated

def method5_column(img,th3,scaling_factor,angle_rec,morph_op=False):
    for ang in angle_rec:
        if morph_op:
            th4=rotate(th3.copy(),ang)
        else:
            kernel=np.ones((100,9),np.uint8)
            th4=cv2.morphologyEx(rotate(th3.copy(),ang), cv2.MORPH_CLOSE, kernel)
        # cv2.imshow('morphologyEx',th4)
        # cv2.waitKey(0)
        th4=cv2.bitwise_not(th4)
        # cv2.imshow('bitwise_not',th4)
        # cv2.waitKey(0)
        th4[th4==255]=1
        # print([np.sum(th4,axis=0)])
        # print(np.max(np.sum(th4,axis=0)),np.mean(np.sum(th4,axis=0)))
        split_candidates=np.where(np.sum(th4,axis=0)>=(np.max(np.sum(th4,axis=0))-np.mean(np.sum(th4,axis=0))))[0]
        split_candidates=np.unique(np.append(split_candidates,[0,th4.shape[1]]))
        
        empty_spc_clm_dif=np.ediff1d(split_candidates)
        Column_boundries=(split_candidates[np.where(empty_spc_clm_dif>np.mean(empty_spc_clm_dif))[0]+1]/(scaling_factor/2)).astype(int)
        # print('Col0umn_boundries1:',Column_boundries)
        
        Column_boundries=np.append(Column_boundries,[0,img.shape[1]])
        Column_boundries=np.unique(Column_boundries)
        for i in range(len(Column_boundries)):
            closer=np.where(np.ediff1d(Column_boundries)<(img.shape[1])/5)[0]
            if len(closer)>0:
                Column_boundries=np.delete(Column_boundries,closer[-1])
            else:
                break
        Column_boundries=Column_boundries[1:]
        # print('Column_boundries2:',Column_boundries)
        if len(Column_boundries)>2:
            break
    return Column_boundries,ang
        

def row_split_smaller(th3,image_row_split_ratio,angle,scaling_factor):
    th4=rotate(th3.copy(),angle,value_replace=0)
    image_row_th=int(th4.shape[0]/image_row_split_ratio)
    row_sum_location=np.where(np.sum(th4,axis=1)<2)[0]
    row_sum_location=row_sum_location[np.where(np.ediff1d(row_sum_location)==1)[0]]

    row_split_pos1=[]
#     print('row_sum_location:',row_sum_location)
    for i in range(image_row_split_ratio):
        split_s=row_sum_location[np.where((row_sum_location-(image_row_th*i))>=0)[0]]
#         print('split_s:',split_s)
        try:
            point_place=split_s[np.where(split_s>row_split_pos1[-1]+int(image_row_th/3))[0][0]]
            row_split_pos1.append(point_place)
        except:
            if len(split_s)>0:
                row_split_pos1.append(split_s[0])
    row_split_pos1=np.array(row_split_pos1)
    row_split_pos1=np.append(row_split_pos1,[0,th4.shape[0]])
    
    row_split_pos1=np.unique(row_split_pos1)
    for i in range(len(row_split_pos1)):
        closer=np.where(np.ediff1d(row_split_pos1)<(th4.shape[0])/5)[0]
        if len(closer)>0:
            row_split_pos1=np.delete(row_split_pos1,closer[-1])
        else:
            break
    if row_split_pos1[0]<(th4.shape[0])/5:
        row_split_pos1[0]=0
    return (row_split_pos1/(scaling_factor)).astype(int)

def angle_out_row(column_img,scaling_factor):
    blur_cr_img=cv2.blur(column_img,(13,13))
    crop_img_resize=cv2.resize(blur_cr_img, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
    crop_img_resize_n=cv2.fastNlMeansDenoisingColored(crop_img_resize,None,10,10,7,21)
    crop_img_resize_n_gray=cv2.cvtColor(crop_img_resize_n,cv2.COLOR_BGR2GRAY)

    th3 = cv2.adaptiveThreshold(crop_img_resize_n_gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,21,7)
    Angles_outputs=[]
    for angle in [-0.1,0.2,-0.2,0.3,-0.3,0.4,-0.4,0.5,-0.5]:
        result=row_split_smaller(th3,int(1/scaling_factor),angle,scaling_factor)
        Angles_outputs.append([angle,len(result),result])

    Angles_outputs=np.array(Angles_outputs,dtype=object)
    set_angle,_,row_split_pos1=Angles_outputs[np.argmax(Angles_outputs[:,1])]
    return set_angle,row_split_pos1

def Row_splitter(column_img,scaling_factor,image_row_split_ratio=8):

    blur_cr_img=cv2.blur(column_img,(13,13))
    crop_img_resize=cv2.resize(blur_cr_img, None, fx=scaling_factor/2, fy=scaling_factor/2, interpolation=cv2.INTER_AREA)
    crop_img_resize_n=cv2.fastNlMeansDenoisingColored(crop_img_resize,None,10,10,7,21)
    crop_img_resize_n_gray=cv2.cvtColor(crop_img_resize_n,cv2.COLOR_BGR2GRAY)
    th3 = cv2.adaptiveThreshold(crop_img_resize_n_gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,21,7)
    image_row_th=int(th3.shape[0]/image_row_split_ratio)
    row_sum_location=np.where(np.sum(th3,axis=1)<2)[0]
    row_sum_location=row_sum_location[np.where(np.ediff1d(row_sum_location)==1)[0]]
    
#     print(row_sum_location)
    row_split_pos=[]
    for i in range(image_row_split_ratio):
        split_s=row_sum_location[np.where((row_sum_location-(image_row_th*i))>=0)[0]]
#         print(split_s)
        try:
            
            point_place=split_s[np.where(split_s>row_split_pos[-1]+int(image_row_th/3))[0][0]]
#             print(split_s,point_place)
            row_split_pos.append(point_place)
        except:
            if len(split_s)>0:
                row_split_pos.append(split_s[0])
    
    row_split_pos=np.array(row_split_pos)
    row_split_pos=np.append(row_split_pos,[0,th3.shape[0]])
    row_split_pos=np.unique(row_split_pos)
    for i in range(len(row_split_pos)):
        closer=np.where(np.ediff1d(row_split_pos)<(th3.shape[0])/10)[0]
        if len(closer)>0:
            row_split_pos=np.delete(row_split_pos,closer[-1])
        else:
            break
    
    row_split_pos=(row_split_pos/(scaling_factor/2)).astype(int)
    return np.unique(row_split_pos)

def Column_main_Extracter_sub(img,scaling_factor):
    blur_cr_img=cv2.blur(img,(13,13))
    crop_img_resize=cv2.resize(blur_cr_img, None, fx=scaling_factor/2, fy=scaling_factor/2, interpolation=cv2.INTER_AREA)

    crop_img_resize_n=cv2.fastNlMeansDenoisingColored(crop_img_resize,None,10,10,7,21)
    crop_img_resize_n_gray=cv2.cvtColor(crop_img_resize_n,cv2.COLOR_BGR2GRAY)
    th3 = cv2.adaptiveThreshold(crop_img_resize_n_gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,21,7)

    kernel=np.ones((100,9),np.uint8)
    th4=cv2.morphologyEx(th3.copy(), cv2.MORPH_CLOSE, kernel)
    th_sum00=np.sum(th4,axis=0)
    empty_spc_clm=np.where(th_sum00<(np.min(th_sum00)+100))[0]
    empty_spc_clm_dif=np.ediff1d(empty_spc_clm)
    Column_boundries=(empty_spc_clm[np.where(empty_spc_clm_dif>np.mean(empty_spc_clm_dif)+5)[0]+1]/(scaling_factor/2)).astype(int)
    Column_boundries=np.delete(Column_boundries,np.where(Column_boundries<(img.shape[1]/5))[0])

    if len(Column_boundries)<3:
        Angles_Records=[]
        for angle in [0.1,-0.1,0.2,-0.2,0.3,-0.3,0.4,-0.4,0.5,-0.5,0.6,-0.6,0.7,-0.7,0.8,-0.8]:
            Column_boundries=rotate_check_column_border(img,th3.copy(),angle,scaling_factor,img.shape[1])
#             print(Column_boundries)
            Angles_Records.append([angle,len(Column_boundries)])
            if len(Column_boundries)>2:
                break
        Angles_Records=np.array(Angles_Records)
        if len(Column_boundries)>2:
            img=rotate(img,angle,value_replace=(255,255,255))
            First_Column=img[:,0:Column_boundries[0]+10]
            Second_Column=img[:,Column_boundries[0]:Column_boundries[1]+10]
            Third_Column=img[:,Column_boundries[1]:]
        else:
            angle=np.append([0],Angles_Records)
            angle_rec=Angles_Records[np.where(Angles_Records[:,1]==np.max(Angles_Records[:,1]))[0]][:,0]
            Column_boundries,ang=method5_column(img,th3,scaling_factor,angle_rec)
            if len(Column_boundries)>2:
                img=rotate(img,ang,value_replace=(255,255,255))
                First_Column=img[:,0:Column_boundries[0]+10]
                Second_Column=img[:,Column_boundries[0]:Column_boundries[1]+10]
                Third_Column=img[:,Column_boundries[1]:]
            else:
                return None,None,None
    else:   
        First_Column=img[:,0:Column_boundries[0]+10]
        Second_Column=img[:,Column_boundries[0]:Column_boundries[1]+10]
        Third_Column=img[:,Column_boundries[1]:]
    
    return First_Column,Second_Column,Third_Column

def Column_main_Extracter_sub_second(img,orignal_img,scaling_factor):
    blur_cr_img=cv2.blur(orignal_img,(13,13))
    crop_img_resize=cv2.resize(blur_cr_img, None, fx=scaling_factor/2, fy=scaling_factor/2, interpolation=cv2.INTER_AREA)

    crop_img_resize_n=cv2.fastNlMeansDenoisingColored(crop_img_resize,None,10,10,7,21)
    crop_img_resize_n_gray=cv2.cvtColor(crop_img_resize_n,cv2.COLOR_BGR2GRAY)
    th3 = cv2.adaptiveThreshold(crop_img_resize_n_gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,21,7)

    top=int(th3.shape[1]/40)
    bottom=int(th3.shape[1]/40)
    left=int(th3.shape[1]/20)
    right=int(th3.shape[1]/20)
    th4 = cv2.copyMakeBorder(th3,top=top,bottom=bottom,left=left,right=right,borderType=cv2.BORDER_CONSTANT,value=0)
    
    for angle in [0.1,-0.1,0.2,-0.2,0.3,-0.3,0.4,-0.4,0.5,-0.5,0.6,-0.6,0.7,-0.7,0.8,-0.8]:
        th5=rotate(th4.copy(), angle)
        kernel=np.ones((30,9),np.uint8)
        th5=cv2.morphologyEx(th5.copy(), cv2.MORPH_CLOSE, kernel)

        th5=cv2.bitwise_not(th5)
        th5[th5<255]=0
        th5[th5==255]=1

        
        split_candidates=np.where(np.sum(th5,axis=0)>=(np.max(np.sum(th5,axis=0))-(np.mean(np.sum(th5,axis=0))/1.5)))[0]
        split_candidates=np.unique(np.append(split_candidates,[0,th5.shape[1]]))

        empty_spc_clm_dif=np.ediff1d(split_candidates)
        Column_boundries=(split_candidates[np.where(empty_spc_clm_dif>np.mean(empty_spc_clm_dif))[0]+1]/(scaling_factor/2)).astype(int)
        
        Column_boundries=np.append(Column_boundries,[0,img.shape[1]])
        Column_boundries=np.unique(Column_boundries)
        for i in range(len(Column_boundries)):
            closer=np.where(np.ediff1d(Column_boundries)<(img.shape[1])/5)[0]
            if len(closer)>0:
                Column_boundries=np.delete(Column_boundries,closer[-1])
            else:
                break
        Column_boundries=Column_boundries[1:]
        
        if len(Column_boundries)>2:
            # print(Column_boundries,np.mean(np.sum(th5,axis=0)),np.max(np.sum(th5,axis=0)))
            img=rotate(img,angle,value_replace=(255,255,255))
            First_Column=img[:,0:Column_boundries[0]+10]
            Second_Column=img[:,Column_boundries[0]:Column_boundries[1]+10]
            Third_Column=img[:,Column_boundries[1]:]
        else:
            First_Column,Second_Column,Third_Column=None,None,None
    
        if First_Column is not None:
            break
    return First_Column,Second_Column,Third_Column

def Column_main_Extracter(img,orignal_img,scaling_factor):
    blur_cr_img=cv2.blur(img,(13,13))
    crop_img_resize=cv2.resize(blur_cr_img, None, fx=scaling_factor/2, fy=scaling_factor/2, interpolation=cv2.INTER_AREA)

    crop_img_resize_n=cv2.fastNlMeansDenoisingColored(crop_img_resize,None,10,10,7,21)
    crop_img_resize_n_gray=cv2.cvtColor(crop_img_resize_n,cv2.COLOR_BGR2GRAY)
    th3 = cv2.adaptiveThreshold(crop_img_resize_n_gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,21,7)

    kernel=np.ones((100,9),np.uint8)
    th4=cv2.morphologyEx(th3.copy(), cv2.MORPH_CLOSE, kernel)
    th_sum00=np.sum(th4,axis=0)
    empty_spc_clm=np.where(th_sum00<(np.min(th_sum00)+100))[0]
    empty_spc_clm_dif=np.ediff1d(empty_spc_clm)
    Column_boundries=(empty_spc_clm[np.where(empty_spc_clm_dif>np.mean(empty_spc_clm_dif)+5)[0]+1]/(scaling_factor/2)).astype(int)
    Column_boundries=np.delete(Column_boundries,np.where(Column_boundries<(img.shape[1]/5))[0])
    # print('Column_boundries:1',Column_boundries)
    if len(Column_boundries)<3:
        Angles_Records=[]
        for angle in [0.1,-0.1,0.2,-0.2,0.3,-0.3,0.4,-0.4,0.5,-0.5,0.6,-0.6,0.7,-0.7,0.8,-0.8]:
            Column_boundries=rotate_check_column_border(img,th3.copy(),angle,scaling_factor,img.shape[1])
            # print(Column_boundries)
            ############################################
            Column_boundries=np.append(Column_boundries,[0,img.shape[1]])
            Column_boundries=np.unique(Column_boundries)
            for i in range(len(Column_boundries)):
                closer=np.where(np.ediff1d(Column_boundries)<(img.shape[1])/5)[0]
                if len(closer)>0:
                    Column_boundries=np.delete(Column_boundries,closer[-1])
                else:
                    break
            Column_boundries=Column_boundries[1:]
            ############################################
            
            Angles_Records.append([angle,len(Column_boundries)])
            if len(Column_boundries)>2:
                break
        Angles_Records=np.array(Angles_Records)
        if len(Column_boundries)>2:
            img=rotate(img,angle,value_replace=(255,255,255))
            First_Column=img[:,0:Column_boundries[0]+10]
            Second_Column=img[:,Column_boundries[0]:Column_boundries[1]+10]
            Third_Column=img[:,Column_boundries[1]:]
        else:
            angle=np.append([0],Angles_Records)
            angle_rec=Angles_Records[np.where(Angles_Records[:,1]==np.max(Angles_Records[:,1]))[0]][:,0]
            
            Column_boundries,ang=method5_column(img,th3,scaling_factor,angle_rec)
            if len(Column_boundries)>2:
                img=rotate(img,ang,value_replace=(255,255,255))
                First_Column=img[:,0:Column_boundries[0]+10]
                Second_Column=img[:,Column_boundries[0]:Column_boundries[1]+10]
                Third_Column=img[:,Column_boundries[1]:]
            else:
                First_Column,Second_Column,Third_Column=Column_main_Extracter_sub_second(img,orignal_img,scaling_factor)
                
                if First_Column is None:
                    First_Column,Second_Column,Third_Column=Column_main_Extracter_sub(orignal_img,scaling_factor)
                    # print([First_Column])
                
    else:
        First_Column=img[:,0:Column_boundries[0]+10]
        Second_Column=img[:,Column_boundries[0]:Column_boundries[1]+10]
        Third_Column=img[:,Column_boundries[1]:]
    return First_Column,Second_Column,Third_Column

def Image_blur_for_low_size(img,scaling_factor):
    blur_cr_img=cv2.blur(img,(5,5))
    if not round(blur_cr_img.nbytes / 1024 / 1024,2)>4.0:
        blur_cr_img=cv2.resize(blur_cr_img, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
    crop_img_resize_n=cv2.fastNlMeansDenoisingColored(blur_cr_img,None,10,10,5,4)
    i=0
    if not round(crop_img_resize_n.nbytes / 1024 / 1024,2)<4.0:
        for i in range(10):
            crop_img_resize_n=cv2.resize(crop_img_resize_n, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
            if round(crop_img_resize_n.nbytes / 1024 / 1024,2)<4.0:
                break
    return crop_img_resize_n,i



def get_croped_images(image_path):
    img=cv2.imread(image_path)

    img_height,img_width,_=img.shape
    scaling_factor=0.25

    max_start=Header_Boundary(img,scaling_factor)
    orignal_img=img[max_start:,:,:]
    img = cv2.copyMakeBorder(orignal_img,top=int(orignal_img.shape[1]/40),bottom=int(orignal_img.shape[1]/40),
        left=int(orignal_img.shape[1]/20),right=int(orignal_img.shape[1]/20),borderType=cv2.BORDER_CONSTANT,
        value=[255, 255, 255])

    First_Column,Second_Column,Third_Column=Column_main_Extracter(img,orignal_img,scaling_factor)
    # cv2.imwrite(image_path.replace('.jpg','_first.png'),First_Column)
    # cv2.imwrite(image_path.replace('.jpg','_second.png'),Second_Column)
    # cv2.imwrite(image_path.replace('.jpg','_third.png'),Third_Column)
    
    Images_croped_saver=[]
    
    i=0
    for image in [First_Column,Second_Column,Third_Column]:
        row_split_pos=Row_splitter(image,scaling_factor,image_row_split_ratio=8)
        gn=0
        for g in range(len(row_split_pos)-1):
            CROP_Image=image[row_split_pos[g]:row_split_pos[g+1],:]
            if CROP_Image.shape[0]>image.shape[0]/2:
                crp_image = cv2.copyMakeBorder(CROP_Image,top=0,bottom=0,left=int(CROP_Image.shape[1]/20),right=int(CROP_Image.shape[1]/20),borderType=cv2.BORDER_CONSTANT,value=[255, 255, 255])
                angle,row_sp_pos=angle_out_row(crp_image,scaling_factor)
                crp_image=rotate(crp_image,angle,value_replace=(255,255,255))
                for gn in range(len(row_sp_pos)-1):
                    crp_image_n=crp_image[row_sp_pos[gn]:row_sp_pos[gn+1],:]
                    crp_image_n,down_i=Image_blur_for_low_size(crp_image_n,scaling_factor)
                    Images_croped_saver.append([crp_image_n,down_i])
#                     cv2.imwrite(''+image_path.split('.jpg')[0]+'__'+str(i)+'__'+str(g)+'__'+str(gn)+'.png',crp_image_n)
            else:
                CROP_Image,down_i=Image_blur_for_low_size(CROP_Image,scaling_factor)
                Images_croped_saver.append([CROP_Image,down_i])
#                 cv2.imwrite(''+image_path.split('.jpg')[0]+'___'+str(i)+'__'+str(g)+'__'+str(gn)+'.png',CROP_Image)
        i+=1
    return Images_croped_saver

    
def text_spliter(Text_line,cntact_attention=False):
    Title=''
    first_name=''
    Surname=''
    address=''
    Surname1=''
    contact_number=''
    text_a=''
#     print(Text_line)
    ####--------------------------------------------------####
    if cntact_attention:
        if len(Text_line.split('..'))>1:
            address=Text_line.split('..')[0]
            contact_number=Text_line.split('..')[-1]
        if len(Text_line.split('('))>1:
            address=Text_line.split('(')[0]
            contact_number=Text_line.split('(')[-1]
        if contact_number=='':
            num_r0=re.findall(r'\d+',address)
            for num_r in num_r0:
                if len(num_r)>3:
                    address,contact_number=address[:address.index(num_r)],address[address.index(num_r):]
                    break
        if contact_number=='.':
            contact_number=''
        if contact_number!='':
            contact_number=contact_number.replace('.','').replace('*','')
            if '(' not in contact_number:
                contact_number='('+contact_number
                
        if len(address)>20:
            num_r0=re.findall(r'\d+',address)
            for num_r in num_r0:
                if len(num_r)>3:
                    address,extra_text=address[:address.index(num_r)],address[address.index(num_r):]
                    break
        if len(''.join(re.findall(r'\d+',contact_number)))<6:
            contact_number=''
        contact_number=''.join([g for g in contact_number if not g.isalpha()])
        return Surname,Title,first_name,address,contact_number
    ####--------------------------------------------------####    
    if len(Text_line.split(','))==3:
        first_name,Title,text_a=Text_line.split(',')
        
        if len(Title)>14:
            
            first_name=Text_line.split(',')[0]
            text_a=','.join(Text_line.split(',')[1:])
            Title=''
            
    if len(Text_line.split(','))==4:
        first_name=Text_line.split(',')[0]
        text_a=','.join(Text_line.split(',')[1:])
        Title=''
        
    if len(Text_line.split(','))==2:
        first_name,text_a=Text_line.split(',')
        
    if len(Text_line.split(','))==1:
        # print(Text_line)
        if '***' in Text_line:
            Text_line=Text_line.replace('**','')
        if len(Text_line.split('..'))>1:
            address=Text_line.split('..')[0]
            contact_number=Text_line.split('..')[-1]
        
        if '(' not in Text_line:
            if ')' not in Text_line:
                if len(Text_line.split('.'))==2:
                    first_name,address=Text_line.split('.')
                else:
                    address=Text_line
        else:
            if ')' in Text_line:
                address,contact_number=Text_line.split('(')
            
       
    else:     
        # print([first_name,text_a])
        if '(' in text_a:
            if text_a[-1]=='(':
                text_a=text_a[:-1]
            text_a_spliter=text_a.replace('( (',' (').split('(')
            if len(text_a_spliter)==2:
                text_a_spliter=text_a.replace('( (',' (').split('(')
                address,contact_number=text_a_spliter
                
            
            elif len(text_a_spliter)==3:
                address,_,contact_number=text_a.replace('( (',' (').split('(')
            else:
                address=text_a
        else:
            address,contact_number=text_a,''
        if len(first_name.split())==2:
            Surname,first_name=first_name.split()
        elif len(first_name.split())==3:
            Surname,Surname1,first_name=first_name.split()
        else:
            Surname=''
            Surname1=''
        
        if len(first_name.split())==4:
            handling_text=first_name.split()
            first_name=' '.join(handling_text[0:2])
            address=' '.join(handling_text[2:])+' '+address
            
        if Title!='':
            if Title not in ['Mr','Ms','Mrs','Miss','Sir','Dr']:
                address=','.join([Title,address])
                Title=''
            
        if Surname in ['Mr','Ms','Mrs','Miss','Sir','Dr']:
            Title=Surname
            Surname=Surname1
        if Surname1 in ['Mr','Ms','Mrs','Miss','Sir','Dr']:
            Title=Surname1
    
    if contact_number=='':
        num_r0=re.findall(r'\d+',address)
        for num_r in num_r0:
            if len(num_r)>3:
                address,contact_number=address[:address.index(num_r)],address[address.index(num_r):]
                break
    if contact_number=='.':
        contact_number=''
    if contact_number!='':
        contact_number=contact_number.replace('.','').replace('*','')
        if '(' not in contact_number:
            contact_number='('+contact_number
            
    if first_name.isdigit():
        first_name=''
    if Title=='':
        for g in ['Mr','Ms','Mrs','Miss','Sir','Dr']:
            if g in first_name:
                first_name=first_name.replace(g,'')
                Title=g
                break
                
    if len(address)>20:
        num_r0=re.findall(r'\d+',address)
        for num_r in num_r0:
            if len(num_r)>3:
                address,extra_text=address[:address.index(num_r)],address[address.index(num_r):]
                break
    if len(''.join(re.findall(r'\d+',contact_number)))<6:
        contact_number=''
    
    contact_number=''.join([g for g in contact_number if not g.isalpha()])
    
    return Surname,Title,first_name,address,contact_number

def ocr_text_to_pd(resp,down_i,cntact_attention,th=20):
    Results=[]
    for i in range(1,len(resp.text_annotations)):
        text=resp.text_annotations[i].description
        cordinates=np.array([[cor.x,cor.y] for cor in resp.text_annotations[i].bounding_poly.vertices])
        xmin,ymin,xmax,ymax=np.min(cordinates[:,0]),np.min(cordinates[:,1]),np.max(cordinates[:,0]),np.max(cordinates[:,1])
        Results.append([text,xmin,ymin,xmax,ymax])

    Results=pd.DataFrame(Results,columns=['text','xmin','ymin','xmax','ymax'])

    Results=Results.sort_values('ymin')
    Results=Results.reset_index(drop=True)

    Values=Results['ymin'].values+(Results['ymax'].values-Results['ymin'].values)/2
    Values_dif=np.ediff1d(Values)
    Values_dif[Values_dif<0]=0
    Values_dif[Values_dif<(th/(1+(down_i*2)))]=0
    Values_dif=np.append([0],Values_dif)
    Values_dif_p=np.where(Values_dif>0)[0]
    Values_dif_p=np.append([0,len(Results)],Values_dif_p)
    Values_dif_p=np.unique(Values_dif_p)

    Pd_results=[]
#     cntact_attention=False
    for g in range(len(Values_dif_p)-1):
        Text_line=' '.join(list(Results.iloc[Values_dif_p[g]:Values_dif_p[g+1],:].sort_values('xmin')['text']))
        Surname,Title,first_name,address,contact_number=text_spliter(Text_line,cntact_attention=cntact_attention)
        if len(''.join([Surname,Title,first_name,address,contact_number]))>0:
#             print([Surname,Title,first_name,contact_number,address])
            Pd_results.append([Surname,Title,first_name,contact_number,address])
        if contact_number=='':
            if address!='':
                cntact_attention=True
            else:
                cntact_attention=False
        else:
            cntact_attention=False
    Pd_results=pd.DataFrame(Pd_results,columns=['Surname','Title','First name','Tel','Address'])
    return Pd_results,cntact_attention

def Pandas_manipulation(Pd_results_all):
    Indexes=Pd_results_all[Pd_results_all['Tel']==''].index #isnull()
    if len(Pd_results_all)==Indexes[-1]+1:
        Indexes=Indexes[:-1]
    
    for i in range(len(Indexes)):
        
        Pd_results_all.values[Indexes[i]][3]=Pd_results_all.values[Indexes[i]+1][3]
        Pd_results_all.values[Indexes[i]][4]=Pd_results_all.values[Indexes[i]][4]+' '+Pd_results_all.values[Indexes[i]+1][4]

    Pd_results_all_np=np.delete(Pd_results_all.values,Indexes+1,axis=0)

    Pd_results_all_np=pd.DataFrame(Pd_results_all_np,columns=Pd_results_all.columns)
    Pd_results_all_np['Surname']=Pd_results_all_np['Surname'].replace('', np.NaN)
    Pd_results_all_np['Surname']=Pd_results_all_np['Surname'].fillna(method='ffill')
    return Pd_results_all_np


