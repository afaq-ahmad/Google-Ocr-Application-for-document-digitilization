#pyinstaller.exe -c --onefile  ocr_ui.py
import ocr_backend
import tkinter
import os.path
import os
import threading
import sys
import time

from tkinter import filedialog
from tkinter import scrolledtext
from tkinter import messagebox
from tkinter import ttk
from datetime import datetime
import cv2

import pandas as pd
from utils import ocr_text_to_pd,get_croped_images,Pandas_manipulation
import ocr_backend
directory_temp='processing_images/'



source_directory = None

file_label = None
log_area = None
progress_bar = None

write_text = None
write_ms_word_doc = None
rotate = None

def select_directory():
    
    source_directory_str = filedialog.askdirectory() 

    if (not os.path.exists(str(source_directory_str)) ): 
        file_label["text"] = "<no directory selected>"
        source_directory.set("")

    else:

        file_label["text"] = source_directory_str
        source_directory.set(str(source_directory_str))

    #print("Set source directory to {0}".format(source_directory))

class RunOCR(threading.Thread):

    def __init__(self, directory,directory_temp):

        threading.Thread.__init__(self)
        self.directory = directory
        self.directory_temp=directory_temp

    def run(self):

        supported_extensions = ["jpg", "png", "gif", "bmp", "webp", "raw", "ico", "pdf", "tiff"]

        valid_files = {}

        for file_name in os.listdir(self.directory):

            if (not os.path.isfile(self.directory + os.sep + file_name) ):

                log_area.insert(tkinter.END, "Skipping directory: {0}\n".format(file_name))
                continue 

            extension_index = file_name.rfind(".")

            if (extension_index >= 0):

                extension = file_name[extension_index + 1:].lower()

                if (extension in supported_extensions):

                    #already processed
                    if ( os.path.exists(self.directory + os.sep + file_name + ".xlsx") ):

                        log_area.insert(tkinter.END, "Skipping file with matching Excel file: {0}\n".format(file_name))

                    else:
                        valid_files[self.directory + os.sep + file_name] = file_name

                else:
                    log_area.insert(tkinter.END, "Skipping unrecognized file: {0}\n".format(file_name))

        num_image_files = len(valid_files)
        
        
        progress_bar_step = 100.0

        if (num_image_files > 0 ):
            progress_bar_step = 100.0 / num_image_files
       
        # print(valid_files)
        

        if not os.path.exists(self.directory_temp):
            os.makedirs(self.directory_temp)
        
        for image_path in valid_files:

            print("processing {0}".format(image_path))

            file_name = valid_files[image_path]

            now = datetime.now()

            log_area.insert(tkinter.END, "{1} Processing file: {0}\n".format(file_name, now.strftime("%m/%d/%Y %H:%M:%S")))

            compress_attempted = False

            #call backend
            try:

                #upload
                now = datetime.now()
                log_area.insert(tkinter.END, "{1} Uploading to GCS {0}\n".format(file_name, now.strftime("%m/%d/%Y %H:%M:%S")))

            
                Pd_results_all=pd.DataFrame(columns=['Surname','Title','First name','Tel','Address'])

                Images_croped_saver=get_croped_images(image_path)
                cntact_attention=False
                
                now = datetime.now()
                log_area.insert(tkinter.END, "{1} Successfully Divided image into parts {0}\n".format(file_name, now.strftime("%m/%d/%Y %H:%M:%S")))

                for im_n in range(len(Images_croped_saver)):
                    img,down_i=Images_croped_saver[im_n][0],Images_croped_saver[im_n][1]

                    try:
                        img_path_temp=self.directory_temp+'/current_img.png'
                        cv2.imwrite(img_path_temp,img)
                        gcs_location=ocr_backend.upload(img_path_temp)
                        resp = ocr_backend.extract_text(gcs_location)
                        th=20
                        Pd_results,cntact_attention=ocr_text_to_pd(resp,down_i,cntact_attention,th=20)
                        Pd_results_all=Pd_results_all.append(Pd_results,ignore_index=True)
                        ocr_backend.delete_blob(gcs_location)
                    
                    except Exception as excptn:
                        now = datetime.now()
                        log_area.insert(tkinter.END, "{1} Error processing Current Crop file {2}: {0}\n".format(img_path_temp, now.strftime("%m/%d/%Y %H:%M:%S"), str(excptn)))
                
                Pd_results_all=Pandas_manipulation(Pd_results_all)
                
                cntact_attention=False

                #save to Excel
                now = datetime.now()
                log_area.insert(tkinter.END, "{1} Saving phonebook to Excel {0}\n".format(file_name, now.strftime("%m/%d/%Y %H:%M:%S")))

                excel_path = self.directory + os.sep + file_name + ".xlsx"
                
                Pd_results_all.to_excel(excel_path,index=False)
                

            except Exception as excptn:

                now = datetime.now()
                log_area.insert(tkinter.END, "{1} Error processing file {2}: {0}\n".format(file_name, now.strftime("%m/%d/%Y %H:%M:%S"), str(excptn)))

            #update progress bar
            progress_bar.step(progress_bar_step)

        
        now = datetime.now()
        log_area.insert(tkinter.END, "{0} : Complete!\n".format(now.strftime("%m/%d/%Y %H:%M:%S")))


def run_ocr():

    source_dir = source_directory.get()
    if (source_dir == ""):
        messagebox.showerror("Error", "No directory selected")
        return

    ocr_thread = RunOCR(source_dir,directory_temp)
    ocr_thread.start()

    pass

#draw window
main_window = tkinter.Tk()
main_window.title("OCR")
main_window.geometry("600x450")

source_directory = tkinter.StringVar()
source_directory.set("")

row = 0

#file label
file_label = tkinter.Label(main_window, text="<no directory selected>")
file_label.grid(row=row, column=0)

#file select
file_select = tkinter.Button(main_window, text="Select Directory", command=select_directory).grid(row=row, column=1)

# row += 1
##write text
# write_text = tkinter.IntVar()
# write_text_checkbutton = tkinter.Checkbutton(main_window, text="Write text file", variable=write_text, onvalue=1, offvalue=0, height=2)
# write_text_checkbutton.grid(row=row, column= 0)

# write_text_checkbutton.select()

# row += 1
##write MS Word Doc
# write_ms_word_doc = tkinter.IntVar()
# write_ms_word_doc_checkbutton = tkinter.Checkbutton(main_window, text="Write MS Word Doc", variable=write_ms_word_doc, onvalue=1, offvalue=0, height=2)
# write_ms_word_doc_checkbutton.grid(row=row, column= 0)


# row += 1
# #Rotate Input Image
# rotate = tkinter.IntVar()
# rotate_checkbutton = tkinter.Checkbutton(main_window, text="Rotate Image 180\u00B0", variable=rotate, onvalue=1, offvalue=0, height=2)
# rotate_checkbutton.grid(row=row, column= 0)

# rotate_checkbutton.select()

row += 1
#run
tkinter.Button(main_window, text="Run OCR", command=run_ocr).grid(row=row,column=0, pady=10)
row += 1

#progress bar
progress_bar = ttk.Progressbar(main_window, orient=tkinter.HORIZONTAL, length=400, mode="determinate", maximum=100, value=0)
progress_bar.grid(row=row, column=0, columnspan=2)

row += 1
#log
log_area = tkinter.scrolledtext.ScrolledText(main_window, height=15, width=90, pady=10)
log_area.grid(row=row,column=0, columnspan=2)

log_area.insert(tkinter.END, "Initializing\n")

#file_label.pack()
#file_select.pack()

#file selection



main_window.mainloop()


