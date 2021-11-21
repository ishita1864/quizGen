#!/usr/bin/env python
# coding: utf-8


# pip install PyPDF2

# pip install pdfminer.six

from io import StringIO

from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfparser import PDFParser




# output_string = StringIO()


# # In[7]:


# with open('LiteratureSurveyFinal.pdf', 'rb') as in_file:
#     parser = PDFParser(in_file)
#     doc = PDFDocument(parser)
#     rsrcmgr = PDFResourceManager()
#     device = TextConverter(rsrcmgr, output_string, laparams=LAParams())
#     interpreter = PDFPageInterpreter(rsrcmgr, device)
#     for page in PDFPage.create_pages(doc):
#         interpreter.process_page(page)


# # In[9]:


# x= output_string.getvalue()


# # In[10]:


# print(x)


# # In[11]:


# text_file = open('new.txt', 'w')
# n = text_file.write(x)
# text_file.close()


# # In[ ]:




def pdfconv():
    output_string = StringIO()

    with open('nature.pdf', 'rb') as in_file:
        parser = PDFParser(in_file)
        doc = PDFDocument(parser)
        rsrcmgr = PDFResourceManager()
        device = TextConverter(rsrcmgr, output_string, laparams=LAParams())
        interpreter = PDFPageInterpreter(rsrcmgr, device)
        for page in PDFPage.create_pages(doc):
            interpreter.process_page(page)

        x= output_string.getvalue()

        text_file = open('new.txt', 'w')
        n = text_file.write(x)
        text_file.close()

        return x


