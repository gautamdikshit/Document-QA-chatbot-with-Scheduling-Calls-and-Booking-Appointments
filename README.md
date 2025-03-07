# Document QA Chatbot with Scheduling Calls and Booking Appointments

This project integrates a Document Question-Answering (QA) chatbot with appointment scheduling and call booking functionalities. It processes uploaded PDFs, builds a vector store using embeddings for efficient document search, and allows users to query the document content. Additionally, it supports scheduling calls and booking appointments, sending confirmation emails, and validating input data such as phone numbers, email addresses, and appointment dates.

## Features

- **PDF Document Processing:** Upload PDFs to extract content, split into chunks, and build a vector store for searching.
- **Document Question-Answering:** Ask questions related to the content of the uploaded PDF documents, and the chatbot will provide relevant answers based on the context.
- **Appointment Scheduling:** Allows users to schedule appointments by entering details such as name, phone number, email, and appointment date.
- **Call Booking:** Allows users to schedule calls by entering details such as name, phone number, and date for the call.
- **Email Confirmation:** Sends confirmation emails for scheduled appointments with a provided email address.

## Technologies Used

- **Streamlit:** For creating the interactive web interface.
- **PyPDF2:** To extract text from PDF files.
- **LangChain:** For managing LLMs, agents, and document processing workflows.
- **Groq:** Custom LLM (ChatGroq) for handling NLP tasks.
- **FAISS:** For efficient similarity search using embeddings.
- **HuggingFace:** For embeddings with the `all-MiniLM-L6-v2` model.
- **SMTP:** For sending email confirmations.
- **dotenv:** For securely managing environment variables like email credentials.
- **JSON:** For saving and managing appointment data.


## How to Use

1. **Upload PDF Documents:** Click on the "Upload PDF Documents" button to upload one or more PDF files. The app will process them and build a vector store for document search.
   
2. **Ask Questions:** After uploading PDFs, you can ask questions related to the content. The chatbot will return answers based on the document's text.

3. **Schedule Calls or Appointments:**
    - To **schedule a call**, enter your name, phone number, and preferred date for the call.
    - To **book an appointment**, enter your name, phone number, email, and preferred appointment date.
    - After submitting, you’ll receive a confirmation message indicating the success or failure of the booking.

## Known Issues & Limitations

- After submitting the form, the details still remain populated in the form fields. Currently, there is no automatic reset or clearing of the form after submission. As a result, users need to manually clear the form fields if they wish to submit new entries. This may cause confusion if the user is unaware that the form data persists. A feature to automatically reset the form after submission could be added in future versions.

