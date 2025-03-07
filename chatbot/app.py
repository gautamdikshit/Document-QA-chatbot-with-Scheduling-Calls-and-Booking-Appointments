import streamlit as st
from PyPDF2 import PdfReader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.agents import Tool, initialize_agent, AgentType
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv
import os
import datetime
import re
import validators
import json
from pathlib import Path
import smtplib

load_dotenv()
user_email = os.getenv("EMAIL_ADDRESS")
user_password = os.getenv("EMAIL_PASSWORD")

class PDFHandler:
    @staticmethod
    def extract_pdf_content(pdf_files):
        try:
            document_text = ""
            for pdf in pdf_files:
                reader = PdfReader(pdf)
                for page in reader.pages:
                    document_text += page.extract_text()
            return document_text
        except Exception as error:
            st.error(f"Failed to process PDFs: {error}")
            return ""

    @staticmethod
    def split_text_into_chunks(text):
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        return splitter.split_text(text)

    @staticmethod
    def build_vector_store(chunks):
        try:
            embeddings = HuggingFaceBgeEmbeddings(model_name="all-MiniLM-L6-v2")
            vector_db = FAISS.from_texts(chunks, embedding=embeddings)
            vector_db.save_local("faiss_index")
            return True
        except Exception as error:
            st.error(f"Vector database creation failed: {error}")
            return False

class ChatTools:
    def __init__(self):
        self.llm = ChatGroq(model="Llama3-8b-8192", temperature=0.3)
        
    def validate_email(self, email: str) -> bool:
        return validators.email(email.strip())
    
    def validate_phone(self, phone: str) -> bool:
        phone = phone.strip().replace(" ", "")
        return bool(re.match(r'^\+?\d{10}$', phone))
    
    def parse_date(self, query: str) -> str:
        try:
            prompt = f"""Today is {datetime.datetime.now().strftime('%Y-%m-%d')}.
            Convert this date query: "{query}" into YYYY-MM-DD format.
            Return only the date, nothing else."""
            return self.llm.predict(prompt).strip()
        except:
            return "Invalid date format"
    
    def send_email_confirmation(self, email_details: str) -> str:
        try:
            content = json.loads(email_details)
            if not all([user_email, user_password]):
                return "Email credentials are missing"

            msg = MIMEMultipart()
            msg['From'] = user_email
            msg['To'] = content['email']
            msg['Subject'] = content['subject']
            msg.attach(MIMEText(content['body'], 'plain'))

            with smtplib.SMTP('smtp.gmail.com', 587) as server:
                server.starttls()
                server.login(user_email, user_password)
                server.send_message(msg)
            return "Email successfully sent"
        except Exception as error:
            return f"Email sending failed: {str(error)}"

    def save_appointment(self, details: dict) -> bool:
        try:
            file_path = Path("appointments.json")
            
            if file_path.exists():
                with open(file_path, "r") as file:
                    appointments = json.load(file)
            else:
                appointments = []

            appointments.append(details)

            with open(file_path, "w") as file:
                json.dump(appointments, file, indent=4)
            
            return True
        except Exception as e:
            st.error(f"Error saving appointment: {e}")
            return False
        
class ChatbotInterface:
    def __init__(self):
        self.pdf_processor = PDFHandler()
        self.tools = ChatTools()
        self.setup_agent()
        self.prepare_qa_chain()

    def setup_agent(self):
        tools = [
            Tool(name="validate_email", func=self.tools.validate_email,
                 description="Verify email format"),
            Tool(name="validate_phone", func=self.tools.validate_phone,
                 description="Verify phone number format"),
            Tool(name="parse_date", func=self.tools.parse_date,
                 description="Transform date query into YYYY-MM-DD format"),
            Tool(name="send_email_confirmation", func=self.tools.send_email_confirmation,
                 description="Send a confirmation email. Input should be a JSON string with 'email', 'subject', and 'body'.")
        ]

        self.agent = initialize_agent(
            tools,
            self.tools.llm,
            agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            handle_parsing_error = True

        )

    def prepare_qa_chain(self):
        query_template = """
        Provide a detailed response using the available context. If no relevant information is found,
        respond with: "No relevant information available." Avoid making up answers.
        Context:\n {context}\n
        Query:\n {question}\n
        Response:
        """
        self.query_prompt = PromptTemplate(template=query_template, input_variables=["context", "question"])
        self.llm = ChatGroq(model="Llama3-8b-8192", temperature=0.3)

    def process_query(self, user_query):
        try:
            if not Path("faiss_index").exists():
                return "Please upload and process a PDF file first."
            
            embeddings = HuggingFaceBgeEmbeddings(model_name="all-MiniLM-L6-v2")
            database = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
            retrieved_docs = database.similarity_search(user_query)
            
            qa_chain = load_qa_chain(self.llm, chain_type="stuff", prompt=self.query_prompt)
            response = qa_chain(
                {"input_documents": retrieved_docs, "question": user_query}, 
                return_only_outputs=True
            )
            return response["output_text"]
        except Exception as error:
            return f"Error processing your question: {str(error)}"
    
    def schedule_call(self, name, phone, date_query):
        try:
            formatted_date = self.tools.parse_date(date_query)
            if "Invalid" in formatted_date:
                return {"success": False, "message": "Invalid date format"}
            
            # Agent validation and processing
            task_prompt = f"""Process this appointment:
            Name: {name}
            Phone: {phone}
            Date: {formatted_date}

            Steps:
            1. Validate the phone number
            """

            agent_response = self.agent.run(task_prompt)

            # If no errors, save the call details
            if "error" not in agent_response.lower():
                call_details = {
                    "name": name,
                    "phone": phone,
                    "call_date": formatted_date
                }
                if self.tools.save_appointment(call_details):
                    return {"success": True, "message": f"Call scheduled successfully for {formatted_date}"}
                
            return {"success": False, "message": agent_response}

        except Exception as error:
            return {"success": False, "message": str(error)}

    def book_appointment(self, name, phone, email, date_query):
        try:
            formatted_date = self.tools.parse_date(date_query)
            if "Invalid" in formatted_date:
                return {"success": False, "message": "Invalid date format"}

            # Email details
            email_details = json.dumps({
                "email": email,
                "subject": "Appointment Confirmation",
                "body": f"Dear {name},\n\nThis email confirms your appointment on {formatted_date}.\n\nThank you,\nThe Appointment Team"
            })

            # Agent validation and processing
            task_prompt = f"""Process this appointment:
            Name: {name}
            Email: {email}
            Phone: {phone}
            Date: {formatted_date}

            Steps:
            1. Validate the email address
            2. Validate the contact number
            3. Dispatch a confirmation email with these details: {email_details}"""

            agent_response = self.agent.run(task_prompt)
            
            # If successful, save the appointment details
            if "error" not in agent_response.lower():
                appointment_details = {
                        "name": name,
                        "phone": phone,
                        "email": email,
                        "appointment_date": formatted_date
                }
                if self.tools.save_appointment(appointment_details):
                    return {"success": True, "message": f"Appointment booked successfully for {formatted_date}"}
            
            return {"success": False, "message": agent_response}
        except Exception as error:
            return {"success": False, "message": str(error)}


def main():
    st.set_page_config(page_title="AI Chatbot with Scheduling and booking appointment", layout="wide")
    st.header("Documnent QA chatbot with Scheduling calls and making appointent")

    chatbot = ChatbotInterface()

    pdf_files = st.file_uploader("Upload PDF Documents", accept_multiple_files=True)
    if st.button("Process Documents"):
        if pdf_files:
            with st.spinner("Extracting and processing..."):
                raw_text = chatbot.pdf_processor.extract_pdf_content(pdf_files)
                if raw_text:
                    text_chunks = chatbot.pdf_processor.split_text_into_chunks(raw_text)
                    if chatbot.pdf_processor.build_vector_store(text_chunks):
                        st.success("Documents successfully processed!")
        else:
            st.error("Please upload a PDF file first.")
            
    
    # main chat interface
    user_question = st.text_input("Enter your question:")
    if user_question:
        if any(keyword in user_question.lower() for keyword in ["make appointment", "schedule meeting", "book meeting", "book"]):
            # Appointment booking flow
            st.write("Let's schedule your appointment!")
            name = st.text_input("Name:")
            phone = st.text_input("Phone Number:")
            email = st.text_input("Email:")
            date_query = st.text_input("When would you like to schedule? (e.g., 'next Monday')")

            if st.button("Book Appointment"):
                if all([name, phone, email, date_query]):
                    with st.spinner("Processing appointment..."):
                        result = chatbot.book_appointment(name, phone, email, date_query)
                        if result["success"]:
                            st.success(result["message"])
                        else:
                            st.error(result["message"])
                else:
                    st.error("Please fill in all fields.")

        elif any(keyword in user_question.lower() for keyword in ["call me", "schedule call", "book call", "call"]):
            # Appointment booking flow
            st.write("Let's schedule your call!")
            name = st.text_input("Name:")
            phone = st.text_input("Phone Number:")
            date_query = st.text_input("When would you like to schedule? (e.g., 'next Monday')")

            if st.button("Book Call"):
                if all([name, phone, date_query]):
                    with st.spinner("Processing appointment..."):
                        result = chatbot.schedule_call(name, phone, date_query)
                        if result["success"]:
                            st.success(result["message"])
                        else:
                            st.error(result["message"])
                else:
                    st.error("Please fill in all fields.")
        else:
            # PDF query handling
            with st.spinner("Processing your question..."):
                response = chatbot.process_query(user_question)
                st.write(response)

if __name__ == "__main__":
    main()
