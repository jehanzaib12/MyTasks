from fastapi import FastAPI, Request, Form, Response, Depends
from fastapi.responses import HTMLResponse, RedirectResponse
from starlette.middleware.sessions import SessionMiddleware
from starlette.status import HTTP_302_FOUND
import sqlite3
from pydantic import BaseModel
from langchain_openai import OpenAI
from langchain.chains import LLMChain
import os
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
import json

load_dotenv()  # will search for .env file in local folder and load variables

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

app = FastAPI()


"""
To run this code,

We need to give get request to CreateUserTable and CreateChatTable, and then we can Provide Signup details
After Signup, we can give a post request to Signin.
We can then give a post request to QueryBot to ask bot the query.
We can then give a get request to ShowChat to get all the queries specific to user login id

User id is maintained though session

We can give a get request to logout as it will destroy the session

Without session, you can post a request to QueryBot

"""
# Secret key for encrypting session cookies
app.add_middleware(SessionMiddleware, secret_key="123")

llm = OpenAI(model="gpt-3.5-turbo-instruct", temperature=0.7)


ice_cream_assistant_template = """
You are an ice cream assistant chatbot named "Scoopsie". Your expertise is 
exclusively in providing information and advice about anything related to ice creams. This includes flavor combinations, ice cream recipes, and general 
ice cream-related queries. You do not provide information outside of this 
scope. If a question is not about ice cream, respond with, "I specialize only in ice cream related queries." 
Question: {question} 
Answer:"""

ice_cream_assistant_prompt_template = PromptTemplate(
    input_variables=["question"], template=ice_cream_assistant_template
)
llm_chain = LLMChain(llm=llm, prompt=ice_cream_assistant_prompt_template)

conn = sqlite3.connect("user.db")  # Creates a new database file if it doesn’t exist
cursor = conn.cursor()


def query_llm(question):
    print(llm_chain.invoke({"question": question})["text"])
    return llm_chain.invoke({"question": question})["text"]


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    user = request.session.get("user")
    id = request.session.get("id")

    if user:
        return "Welcome Back"

    else:
        return "You are not logged in"

# creating the user table
@app.get("/createusertable")
def createtable():

    conn = sqlite3.connect("user.db")
    with conn:
        cursor = conn.cursor()
    cursor.execute(
        "CREATE TABLE IF NOT EXISTS User (id integer primary key autoincrement, Name TEXT,Email TEXT,Password Text,Gender TEXT)"
    )
    conn.commit()

    return "User Table Created Successfully"

# creating the chat table

@app.get("/CreateChatTable")
def create_chat_table():
    conn = sqlite3.connect("user.db")
    with conn:
        cursor = conn.cursor()
    cursor.execute(
        "CREATE TABLE IF NOT EXISTS Chat (id integer primary key autoincrement, UserId integer,UserQuery TEXT,BotAnswer TEXT)"
    )
    conn.commit()

    return "Chat Table Created Successfully"


@app.get("/ShowChat")
def ShowChat(request: Request):
    user = request.session.get("user")

    if user:
        id = request.session.get("id")
        conn = sqlite3.connect("user.db")
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("SELECT userquery, botanswer FROM Chat WHERE userid = ?", (id,))
        rows = cursor.fetchall()
        print("rows: ", rows)
        conn.close()

        # ✅ Now this works
        chat_data = [dict(row) for row in rows]
        return chat_data

    else:
        return "Please Login First"


class UserCreate(BaseModel):
    name: str
    email: str
    password: str
    reenterpassword: str
    gender: str


class Chat(BaseModel):
    userquery: str


@app.post("/query_bot")
def querybot(chat: Chat, request: Request):

    user = request.session.get("user")

    if user:

        userquery = chat.userquery

        botanswer = llm_chain.invoke({"question": userquery})["text"]

        id = request.session.get("id")

        conn = sqlite3.connect("user.db")

        with conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO Chat (UserId,UserQuery,BotAnswer) VALUES(?,?,?)",
                (id, userquery, botanswer),
            )

            return {"UserQuery": userquery, "BotAnswer": botanswer}
    else:
        return "Please login first"


@app.post("/signup")
def insertuser(user_data: UserCreate):

    name = user_data.name
    email = user_data.email

    password = user_data.password
    reenterpassword = user_data.reenterpassword

    gender = user_data.gender

    conn = sqlite3.connect("user.db")
    with conn:
        cursor = conn.cursor()

    if password == reenterpassword:

        c1 = cursor.execute("Select COUNT(*) FROM user WHERE email ='{}'".format(email))

        for row in c1.fetchall():
            if row[0] >= 1:
                print("Email address already exists")
                return "Email address already exists"
            else:
                cursor.execute(
                    "INSERT INTO User (Name,Email,Password,Gender) VALUES(?,?,?,?)",
                    (name, email, password, gender),
                )
                if cursor.rowcount > 0:
                    print("Signup Done")
                    conn.commit()

                    return "Signup Done"
                else:
                    print("Signup Error")
                    return "Signup Error"
    else:
        return "Password and Re enter password don't match"


class Userlogin(BaseModel):
    email: str
    password: str


# method to perform login
@app.post("/login")
def loginNow(user_login: Userlogin, request: Request = None):

    email = user_login.email
    password = user_login.password

    conn = sqlite3.connect("user.db")
    with conn:
        cursor = conn.cursor()
    cursor.execute("Select * from user Where Email=? AND Password=?", (email, password))

    count = None
    id = None
    for row in cursor.fetchall():
        print("row: ", row[0])
        count = row
        id = row[0]
        print("id", id)

    if count is not None:
        print("Welcome")
        request.session["user"] = email
        request.session["id"] = id

        return "Login Successfull"
    else:
        print("Login failed")
        return "Login Not Successfull"

    conn.commit()


@app.get("/logout")
def logout(request: Request):
    request.session.clear()
    request.session["user"] = None
    request.session["id"] = None

    return RedirectResponse("/", status_code=HTTP_302_FOUND)
