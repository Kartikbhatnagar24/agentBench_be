from pydantic import BaseModel

class SignUp(BaseModel):
    email:str 
    password:str
    first_name:str
    last_name:str

class SignIn(BaseModel):
    email:str 
    password:str

class ResetPassword(BaseModel):
    email: str
    new_password: str
    