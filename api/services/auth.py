from fastapi import HTTPException
from api.models.auth import SignUp
from api.models.auth import SignIn
from database import insert_one,find_one,logger
import bcrypt

def hash_password(password: str):
    """Hash a password."""
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')

def verify_password(password: str, hashed_password: str) -> bool:
    """Verify a password directly using bcrypt."""
    return bcrypt.checkpw(password.encode('utf-8'), hashed_password.encode('utf-8'))

class AuthService:

    def sign_up(self,data:SignUp):
        password_hash = hash_password(data.password)
        user_data={
            "email":data.email,
            "password_hash":password_hash,
            "first_name":data.first_name,
            "last_name":data.last_name,
        }
        results=insert_one("users",user_data)
        return results
    
    def login(self,data:SignIn):
        try:
            user_data = find_one("users", {"email": data.email})
            if not user_data:
                raise HTTPException(status_code=400, detail="Please sign up")
            if not verify_password(data.password, user_data["password_hash"]):
                raise HTTPException(status_code=400, detail="Incorrect password")
            return user_data
        except Exception as e:
            logger.error("login failed on '%s': %s", data.email, e)
            raise HTTPException(status_code=400, detail="Login failed")
    def get_user_details(self, email: str):
        try:
            user_data = find_one("users", {"email": email})
            if not user_data:
                raise HTTPException(status_code=404, detail="User not found")
            return user_data
        except Exception as e:
            logger.error("get_user_details failed on '%s': %s", email, e)
            raise