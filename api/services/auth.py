from fastapi import HTTPException
from api.models.auth import SignUp, SignIn, ResetPassword
from database import insert_one,find_one,update_one,logger
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
        email=data.email.lower()
        try:
            user_data = find_one("users", {"email": email})
            if user_data:
                raise HTTPException(status_code=400, detail="Email already exists")
        except Exception as e:
            logger.error("sign_up failed on '%s': %s", email, e)
            raise HTTPException(status_code=400, detail="Sign up failed")
        password_hash = hash_password(data.password)
        user_data={
            "email":data.email,
            "password_hash":password_hash,
            "first_name":data.first_name,
            "last_name":data.last_name,
        }
        results=insert_one("users",user_data)
        if results and "id" in results:
            from utils.user_validation import create_access_token
            token = create_access_token({"sub": results["id"], "email": results["email"]})
            results = dict(results)
            results["token"] = token
        return results
    
    def login(self,data:SignIn):
        try:
            user_data = find_one("users", {"email": data.email})
            if not user_data:
                raise HTTPException(status_code=400, detail="Please sign up")
            if not verify_password(data.password, user_data["password_hash"]):
                raise HTTPException(status_code=400, detail="Incorrect password")
            
            from utils.user_validation import create_access_token
            token = create_access_token({"sub": user_data["id"], "email": user_data["email"]})
            user_data = dict(user_data)
            user_data["token"] = token
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

    def reset_password(self, data: ResetPassword):
        try:
            user_data = find_one("users", {"email": data.email})
            if not user_data:
                raise HTTPException(status_code=404, detail="User not found")
            
            password_hash = hash_password(data.new_password)
            update_one(
                table_name="users",
                data={"password_hash": password_hash},
                record_id=user_data["id"]
            )
            return {"status": "success", "message": "Password reset successfully"}
        except Exception as e:
            logger.error("reset_password failed on '%s': %s", data.email, e)
            raise HTTPException(status_code=400, detail="Password reset failed")