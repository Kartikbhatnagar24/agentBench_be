from api.services.auth import AuthService
from api.models.auth import SignUp, SignIn, ResetPassword
from fastapi import APIRouter, Depends
from utils.user_validation import get_current_user

router=APIRouter(prefix="/auth",tags=["Auth"])

auth_service = AuthService()

@router.post("/signup")
async def signup(user:SignUp):
    """
    Register a new user in the system.

    Expected Return (JSON dict):
    {
        "id": "user-uuid-string",
        "email": "user@example.com",
        "password_hash": "$2b$12$...",
        "first_name": "Firstname",
        "last_name": "Lastname"
    }
    """
    return auth_service.sign_up(data=user)

@router.post("/signin")
async def signin(body:SignIn):    
    """
    Authenticate a user with email and password.

    Expected Return (JSON dict):
    {
        "id": "user-uuid-string",
        "email": "user@example.com",
        "password_hash": "$2b$12$...",
        "first_name": "Firstname",
        "last_name": "Lastname"
    }
    """
    return auth_service.login(data=body)

@router.get("/user-details")
async def get_user_details(email: str, current_user: dict = Depends(get_current_user)):
    """
    Retrieve database profile details for a given email.

    Expected Return (JSON dict):
    {
        "id": "user-uuid-string",
        "email": "user@example.com",
        "password_hash": "$2b$12$...",
        "first_name": "Firstname",
        "last_name": "Lastname"
    }
    """
    return auth_service.get_user_details(email=email)

@router.post("/reset-password")
async def reset_password(body: ResetPassword):
    """
    Directly update a user's password in the database (stateless/unauthenticated).
    """
    return auth_service.reset_password(data=body)