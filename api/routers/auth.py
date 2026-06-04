from api.services.auth import AuthService
from api.models.auth import SignUp,SignIn
from fastapi import APIRouter

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
async def get_user_details(email: str):
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