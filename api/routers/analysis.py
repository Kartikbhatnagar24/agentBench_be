from fastapi import APIRouter, Depends
from api.services.analysis import AnalysisService
from utils.user_validation import get_current_user

router = APIRouter(prefix="/analysis", tags=["Analysis"])
analysis_service = AnalysisService()


@router.get("/overview")
async def get_analysis_overview(user_id: str, current_user: dict = Depends(get_current_user)):
    """
    Get an analytics-ready overview of RAG evaluation health for a user.
    """
    return analysis_service.get_overview(user_id)
