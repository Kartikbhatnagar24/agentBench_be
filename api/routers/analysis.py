from fastapi import APIRouter

from api.services.analysis import AnalysisService

router = APIRouter(prefix="/analysis", tags=["Analysis"])
analysis_service = AnalysisService()


@router.get("/overview")
async def get_analysis_overview(user_id: str):
    """
    Get an analytics-ready overview of RAG evaluation health for a user.
    """
    return analysis_service.get_overview(user_id)
