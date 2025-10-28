from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
import os
import logging

from services.devdockservice import DevDockService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/devdock", tags=["devdock"])

class VerifyRequest(BaseModel):
    payload: Dict[str, Any]

@router.post("/verify")
async def verify(payload: VerifyRequest):
    try:
        service = DevDockService(api_key=os.getenv('DEVDOCK_API_KEY'))
        result = service.verify_credential(payload.payload)
        return result
    except Exception as e:
        logger.error(f"DevDock verify failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


